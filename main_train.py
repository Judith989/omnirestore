import os
import time
import torch
import argparse
import torch.nn as nn
from torchvision.utils import save_image as imwrite
from utils.dataset_loader import load_combined_dataset
from utils.ckpt_utils import load_restore_ckpt_with_optim, save_checkpoint
from utils.utils import print_args, adjust_learning_rate, tensor_metric, load_excel, seed_all
from utils.losses import Total_loss
from model.embedder import build_embedder
from model.wadt_net import WADNet 

# --- Helper function for list-to-tensor conversion ---
def _safely_to_device(data, device):
    """Handles lists of tensors by stacking them and safely filtering out None elements."""
    if isinstance(data, list):
        non_none_data = [item for item in data if item is not None]
        
        if not non_none_data:
            return torch.empty(0).to(device) 

        return torch.stack(non_none_data).to(device)
        
    elif isinstance(data, torch.Tensor):
        return data.to(device)
        
    return data 

# --------------------------------------------------------------------------

def main(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('✨ > Model Initialization...')
    
    # 1. Initialize and Freeze Embedder (Source of E: [B, 512])
    embedder_ckpt = torch.load(args.embedder_model_path, map_location=device)
    embedder_state = embedder_ckpt.get('state_dict', embedder_ckpt)
    embedder = build_embedder(backbone=args.type_name, out_dim=512).to(device)
    embedder.load_state_dict(embedder_state)
    for param in embedder.parameters():
        param.requires_grad = False
    embedder.eval()

    # 2. Initialize Trainable Components
    projection = nn.Linear(1024, 512).to(device) 
    bottleneck = nn.Identity().to(device) 

    # 3. WADNet restorer initialization 
    models_to_load = {
        'restorer': WADNet(channel=16, embed_dim=512), 
        'projection': projection,
        'bottleneck': bottleneck 
    }
    
    # Load checkpoint and optimizer state
    # NOTE: The 'weight_decay' argument MUST be REMOVED from the load_restore_ckpt_with_optim call
    # since that function signature does not contain it.
    restorer, projection, bottleneck, optimizer, cur_epoch = load_restore_ckpt_with_optim(
        device,
        models_to_load=models_to_load, 
        local_rank=None,
        freeze_model=False,
        ckpt_name=args.restore_model_path,
        lr=args.lr # Only arguments accepted by the utility function are passed here
    )
    
    # 🟢 CRITICAL FIX: Manually apply weight_decay to ALL parameter groups after loading/creation.
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = args.weight_decay

    # Update optimizer with trainable parameters (projection only)
    if not isinstance(bottleneck, nn.Identity) and \
       not any(param_group['params'] for param_group in optimizer.param_groups if param_group.get('label') == 'bottleneck'):
         optimizer.add_param_group({'params': bottleneck.parameters(), 'label': 'bottleneck', 'weight_decay': args.weight_decay})

    if not any(param_group['params'] for param_group in optimizer.param_groups if param_group.get('label') == 'projection'):
         optimizer.add_param_group({'params': projection.parameters(), 'label': 'projection', 'weight_decay': args.weight_decay})


    restorer = restorer.to(device)
    projection = projection.to(device)

    # Loss initialization
    loss = Total_loss(args, device=device) 
    
    print('💾 > Loading dataset...')
    
    train_loader, val_loader, _, _ = load_combined_dataset(
        acdc_root=args.acdc_root if args.acdc_root != "DUMMY_PATH" else None,
        cdd_train_root=args.cdd_train_root if args.cdd_train_root != "DUMMY_PATH" else None,
        cdd_val_ratio=0.1,
        batch_size=args.bs,
        workers=args.num_works,
        distributed=False,
        image_size=args.image_size, 
    )

    print('🚀 > Start training...')
    start_all = time.time()
    train(restorer, embedder, projection, bottleneck, optimizer, loss, cur_epoch, 
          args, train_loader, val_loader, device)
    end_all = time.time()
    print(f'Whole Training Time: {end_all - start_all:.2f}s.')


def train(restorer, embedder, projection, bottleneck, optimizer, loss, 
          cur_epoch, args, train_loader, val_loader, device):
    metrics = []
    best_psnr = float('-inf')
    epochs_no_improve = 0
    patience = 5

    for epoch in range(cur_epoch, args.epoch):
        optimizer = adjust_learning_rate(optimizer, epoch, args.adjust_lr)
        learnrate = optimizer.param_groups[-1]['lr']
        
        restorer.train()
        projection.train() 
        bottleneck.train() 

        for i, batch in enumerate(train_loader):
            pos = _safely_to_device(batch['target'], device) 
            inp = _safely_to_device(batch['input'], device) 
            
            if pos.numel() == 0 or inp.numel() == 0:
                print(f"Skipping batch {i} due to empty/None samples.")
                continue
                
            style_names = batch["source_weather"]

            # 1. Compute and fuse embeddings
            with torch.no_grad():
                text_emb = embedder.embed_text_for_style(style_names)
                img_emb = embedder.embed_for_style_transfer(inp)
            
            fused_emb = torch.cat([text_emb, img_emb], dim=-1)
            proj_512 = projection(fused_emb)
            proj_emb = bottleneck(proj_512)
            
            # 2. Forward pass through restorer
            out, feat_l, feat_m, feat_s = restorer(inp, proj_emb)

            optimizer.zero_grad()
            
            # 3. Calculate Loss
            total_loss = loss(inp, pos, inp, out, feat_l=feat_l, feat_m=feat_m, feat_s=feat_s)
            
            total_loss.backward()
            optimizer.step()

            # Metrics calculation
            mse = tensor_metric(pos, out, 'MSE', data_range=1)
            psnr = tensor_metric(pos, out, 'PSNR', data_range=1)
            ssim = tensor_metric(pos, out, 'SSIM', data_range=1)

            if (i + 1) % 10 == 0:
                print(f"[epoch {epoch + 1}][{i + 1}/{len(train_loader)}] "
                      f"lr: {learnrate:.6f} Loss: {total_loss.item():.4f} "
                      f"MSE: {mse:.4f} PSNR: {psnr:.4f} SSIM: {ssim:.4f}")

        # Validation step
        psnr_t1, ssim_t1 = test(args, restorer, embedder, projection, bottleneck, val_loader, device, epoch)
        metrics.append([psnr_t1, ssim_t1])
        print(f"[epoch {epoch + 1}] Val images PSNR1: {psnr_t1:.4f} SSIM1: {ssim_t1:.4f}")

        load_excel(metrics)

        # 4. Checkpoint Saving Logic
        if psnr_t1 > best_psnr:
            best_psnr = psnr_t1
            epochs_no_improve = 0
            
            models_to_save = {
                'restorer': restorer,
                'projection': projection,
                'bottleneck': bottleneck
            }
            
            save_checkpoint(
                models_to_save,
                optimizer,
                epoch + 1,
                os.path.join(args.save_model_path, "best.ckpt"),
                {'psnr': psnr_t1, 'ssim': ssim_t1}
            )
            print(f"New best PSNR: {best_psnr:.4f}, saving best checkpoint.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break


def test(args, restorer, embedder, projection, bottleneck, val_loader, device, epoch=-1):
    psnr_1, ssim_1, count = 0, 0, 0
    restorer.eval()
    projection.eval() 
    bottleneck.eval() 
    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for batch in val_loader:
            pos = _safely_to_device(batch['target'], device)
            inp = _safely_to_device(batch['input'], device)
            
            if pos.numel() == 0 or inp.numel() == 0:
                continue
                
            style_names = batch["source_weather"]

            text_emb = embedder.embed_text_for_style(style_names)
            img_emb = embedder.embed_for_style_transfer(inp)
            fused_emb = torch.cat([text_emb, img_emb], dim=-1)
            
            proj_512 = projection(fused_emb)
            proj_emb = bottleneck(proj_512)

            out, _, _, _ = restorer(inp, proj_emb)

            if count == 0 and (epoch % 20 == 0 or epoch == -1):
                for j in range(min(inp.shape[0], 4)):
                    fn = os.path.basename(str(batch["file"][j])).replace(".png", f"_epoch{epoch}.png")
                    imwrite(torch.cat((inp[j:j+1], out[j:j+1], pos[j:j+1]), dim=3),
                                          os.path.join(args.output, fn))

            psnr_1 += tensor_metric(pos, out, 'PSNR', data_range=1) * inp.shape[0]
            ssim_1 += tensor_metric(pos, out, 'SSIM', data_range=1) * inp.shape[0]
            count += inp.shape[0]

    return psnr_1 / max(1, count), ssim_1 / max(1, count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="WADNet Training")
    
    # Argument Definitions
    parser.add_argument("--embedder-model-path", type=str, required=True, help='embedder model path')
    parser.add_argument("--restore-model-path", type=str, default=None, help='restore model path')
    parser.add_argument("--save-model-path", type=str, required=True, help='save model directory')
    parser.add_argument("--epoch", type=int, default=300, help='number of epochs')
    parser.add_argument("--bs", type=int, default=4, help='batch size')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
    parser.add_argument("--adjust-lr", type=int, default=30, help='learning rate decay step')
    parser.add_argument("--num-works", type=int, default=4, help='worker threads')
    parser.add_argument("--acdc-root", type=str, default="DUMMY_PATH")
    parser.add_argument("--cdd-train-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help='output folder')
    parser.add_argument("--type_name", type=str, default="resnet18", help="embedder backbone")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--image_size", nargs=2, type=int, default=[224, 224], 
                             help="Image size [H, W] for model input.")

    # 🟢 CRITICAL ADDITION: Weight Decay for stabilization
    parser.add_argument("--weight-decay", type=float, default=1e-4, 
                             help='Weight decay (L2 regularization) for optimizer.')
                             
    # Loss Weights
    parser.add_argument("--loss_weight", nargs=4, type=float,
                             default=[1.0, 0.5, 0.5, 0.5], 
                             help="Weights for L1/SL1, MSSSIM, Perceptual Loss, and Feature Matching (DRL) losses")

    args = parser.parse_args()
    args.image_size = tuple(args.image_size) 
    print_args(args)
    main(args)