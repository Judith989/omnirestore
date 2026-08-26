#!/usr/bin/env python3
# train_embedder.py
# -------------------------------------------------------------------------------------------------
# Training loop for the new ImageStyleEmbedder.
# Includes parameter counting and logging.
# -------------------------------------------------------------------------------------------------

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import csv
import os
import json 
from datetime import datetime
from itertools import product
from utils.dataset_loader import load_combined_dataset
from model.embedder import build_embedder

# --- Hyperparameter for Embedding Dimension ---
# NOTE: This must match the out_dim used in build_embedder.
EMBEDDING_DIM = 512

#  NEW FUNCTION: Calculate total trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_embedder(
    model, train_loader: DataLoader, val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau, 
    device: torch.device,
    epochs: int,
    class_loss_weight: float,
    contrastive_weight: float,
    temperature: float,
    csv_log_path: str,
    run_config: dict,
    early_stop_patience: int = 10
):
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    
    # Save run config to a separate JSON file for clean logging
    run_config_path = csv_log_path.replace(".csv", ".json")
    with open(run_config_path, 'w') as f:
        json.dump(run_config, f, indent=4)
    print(f"Saved run configuration to {run_config_path}")

    with open(csv_log_path, "w", newline='') as csvfile:
        fieldnames = [
            "epoch", "train_total_loss", "train_class_loss", "train_contrastive_loss", "train_acc",
            "val_total_loss", "val_class_loss", "val_contrastive_loss", "val_acc", "current_lr" 
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        use_contrastive = contrastive_weight > 0.0

        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]['lr']
            
            model.train()
            train_total_loss_accum = 0.0
            class_losses = []
            contrastive_losses = [] 
            accs = []
            
            # TQDM on train loader
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
                imgs = batch['input'].to(device)
                labels = batch['wid'].to(device)
                
                max_label_index = model.text_embeddings.shape[0] - 1
                assert (labels >= 0).all() and (labels <= max_label_index).all(), (
                    f"Label indices out of range! Labels min: {labels.min().item()}, max: {labels.max().item()}, "
                    f"expected max: {max_label_index}"
                )

                optimizer.zero_grad()
                
                total_loss, class_loss, contrastive_loss = model.total_loss(
                    imgs, labels, 
                    contrastive=use_contrastive,
                    return_components=True
                )
                
                # Backpropagate the total loss
                total_loss.backward()
                optimizer.step()
                
                # Manually calculate accuracy for logging purposes
                with torch.no_grad():
                    preds = model.classify(imgs) 
                    pred_labels = preds.argmax(dim=1)
                    acc = (pred_labels == labels).float().mean().item()

                train_total_loss_accum += float(total_loss.item())
                class_losses.append(class_loss.item())
                contrastive_losses.append(contrastive_loss.item() if use_contrastive else 0.0)
                accs.append(acc)

            avg_train_total_loss = train_total_loss_accum / len(train_loader)
            avg_class_loss = np.mean(class_losses)
            avg_contrastive_loss = np.mean(contrastive_losses)
            avg_acc = np.nanmean(accs)
            
            print(f"Epoch {epoch+1} Train Total Loss: {avg_train_total_loss:.4f} | Class Loss: {avg_class_loss:.4f} | Contrastive Loss: {avg_contrastive_loss:.4f} | Acc: {avg_acc:.4f} | LR: {current_lr:.2e}")

            # Validation loop
            model.eval()
            val_total_loss_accum = 0.0
            val_class_losses = []
            val_contrastive_losses = []
            val_acc = []

            # TQDM on val loader
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                with torch.no_grad():
                    imgs = batch['input'].to(device)
                    labels = batch['wid'].to(device)
                    
                    max_label_index = model.text_embeddings.shape[0] - 1
                    assert (labels >= 0).all() and (labels <= max_label_index).all(), (
                        f"Validation labels out of range! Labels min: {labels.min().item()}, max: {labels.max().item()}, "
                        f"expected max: {max_label_index}"
                    )
                    
                    total_loss, class_loss, contrastive_loss = model.total_loss(
                        imgs, labels, 
                        contrastive=use_contrastive, 
                        return_components=True
                    )
                    
                    val_total_loss_accum += float(total_loss.item())
                    val_class_losses.append(class_loss.item())
                    val_contrastive_losses.append(contrastive_loss.item() if use_contrastive else 0.0)
                    
                    preds = model.classify(imgs)
                    pred_labels = preds.argmax(dim=1)
                    acc = (pred_labels == labels).float().mean().item()
                    val_acc.append(acc)

            avg_val_total_loss = val_total_loss_accum / len(val_loader)
            avg_val_class_loss = np.mean(val_class_losses)
            avg_val_contrastive_loss = np.mean(val_contrastive_losses)
            avg_val_acc = np.nanmean(val_acc)
            
            print(f"Epoch {epoch+1} Val Total Loss: {avg_val_total_loss:.4f} | Class Loss: {avg_val_class_loss:.4f} | Contrastive Loss: {avg_val_contrastive_loss:.4f} | Acc: {avg_val_acc:.4f}")
            
            writer.writerow({
                "epoch": epoch + 1,
                "train_total_loss": avg_train_total_loss,
                "train_class_loss": avg_class_loss,
                "train_contrastive_loss": avg_contrastive_loss,
                "train_acc": avg_acc,
                "val_total_loss": avg_val_total_loss,
                "val_class_loss": avg_val_class_loss,
                "val_contrastive_loss": avg_val_contrastive_loss,
                "val_acc": avg_val_acc,
                "current_lr": current_lr
            })

            # Scheduler step based on the total validation loss
            scheduler.step(avg_val_total_loss)

            # Save best model using the total validation loss
            if avg_val_total_loss < best_val_loss:
                best_val_loss = avg_val_total_loss
                epochs_since_improvement = 0
                # Saving model state dictionary with a .pt extension for the embedder
                torch.save(model.state_dict(), csv_log_path.replace(".csv", ".pt")) 
                print(f"Saved best model at epoch {epoch+1} with total loss: {best_val_loss:.4f}")
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= early_stop_patience:
                print(f"Early stopping: no improvement for {early_stop_patience} epochs.")
                break

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acdc_root', type=str, default='DUMMY_PATH')
    parser.add_argument('--cdd_train_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--embedder_modes', nargs='+', type=str, default=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--logdir', type=str, default="logs")
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[1e-4,5e-4, 5e-5, 1e-5])
    parser.add_argument('--optimizers', nargs='+', type=str, default=['adamw', 'sgd'])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[8, 16, 32, 64, 256])
    parser.add_argument('--class_loss_weights', nargs='+', type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument('--contrastive_weights', nargs='+', type=float, default=[0.1, 0.2, 0.5, 0.0])
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.05, 0.07, 0.1])
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Optimizer map for cleaner initialization ---
    optimizer_map = {
        "adamw": lambda params, lr: torch.optim.AdamW(params, lr=lr, weight_decay=1e-4),
        "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    }

    # Use a dummy root for ACDC when training without it
    acdc_root_path = args.acdc_root if args.acdc_root != 'DUMMY_PATH' else None
    
    for backbone, lr, optimizer_name, batch_sz, class_w, contrast_w, temp in product(
        args.embedder_modes, args.learning_rates, args.optimizers, args.batch_sizes, 
        args.class_loss_weights, args.contrastive_weights, args.temperatures):
        
        # 1. Initialize Model
        model = build_embedder(
            backbone=backbone, 
            out_dim=EMBEDDING_DIM, 
            class_loss_weight=class_w,
            contrastive_weight=contrast_w,
            temperature=temp
        ).to(device)

        # 🟢 NEW: Calculate and print model size (Total Trainable Parameters)
        total_params = count_parameters(model)
        
        print(f"\n--- Starting Run: {backbone} | Params: {total_params/1e6:.2f}M | LR: {lr:.2e} ---")

        # 2. Load dataset loaders
        train_loader, val_loader, _, _ = load_combined_dataset(
            acdc_root=acdc_root_path, 
            cdd_train_root=args.cdd_train_root,
            batch_size=batch_sz,
            distributed=False,
        )
        
        if optimizer_name.lower() in optimizer_map:
            optimizer = optimizer_map[optimizer_name.lower()](model.parameters(), lr)
        else:
            raise ValueError("Unsupported optimizer: " + optimizer_name)
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            min_lr=1e-7,
        )

        os.makedirs(args.logdir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = (f"embedder_{backbone}_bs{batch_sz}_opt{optimizer_name}_lr{lr:.2e}_"
                    f"cw{class_w}_conw{contrast_w}_temp{temp}_{timestamp}.csv")
        csv_log_path = os.path.join(args.logdir, run_name)
        
        run_config = {
            "lr": lr, "optimizer": optimizer_name, "batch_size": batch_sz, "class_loss_weight": class_w,
            "contrastive_weight": contrast_w, "temperature": temp, "epochs": args.epochs,
            "embedder_mode": backbone, "embedding_dim": EMBEDDING_DIM, "acdc_root": args.acdc_root,
            "cdd_train_root": args.cdd_train_root, "seed": args.seed, "patience": args.patience,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
            "total_params_M": total_params / 1e6 # 🟢 LOG model size
        }

        # ---- Verify: print text and visual embeddings for the first batch ----
        print("\n==== Verifying Embedding Outputs (Sample Batch) ====")
        sample_batch = next(iter(train_loader))
        imgs = sample_batch['input'].to(device)
        labels = sample_batch['wid'].to(device)

        model.eval()
        with torch.no_grad():
            img_emb = model.embed_for_style_transfer(imgs) 
            
            print(f"Sample visual embedding shape: {img_emb.shape}")
            print(f"Sample visual embedding (first vector): {img_emb[0][:5].tolist()}...")

            sample_labels_to_print = labels[:3].tolist()
            sample_classes = [model.labels[i] for i in sample_labels_to_print]
            text_emb = model.embed_text_for_style(sample_classes)
            print(f"Sample text embedding shape: {text_emb.shape}")
            print(f"Sample text embedding (first vector for '{sample_classes[0]}'): {text_emb[0][:5].tolist()}...")

        print("==== Embedding verification complete. ====\n")

        train_embedder(
            model, train_loader, val_loader, optimizer, scheduler, device, args.epochs,
            class_loss_weight=class_w, 
            contrastive_weight=contrast_w,
            temperature=temp,
            csv_log_path=csv_log_path, 
            run_config=run_config,
            early_stop_patience=args.patience
        )
    print("Training complete.")