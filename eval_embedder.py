# eval_embedder.py
# ----------------------------------------------------------
# Evaluation for ImageStyleEmbedder on CDD-11 (12 Classes)
# ----------------------------------------------------------
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import glob
import json
from torch.utils.data import DataLoader
from model.embedder import build_embedder, ImageStyleEmbedder 
from utils.dataset_loader import CDD11Dataset, collate_fn 
from sklearn.metrics import (
    f1_score, confusion_matrix, silhouette_score, average_precision_score,
    ConfusionMatrixDisplay
)
from sklearn.manifold import TSNE

# Global bold and clear styling for paper-ready figures
plt.rcParams.update({
    'font.weight': 'bold', 
    'axes.labelweight': 'bold',
    'figure.autolayout': True
})

DEFAULT_EMBEDDING_DIM = 512

# CDD-11 Labels (exactly 12, excluding 'night')
WADNET_LABELS = [
    'clear', 'snow', 'haze', 'rain', 'low', 'haze_rain', 
    'haze_snow', 'low_haze', 'low_rain', 'low_snow', 
    'low_haze_rain', 'low_haze_snow'
]

# --- Metric Calculation Functions ---

def compute_recall_at_1(similarity_matrix, labels):
    n = similarity_matrix.shape[0]
    correct = 0
    labels = np.array(labels)
    for i in range(n):
        sim = similarity_matrix[i].copy()
        sim[i] = -np.inf
        top_index = np.argmax(sim)
        if labels[top_index] == labels[i]:
            correct += 1
    return float(correct) / n

def compute_precision_at_k(sim_matrix, labels, k=5):
    labels = np.array(labels)
    n = sim_matrix.shape[0]
    total_correct = 0
    for i in range(n):
        sim = sim_matrix[i].copy()
        sim[i] = -np.inf
        top_k_indices = sim.argsort()[-k:][::-1]
        if any(labels[idx] == labels[i] for idx in top_k_indices):
            total_correct += 1
    return total_correct / n

def compute_mean_reciprocal_rank(sim_matrix, labels):
    labels = np.array(labels)
    n = sim_matrix.shape[0]
    reciprocal_ranks = []
    for i in range(n):
        sim = sim_matrix[i].copy()
        sim[i] = -np.inf
        ranked_indices = sim.argsort()[::-1]
        rank = 1
        for idx in ranked_indices:
            if labels[idx] == labels[i]:
                reciprocal_ranks.append(1.0 / rank)
                break
            rank += 1
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_map(sim_matrix, labels):
    labels = np.array(labels)
    n = sim_matrix.shape[0]
    average_precisions = []
    for i in range(n):
        mask = np.arange(n) != i
        y_true = (labels[mask] == labels[i]).astype(int)
        y_score = sim_matrix[i][mask]
        if y_true.sum() == 0: continue
        ap = average_precision_score(y_true, y_score)
        average_precisions.append(ap)
    return np.mean(average_precisions) if average_precisions else 0.0

def compute_ndcg_at_k(sim_matrix, labels, k=5):
    labels = np.array(labels)
    n = sim_matrix.shape[0]
    ndcgs = []
    for i in range(n):
        sim = sim_matrix[i].copy()
        sim[i] = -np.inf
        ranked_indices = sim.argsort()[::-1]
        rel = (labels[ranked_indices] == labels[i]).astype(int)[:k]
        gains = (2 ** rel) - 1
        discounts = np.log2(np.arange(2, len(gains) + 2)) 
        dcg = np.sum(gains / discounts)
        ideal_rel = np.sort(rel)[::-1]
        ideal_gains = (2 ** ideal_rel) - 1
        ideal_dcg = np.sum(ideal_gains / discounts)
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcgs.append(ndcg)
    return np.mean(ndcgs)

# --- Plotting Functions ---

def confusion_matrix_with_labels(true_labels, pred_labels, class_labels, out_dir):
    num_classes = len(class_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    
    # Bold plot with high contrast
    disp.plot(xticks_rotation=45, cmap='Blues', values_format='d', ax=ax, colorbar=False)
    ax.set_title("Weather Semantic Classification Confusion Matrix", fontsize=20, fontweight='bold', pad=25)
    ax.set_xlabel("Predicted Weather Category", fontsize=16, fontweight='bold')
    ax.set_ylabel("Ground Truth Weather Category", fontsize=16, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight='bold')
    
    for text in disp.text_.ravel():
        text.set_fontsize(11)
        text.set_fontweight('bold')

    plt.tight_layout()
    save_path = os.path.join(out_dir, "confusion_matrix_final.png")
    plt.savefig(save_path, dpi=300)
    print(f" Confusion Matrix saved to: {save_path}")
    plt.close()

def save_embeddings_batches(embedder, dataloader, device, out_dir, class_labels):
    os.makedirs(out_dir, exist_ok=True)
    batch_files, label_files = [], []
    all_pred_labels, true_all_labels = [], [] 
    
    # Save text prototypes (anchors) for stage 2 similarity classification
    if hasattr(embedder, 'text_embeddings'):
        np.save(os.path.join(out_dir, "text_prototypes.npy"), embedder.text_embeddings.detach().cpu().numpy())

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch['input'].to(device)
            emb = embedder(x).cpu().numpy()
            y = batch['wid'].cpu().numpy()
            
            emb_path = os.path.join(out_dir, f'embeddings_batch_{batch_idx}.npy')
            label_path = os.path.join(out_dir, f'labels_batch_{batch_idx}.npy')
            np.save(emb_path, emb)
            np.save(label_path, y)
            batch_files.append(emb_path)
            label_files.append(label_path)
            true_all_labels.extend(y)

            if hasattr(embedder, 'classify'):
                logits = embedder.classify(x) 
                top1 = logits.argmax(dim=1).cpu().numpy()
                all_pred_labels.extend(top1)
            
    if all_pred_labels:
        true_all = np.array(true_all_labels)
        pred_all = np.array(all_pred_labels)
        acc = (true_all == pred_all).mean()
        print(f"🚀 Batch-based Accuracy: {acc:.4f}")
        
    return batch_files, label_files

def eval_stage2_metrics(all_embs, all_labels, class_labels, out_dir):
    # Normalize for cosine similarity metrics
    norm_embs = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True).clip(min=1e-8)
    sim_matrix = np.dot(norm_embs, norm_embs.T)
    
    # Classification using saved text prototypes
    proto_path = os.path.join(out_dir, "text_prototypes.npy")
    if os.path.exists(proto_path):
        protos = np.load(proto_path)
        norm_protos = protos / np.linalg.norm(protos, axis=1, keepdims=True).clip(min=1e-8)
        preds = np.argmax(np.dot(norm_embs, norm_protos.T), axis=1)
        confusion_matrix_with_labels(all_labels, preds, class_labels, out_dir)

    # Compute Detailed Retrieval Metrics
    metrics = {
        "Recall@1": compute_recall_at_1(sim_matrix, all_labels),
        "Precision@5": compute_precision_at_k(sim_matrix, all_labels, k=5),
        "MRR": compute_mean_reciprocal_rank(sim_matrix, all_labels),
        "mAP": compute_map(sim_matrix, all_labels),
        "NDCG@5": compute_ndcg_at_k(sim_matrix, all_labels, k=5),
        "Silhouette": silhouette_score(norm_embs, all_labels, metric='cosine')
    }

    print("\n--- Retrieval & Clustering Metrics ---")
    for k, v in metrics.items(): print(f"{k}: {v:.4f}")

    # TSNE Plot with Bold Terminology
    print("--- Generating Bold t-SNE Plot ---")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine', n_jobs=-1)
    emb_2d = tsne.fit_transform(norm_embs)
    
    plt.figure(figsize=(15, 11))
    unique_labels = np.unique(all_labels)
    for i in unique_labels:
        mask = (all_labels == i)
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=class_labels[i], alpha=0.75, s=60, edgecolors='white')

    plt.title("Latent Space Visualization of Weather Semantics", fontsize=24, fontweight='bold', pad=25)
    plt.xlabel("Latent Dimension 1", fontsize=18, fontweight='bold')
    plt.ylabel("Latent Dimension 2", fontsize=18, fontweight='bold')
    
    plt.legend(title="Weather Styles", title_fontsize=16, fontsize=14, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.savefig(os.path.join(out_dir, 'tsne_latent_space_bold.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cdd_test_root', type=str, required=False)
    parser.add_argument('--checkpoint', type=str, required=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--backbone', type=str, default="resnet50")
    # Updated default directory name to embedder_eval_out
    parser.add_argument('--emb_out_dir', type=str, default="embedder_eval_out")
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()

    class_labels = WADNET_LABELS 
    
    if args.stage == 1:
        if not args.checkpoint or not args.cdd_test_root:
            parser.error("--stage 1 requires --checkpoint and --cdd_test_root")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = args.checkpoint.rsplit('.', 1)[0] + ".json" 
        with open(config_path, 'r') as f: config = json.load(f)
        
        model = build_embedder(backbone=args.backbone, out_dim=config.get("embedding_dim", 512)).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        
        dataset = CDD11Dataset(args.cdd_test_root, split="test", normalize=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        save_embeddings_batches(model, loader, device, args.emb_out_dir, class_labels)
        print(f" Stage 1 Complete. Batches saved in {args.emb_out_dir}")

    elif args.stage == 2:
        emb_files = sorted(glob.glob(f"{args.emb_out_dir}/embeddings_batch_*.npy"))
        label_files = sorted(glob.glob(f"{args.emb_out_dir}/labels_batch_*.npy"))
        
        if not emb_files:
            print(f"Error: No embeddings found in {args.emb_out_dir}. Run --stage 1 first.")
            return

        all_embs = np.concatenate([np.load(f) for f in emb_files])
        all_labels = np.concatenate([np.load(f) for f in label_files])
        eval_stage2_metrics(all_embs, all_labels, class_labels, args.emb_out_dir)
        print(f"Stage 2 Complete. Plots saved in {args.emb_out_dir}")

if __name__ == "__main__": main()