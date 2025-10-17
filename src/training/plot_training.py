import json
import matplotlib.pyplot as plt
import argparse


def plot_training_history(history_path: str = "./training_history.json"):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_ndcg = history['val_ndcg']
    val_recall = history['val_recall']
    val_mrr = history['val_mrr']
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    plt.tight_layout()
    
    loss_plot_path = "./training_loss.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    print(f"Training loss plot saved to: {loss_plot_path}")
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(epochs, val_ndcg, 'r-o', label='NDCG@10', linewidth=2, markersize=8)
    ax2.plot(epochs, val_recall, 'g-s', label='RECALL@10', linewidth=2, markersize=8)
    ax2.plot(epochs, val_mrr, 'm-^', label='MRR@10', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1])
    plt.tight_layout()
    
    metrics_plot_path = "./validation_metrics.png"
    plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
    print(f"Validation metrics plot saved to: {metrics_plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("--history", type=str, default="./training_history.json", help="Path to training history JSON")
    args = parser.parse_args()
    
    plot_training_history(args.history)
