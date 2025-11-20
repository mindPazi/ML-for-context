from datetime import datetime
from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers.evaluation import SimilarityFunction
from torch.utils.data import DataLoader
import torch
import json

from src.training.config import TrainingConfig
from src.training.data_preparation import prepare_data
from src.evaluation.cosqa_loader import CoSQALoader
import argparse


class EarlyStoppingCallback:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float, epoch: int, steps: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            print(f"      ✓ New best score: {score:.4f}")
        else:
            self.counter += 1
            print(f"      No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                print(f"\n      Early stopping triggered at epoch {epoch}")
                self.should_stop = True
                return True

        return False


def train_model(config: TrainingConfig = None):
    if config is None:
        config = TrainingConfig()

    print("\n" + "=" * 60)
    print("         FINE-TUNING EMBEDDING MODEL ON COSQA")
    print("=" * 60)

    print("\n[Configuration]")
    print(f"  Base Model:        {config.base_model_name}")
    print(f"  Output Path:       {config.output_model_path}")
    print(f"  Device:            {config.device}")
    print(f"  Batch Size:        {config.batch_size}")
    print(f"  Epochs:            {config.num_epochs}")
    print(f"  Learning Rate:     {config.learning_rate}")
    print(f"  Max Seq Length:    {config.max_seq_length}")
    print(f"  Early Stop Patience: {config.patience}")

    train_examples, corpus, val_queries, qrels = prepare_data(config)

    print("\n[Model Initialization]")
    print(f"Loading base model: {config.base_model_name}...")
    model = SentenceTransformer(config.base_model_name, device=config.device)
    model.max_seq_length = config.max_seq_length
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    print(f"  Max seq length: {model.max_seq_length}")

    print("\n[DataLoader Setup]")
    train_dataloader = DataLoader(
        train_examples,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None,
        pin_memory=False,
    )
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")

    train_loss = losses.MultipleNegativesRankingLoss(model)
    print("\n[Loss Function]")
    print("  Using: MultipleNegativesRankingLoss")

    print("\n[Validation Setup]")
    val_queries_dict = {q["query_id"]: q["query_text"] for q in val_queries}
    val_qrels_subset = {
        qid: docs for qid, docs in qrels.items() if qid in val_queries_dict
    }
    corpus_dict = {i: doc for i, doc in enumerate(corpus)}

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=val_queries_dict,
        corpus=corpus_dict,
        relevant_docs=val_qrels_subset,
        name="validation",
        show_progress_bar=True,
        score_function=SimilarityFunction.MANHATTAN,
    )
    print(f"  Validation queries: {len(val_queries_dict)}")
    print(f"  Validation corpus: {len(corpus_dict)}")

    print("\n[Training]")
    print("=" * 60)

    total_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    early_stopping = EarlyStoppingCallback(patience=config.patience)

    history = {
        "train_loss": [],
        "val_ndcg": [],
        "val_recall": [],
        "val_mrr": [],
        "epochs": [],
    }

    best_score = -1
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 0 else 0,
            output_path=None,
            optimizer_params={"lr": config.learning_rate},
            show_progress_bar=True,
            use_amp=config.use_amp,
        )

        if hasattr(model, "_trainer") and hasattr(model._trainer.state, "log_history"):
            for entry in reversed(model._trainer.state.log_history):
                if "train_loss" in entry:
                    epoch_loss = entry["train_loss"]
                    break
            else:
                epoch_loss = 0.0
        else:
            epoch_loss = 0.0

        history["train_loss"].append(epoch_loss)

        if config.device == "mps":
            torch.mps.empty_cache()

        print(f"\n  Validating...")
        val_result = evaluator(model, output_path=None, epoch=epoch, steps=-1)

        if isinstance(val_result, dict):
            val_score = val_result.get("validation_manhattan_ndcg@10", 0.0)
            val_recall = val_result.get("validation_manhattan_recall@10", 0.0)
            val_mrr = val_result.get("validation_manhattan_mrr@10", 0.0)
            print(f"  Validation NDCG@10: {val_score:.4f}")
            print(f"  Validation RECALL@10: {val_recall:.4f}")
            print(f"  Validation MRR@10: {val_mrr:.4f}")
        else:
            val_score = val_result
            val_recall = 0.0
            val_mrr = 0.0
            print(f"  Validation score: {val_score:.4f}")

        history["epochs"].append(epoch + 1)
        history["val_ndcg"].append(val_score)
        history["val_recall"].append(val_recall)
        history["val_mrr"].append(val_mrr)

        if val_score > best_score:
            best_score = val_score
            print(f"  Saving best model (score: {val_score:.4f})...")
            model.save(config.output_model_path)

        if config.device == "mps":
            torch.mps.empty_cache()
            print(f"  Memory cleared (MPS)")

        if early_stopping(val_score, epoch + 1, -1):
            break

    history_path = "./training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Model saved to: {config.output_model_path}")
    print(f"Training history saved to: {history_path}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune embedding model on CoSQA")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)

    args = parser.parse_args()

    config = TrainingConfig()
    if args.base_model is not None:
        config.base_model_name = args.base_model
    if args.output_path is not None:
        config.output_model_path = args.output_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.patience is not None:
        config.patience = args.patience

    train_model(config)
