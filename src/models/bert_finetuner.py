import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
from sklearn.metrics import f1_score


class BertFineTuner:
    """
    Fine-tune a HuggingFace Transformer model (e.g., DistilBERT) for duplicate question detection.
    Handles mini-dataset construction, tokenization, training, and evaluation.
    """

    def __init__(self, model_ckpt="distilbert-base-uncased", max_length=64):
        """
        Initialize the fine-tuner.
        
        Args:
            model_ckpt (str): HuggingFace model checkpoint.
            max_length (int): Maximum token length for input sequences.
        """
        self.model_ckpt = model_ckpt
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt,
            num_labels=2,
            id2label={0: "Not dup", 1: "Dup"},
            label2id={"Not dup": 0, "Dup": 1},
        )
        self.collator = DataCollatorWithPadding(self.tokenizer)
        self.accuracy = evaluate.load("accuracy")

    def df_to_hfds(self, df, q1_col="question1", q2_col="question2", label_col="is_duplicate"):
        """
        Convert DataFrame to HuggingFace Dataset for sequence classification.
        """
        df = df.rename(columns={label_col: "labels"}).copy()
        df["text"] = df[q1_col].fillna("") + " [SEP] " + df[q2_col].fillna("")
        return Dataset.from_pandas(df[["text", "labels"]])

    def tokenize_batch(self, batch):
        """
        Tokenize a batch of text examples.
        """
        return self.tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def metrics(self, eval_pred):
        """
        Compute evaluation metrics (accuracy, log loss, and F1-score) for the trainer.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = softmax(logits, axis=-1)[:, 1]  # Probability for class 1 ("Duplicate")

        acc = self.accuracy.compute(predictions=preds, references=labels)["accuracy"]
        ll = log_loss(labels, probs)
        f1 = f1_score(labels, preds)

        return {
            "accuracy": acc,
            "log_loss": ll,
            "f1": f1
        }

    def train_on_mini(self, train_df, test_df, train_size=1000, test_size=300, random_state=42, n_epochs=3):
        """
        Fine-tune the model on a small sampled dataset (for fast experiments).
        
        Args:
            train_df (pd.DataFrame): Full training data.
            test_df (pd.DataFrame): Full test data.
            train_size (int): Number of training examples.
            test_size (int): Number of test examples.
            random_state (int): Seed.
            n_epochs (int): Number of training epochs.
        """
        # Subsample for speed
        mini_train, _ = train_test_split(train_df, train_size=train_size,
                                        stratify=train_df["is_duplicate"], random_state=random_state)
        mini_test, _ = train_test_split(test_df, train_size=test_size,
                                        stratify=test_df["is_duplicate"], random_state=random_state)

        # Convert to HuggingFace Datasets
        train_ds = self.df_to_hfds(mini_train)
        test_ds = self.df_to_hfds(mini_test)

        # Tokenize datasets
        train_tok = train_ds.map(self.tokenize_batch, batched=True, remove_columns=["text"])
        test_tok = test_ds.map(self.tokenize_batch, batched=True, remove_columns=["text"])

        # Training arguments
        args = TrainingArguments(
            output_dir="distilbert-mini",
            num_train_epochs=n_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir="./logs",  # Instead of logging_strategy
            learning_rate=2e-5
            # Remove evaluation_strategy, save_strategy, logging_strategy
        )

        # HuggingFace Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=test_tok,
            data_collator=self.collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.metrics,
        )
        print("Training DistilBERT on mini subset...")
        trainer.train()
        results = trainer.evaluate()
        print("Evaluation on mini-test set:")
        print(results)
        return results
    
    def train_full(self, train_df, test_df, n_epochs=3, batch_size=16):
    # Prepare HuggingFace Datasets
        train_ds = self.df_to_hfds(train_df)
        test_ds = self.df_to_hfds(test_df)

        # Tokenize
        train_tok = train_ds.map(self.tokenize_batch, batched=True, remove_columns=["text"])
        test_tok = test_ds.map(self.tokenize_batch, batched=True, remove_columns=["text"])

        # Training arguments
        args = TrainingArguments(
            output_dir="distilbert-full",
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir="./logs",
            learning_rate=2e-5
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=test_tok,
            data_collator=self.collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.metrics,
        )

        print("Fine-tuning DistilBERT on full dataset...")
        trainer.train()

        # Save model here
        trainer.save_model("distilbert-full")

        results = trainer.evaluate()
        print("Evaluation on full test set:")
        print(results)

        return results


