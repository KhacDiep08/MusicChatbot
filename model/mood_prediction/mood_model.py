from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np


class BuildModel:
    def __init__(
        self,
        model_name='distilbert-base-uncased',
        num_labels=4,
        learning_rate=2e-5,
        output_dir='./results',
        batch_size=16,
        epochs=5,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        eval_strategy='epoch'
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.epochs = epochs

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Data collator
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Training arguments cho HuggingFace Trainer
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            save_strategy=eval_strategy,
            eval_strategy=eval_strategy,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            report_to='none'  # disable wandb
        )

    def compute_metrics(self, eval_pred):
        """
        Hàm metric truyền cho Trainer để tính accuracy, f1-score.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {
            'accuracy': acc,
            'f1': f1
        }

    def get_trainer(self, tokenized_datasets):
        """
        Trả về HuggingFace Trainer.
        tokenized_datasets: dict chứa 'train' và 'validation' (dataset đã token hóa).
        """
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )
        return trainer
