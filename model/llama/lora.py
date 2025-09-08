import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training


class LoraTrainer:
    def __init__(self,
                 base_model: str,
                 r: int = 64,
                 alpha: int = 128,
                 dropout: float = 0.05,
                 use_4bit: bool = True):

        self.base_model = base_model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.use_4bit = use_4bit

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization configuration
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config if use_4bit else None,
            device_map="auto",
            trust_remote_code=True
        )

        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        # In số lượng tham số có thể huấn luyện
        self.model.print_trainable_parameters()

    def preprocess_dataset(self, dataset_path: str):
        """Load và tokenize dataset JSON"""
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding=False
            )

        tokenized = dataset.map(tokenize_fn, batched=True)
        return tokenized

    def train(self, dataset_path: str, output_dir: str,
              epochs: int = 3,
              batch_size: int = 2,
              gradient_accumulation_steps: int = 4,
              lr: float = 2e-4):

        dataset = self.preprocess_dataset(dataset_path)

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=not self.use_4bit,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            output_dir=output_dir,
            save_total_limit=2,
            optim="paged_adamw_8bit" if self.use_4bit else "adamw_torch",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            report_to="none"
        )

        # Data collator để tự động padding
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✅ LoRA adapter saved to {output_dir}")



