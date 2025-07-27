import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

class BuildData:
    def __init__(self, csv_path, model_name='bert-base-uncased', test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state  

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()

    def load_and_prepare_data(self):
        # Load data
        df = pd.read_csv(self.csv_path)
        
        # Clean lyrics
        df = df[['lyrics', 'mood']].dropna()

        # Encode labels
        df['label'] = self.label_encoder.fit_transform(df['mood'])
        
        # Split data into train and test sets
        train_df, val_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)

        # Convert to Hugging Face Dataset  
        train_ds = Dataset.from_pandas(train_df[['lyrics', 'label']])
        val_ds = Dataset.from_pandas(val_df[['lyrics', 'label']])

        # Tokenize the datasets
        def tokenize(batch):
            return self.tokenizer(batch['lyrics'], padding="max_length", truncation=True, max_length=512)

        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)

        # Set format for PyTorch
        train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        return DatasetDict({    
            'train': train_ds,
            'validation': val_ds
        })

    def get_label_encoder(self):
        return self.label_encoder
