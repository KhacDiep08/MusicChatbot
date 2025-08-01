import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


class MoodModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4, learning_rate=2e-5):
        super(MoodModel, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate

        # Load backbone model từ HuggingFace
        self.bert = AutoModel.from_pretrained(model_name)

        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(0.3)

        # Layer cuối cùng dùng để phân loại
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # Đầu ra từ mô hình BERT backbone
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]  # lấy CLS token: (batch_size, hidden_size)

        # Áp dụng dropout + classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        # Nếu truyền nhãn thì tính loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }

    def compute_metrics(self, preds, labels):
        """
        Truyền vào các logits và labels thật, trả về dict chứa các chỉ số.
        """
        preds_class = torch.argmax(preds, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        acc = accuracy_score(labels, preds_class)
        f1 = f1_score(labels, preds_class, average='weighted')
        return {
            'accuracy': acc,
            'f1_score': f1
        }

    def get_optimizer(self):
        """
        Trả về AdamW optimizer cho mô hình.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
