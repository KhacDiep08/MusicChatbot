import torch
import torch.nn as nn
form transformers import DistilBertModel, DistilBertConfig

class MoodModel(nn.Module):
    def __init__(self, num_labels=4, model_name='distilbert-base-uncased'):
        super(MoodModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

     def forward(self, input_ids, attention_mask):
        # Lấy output từ DistilBERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]  # lấy token [CLS] tương đương đầu chuỗi

        # Đưa qua dropout + classifier
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)

        return logits
    