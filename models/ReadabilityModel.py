from torch import nn
from config import DefaultConfig

config = DefaultConfig()


class ReadabilityModel(nn.Module):

    def __init__(self,model_class):
        super().__init__()

        self.classes_size = config.class_num

        self.bert = model_class.from_pretrained(config.BERT_PATH)

        self.classification_layer_input_size = config.classification_layer_input_size

        self.linear = nn.Linear(self.classification_layer_input_size, self.classes_size)

        self.dropout = nn.Dropout(config.classification_layer_dropout)

    def forward(self, inputs, dropout):
        batch_size = inputs['input_ids'].size(0)
        last_hidden_states = self.bert(**inputs)['last_hidden_state']
        cls = last_hidden_states[:, 0:1, :]
        cls = cls.view(batch_size, -1)
        out = self.linear(cls)
        if dropout:
            out = self.dropout(out)
        return out
