import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'bert'
        self.save_path = 'checkpoint/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = 2
        self.max_length = 128
        self.batch_size = 16
        self.num_epochs = 3
        self.learning_rate = 5e-5
        self.bert_path = 'bert-base-chinese'  # 修改这里为 Hugging Face Hub 上的模型名称
        self.hidden_size = 768


class MyModel(nn.Module):

    def __init__(self, Config):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(Config.hidden_size, Config.num_classes)

    def forward(self, x):
        input_ids = x['input_ids']  # 输入的句子每个词对应的id
        attn_mask = x['attention_mask']  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attn_mask, inputs_embeds=None,
                              return_dict=False)
        out = self.fc(pooled)
        return out
