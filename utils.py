import torch
import transformers
import jieba_fast
from transformers import BertTokenizer, BigBirdModel, AutoModelForMaskedLM, AutoTokenizer, AutoModel


class JiebaTokenizer(BertTokenizer):
    def __init__(
        self, pre_tokenizer=lambda x: jieba_fast.cut(x, HMM=False), *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer
    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for word in self.pre_tokenizer(text):
            if word in self.vocab:
                split_tokens.append(word)
            else:
                split_tokens.extend(super()._tokenize(word))
        return split_tokens

def setup_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  #将这种写法改为更加通用的允许GPU和CPU允许的写法


def get_model(model_type):
    if model_type == 'bert':
        tokenizer_class = BertTokenizer
        model_class = BigBirdModel

    elif model_type == 'chinese-bigbird-base-4096':  #
        tokenizer_class = JiebaTokenizer
        model_class = BigBirdModel

    elif model_type == 'chinese-roberta-wwm-ext':  #
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM

    elif model_type == 'chinese-bert-wwm':  #
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM

    elif model_type == 'chinese-macbert-large':  #
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM

    elif model_type == 'roberta':  #
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM

    elif model_type == 'bart':  #
        tokenizer_class = AutoTokenizer
        model_class = AutoModel

    elif model_type == 'bart-base-chinese':
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM

    elif model_type == 'chinese-pert-base':
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM

    elif model_type == 'chinese-pert-large':
        tokenizer_class = AutoTokenizer
        model_class = AutoModelForMaskedLM
    else:
        raise ValueError("Model must be either BERT, XLNet, RoBERTa, or BART")

    return tokenizer_class, model_class

def clear_cache(device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "cpu":
        pass  # 什么都不做
    else:
        raise ValueError("不支持的设备类型")