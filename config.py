import warnings
import torch

class DefaultConfig(object):
    BERT_PATH = 'bert_models/chinese-bigbird-base-4096'
    dataset_type = "CMT"
    class_num = 12
    root_dir_train = 'dataset/' + dataset_type + '/data_train'
    root_dir_test = 'dataset/' + dataset_type + '/data_test'
    # just default
    batch_size = 1
    train_batch_size = batch_size
    valid_batch_size = batch_size
    test_batch_size = batch_size

    lr = 0.00001
    weight_decay = 0.01
    max_epoch = 10
    dropout = True
    classification_layer_dropout = 0.2
    seed = 1
    sentence_length_limit = True
    schedule = True
    warmup_steps = 0.1 * max_epoch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_sentence_length = 2048  # 所设置的最大长度不得超过512

    classification_layer_input_size = 768

    def parse(self, kwargs):

        for key, value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn("Warnning: has not attribut %s" % key)
            setattr(self, key, value)

        print()
        print("config:")
        print("{")
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('__'):
                if key != 'bert_hidden_size':
                    print('     ', key, ' = ', getattr(self, key))
        print("}")
        print()
