import os

import fire
import torch
from torch import nn
from torch.autograd import Variable
from utils import get_model,setup_seed,clear_cache
from transformers import get_cosine_schedule_with_warmup,AdamW
from data import dataset, dataloader
from models import ReadabilityModel
from config import DefaultConfig
from sklearn.metrics import (precision_score,recall_score,f1_score,cohen_kappa_score)

import warnings

warnings.filterwarnings("ignore")
config = DefaultConfig()
folder = './model_saved/'+config.BERT_PATH.replace("/","-")
os.makedirs(folder, exist_ok=True)
trainingLog_handle = open(folder+'/log_CMT' + '.txt', mode='w')

def train(**kwards):
    config.parse(kwards)
    tokenizer_class,model_class = get_model(config.BERT_PATH.split("")[1])

    trainingLog_handle.write('\n')
    trainingLog_handle.write("config: \n")
    trainingLog_handle.write("{ \n")
    for key, value in config.__class__.__dict__.items():
        if not key.startswith('__'):
            if key != 'bert_hidden_size':
                trainingLog_handle.write('     ' + key + ' = ' + str(getattr(config, key)) + '\n')
    trainingLog_handle.write("} \n")
    trainingLog_handle.write('\n')

    setup_seed(config.seed)

    train_dataset = dataset.TrainDataset(config.root_dir_train, config=config)
    train_loader = dataloader.MyDataloader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                           num_workers=16, config=config,tokenizer_class=tokenizer_class)
    test_dataset = dataset.TrainDataset(config.root_dir_test, config=config)
    test_loader = dataloader.MyDataloader(dataset=test_dataset, batch_size=config.test_batch_size, shuffle=True,
                                          num_workers=16, config=config,tokenizer_class=tokenizer_class)

    readabilityModel = ReadabilityModel(model_class=model_class)
    readabilityModel = readabilityModel.to(config.device)
    print(readabilityModel)
    trainingLog_handle.write(str(readabilityModel))
    trainingLog_handle.write('\n')
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(readabilityModel.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=config.max_epoch)

    best_test = 0
    for epoch in range(1, config.max_epoch + 1):

        running_loss = 0.0

        for i, train_data in enumerate(train_loader, 1):

            text_ids, label = train_data

            text_ids = {k: Variable(v) for k, v in text_ids.items()}
            label = Variable(torch.tensor([*label]))

            text_ids = {k: v.to(config.device) for k, v in text_ids.items()}
            label = label.to(config.device)

            optimizer.zero_grad()

            outputs = readabilityModel(text_ids, dropout=config.dropout)

            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i % (1024 // config.train_batch_size) == 0:
                print('[%d, %5d] loss: %.3f' % (
                    epoch, i * config.train_batch_size, running_loss / (1024 // config.train_batch_size)))
                trainingLog_handle.write('[%d, %5d] loss: %.3f' % (
                    epoch, i * config.train_batch_size, running_loss / (1024 // config.train_batch_size)))
                trainingLog_handle.write('\n')
                running_loss = 0.0

        if epoch % 1 == 0:
            print()
            trainingLog_handle.write('\n')
            clear_cache()
            tr_top1, tr_adj = test(readabilityModel, train_loader, "train")
            clear_cache()
            test_top1, test_adj = test(readabilityModel, test_loader, "test")
            clear_cache()
            print()
            trainingLog_handle.write('\n')
            if test_top1 >= best_test:
                path = './model_save/' + config.BERT_PATH.split("/")[1]
                os.makedirs(path)
                save_dir = path + '/test_' + str(epoch) + '_t_' + str(test_top1)[0:8] + '.pth'
                torch.save(readabilityModel, path)
                best_test = test_top1

        if config.schedule:
            scheduler.step()

    trainingLog_handle.close()


def test(readabilityModel, loader, dataset_type):
    correct_top1 = 0.0
    correct_adj = 0.0
    total = 0.0
    y_predict = []
    y_true = []

    for i, data in enumerate(loader, 1):

        text_ids, label = data
        text_ids = {k: Variable(v) for k, v in text_ids.items()}
        label = Variable(torch.tensor([*label]))

        text_ids = {k: v.to(config.device) for k, v in text_ids.items()}
        label = label.to(config.device)

        with torch.no_grad():
            outputs = readabilityModel(text_ids, dropout=False)

        _, predicted = torch.sort(outputs, dim=1, descending=True)
        y_array = label.cpu().numpy().tolist()
        y_pred_array = predicted.cpu().numpy().tolist()
        y_pred = []
        for i, p in enumerate(y_pred_array):
            y_pred.append(p[0])

        y_predict.extend(y_pred)
        y_true.extend(y_array)
        total += label.size(0)
        for i, p in enumerate(y_pred_array):
            if label[i] == p[0]:
                correct_top1 = correct_top1 + 1
            if label[i] == p[0] or label[i] - p[0] == 1 or label[i] - p[0] == -1:
                correct_adj = correct_adj + 1
    precision_scores = precision_score(y_true,list(outputs) , average='weighted')
    recall_scores = recall_score(y_true, y_predict, average='weighted')
    f1_scores = f1_score(y_true, y_predict, average='weighted')
    cohen_kappa_scores = cohen_kappa_score(y_true, y_predict, weights='quadratic')
    print(dataset_type + '       C_Acc: %.3f%%' % (100 * correct_top1 / total) + '       Acc_adj: %.3f%%' % (
            100 * correct_adj / total) + '      Precision: %.3f%%' % (
                  precision_scores * 100) + '    Recall: %.3f%%' % (
                  recall_scores * 100) + '     F1: %.3f%%' % (
                  f1_scores * 100) + '     QWK: %.3f%%' % (
                  cohen_kappa_scores * 100))
    print('\n')
    trainingLog_handle.write(
        dataset_type + '       C_Acc: %.3f%%' % (100 * correct_top1 / total) + '       Acc_adj: %.3f%%' % (
                100 * correct_adj / total) + '      Precision: %.3f%%' % (
                precision_scores * 100) + '    Recall: %.3f%%' % (
                recall_scores * 100) + '     F1: %.3f%%' % (
                f1_scores * 100) + '     QWK: %.3f%%' % (
                cohen_kappa_scores * 100))
    trainingLog_handle.write('\n')
    return correct_top1 / total, correct_adj / total


if __name__ == '__main__':
    fire.Fire()
    # train()