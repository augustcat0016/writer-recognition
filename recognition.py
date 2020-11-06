import CONFIG
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from processing_data import processing_data, dataloader_maker, convert_text_to_token
from sklearn.model_selection import train_test_split


def model_loading(model_dir, device=None):
    """
    装载 BertTokenizer 和 BertForSequenceClassification

    :param model_dir: 模型存放路径
    :param device: 模型装载到的设备
    :return: tokenizer, model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(model_dir)

    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=CONFIG.LABEL_NUM
    )
    model.to(device)
    return tokenizer, model


def binary_acc(preds, labels):
    """
    计算预测结果的 accuracy

    :param preds: 预测结果: [num_sample * num_label]
    :param labels: 标签: [num_sample]
    :return: accuracy: float
    """
    correct = torch.eq(torch.argmax(preds, dim=1), labels.flatten()).float()
    acc = correct.sum().item() / len(correct)
    return acc


def train(train_dataloader, valid_dataloader, num_epoch):
    """
    装载预训练模型，并在数据集上训练
    模型将保存在CONFIG.MODEL_SAVE_DIR下

    :param train_dataloader: DATALOADER
    :param valid_dataloader: DATALOADER
    :param num_epoch: 训练epoch数
    :return: None
    """

    model = BertForSequenceClassification.from_pretrained(CONFIG.CACHE_DIR, num_labels=CONFIG.LABEL_NUM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, eps=CONFIG.EPSILON)

    total_steps = len(train_dataloader) * num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    record = [[], [], []]

    output_model_file = os.path.join(CONFIG.MODEL_SAVE_DIR, CONFIG.FILENAME_MODEL)
    output_config_file = os.path.join(CONFIG.MODEL_SAVE_DIR, CONFIG.FILENAME_CONFIG)

    for epoch in range(num_epoch):
        train_loss = []

        model.train()
        for step, batch in enumerate(train_dataloader):
            text = batch[0].long().to(device)
            mask = batch[1].long().to(device)
            label = batch[2].long().to(device)
            loss, logits = model(text, attention_mask=mask, labels=label)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            print('\r epoch {} loss:{}'.format(epoch, loss.item()), end='')

        valid_loss, valid_acc = [], []
        with torch.no_grad():
            for batch in valid_dataloader:
                text = batch[0].long().to(device)
                mask = batch[1].long().to(device)
                label = batch[2].long().to(device)
                loss, logits = model(text, attention_mask=mask, labels=label)
                acc = binary_acc(logits, label)
                valid_loss.append(loss.item())
                valid_acc.append(acc)
        avg_train_loss = np.mean(train_loss)
        avg_valid_loss = np.mean(valid_loss)
        avg_valid_acc = np.mean(valid_acc)
        record[0].append(avg_train_loss)
        record[1].append(avg_valid_loss)
        record[2].append(avg_valid_acc)
        print('\n epoch {} complete.'.format(epoch))
        print(' train loss: {}\n valid loss: {}\n valid accuracy: {}'.format(avg_train_loss, avg_valid_loss,
                                                                             avg_valid_acc))

    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)


def predict(sen, tokenizer=None, model=None, device=None):
    """
    预测文本作者

    :param sen: 文本输入
    :param tokenizer: 分词器
    :param model: 模型
    :param device: 设备
    :return: 作者编号
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None or model is None:
        tokenizer, model = model_loading(CONFIG.MODEL_SAVE_DIR, device)

    input_id = convert_text_to_token(tokenizer, sen)
    text = torch.tensor(input_id).long().to(device)
    atten = [float(i > 0) for i in input_id]
    mask = torch.tensor(atten).long().to(device)

    logits = model(text.view(1, -1), attention_mask=mask.view(1, - 1))
    prediction = torch.argmax(logits[0]).item()
    return prediction

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(CONFIG.CACHE_DIR)
    data, labels, masks = processing_data(CONFIG.DATA_PATH, tokenizer)

    train_dataloader, valid_dataloader = dataloader_maker(data, masks, labels)

    train(train_dataloader, valid_dataloader, CONFIG.NUM_EPOCH)
    tokenizer.save_vocabulary(CONFIG.MODEL_SAVE_DIR)
