import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

import CONFIG


def convert_text_to_token(tokenizer, sentence, limit_size=126):
    """
    将句子转化从编码并截断或补长到同一长度

    :param tokenizer: 分词器
    :param sentence: 原句
    :param limit_size: 规定长度
    :return: list: tokens
    """
    tokens = tokenizer.encode(sentence[:limit_size])
    if len(tokens) < limit_size + 2:
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


def processing_data(data_path, tokenizer):
    """
    数据处理

    :param data_path: 数据集路径
    :param split_ratio: 训练集占比
    :return: train_iter, val_iter
    """
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    sentences = []
    target = []

    files = os.listdir(data_path)
    for file in files:
        if not os.path.isdir(file) and not file[0] == '.': # 跳过文件夹和隐藏文件
            f = open(data_path + '/' + file, 'r', encoding='UTF-8')
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    target = torch.tensor(target)
    sentences_ids = [convert_text_to_token(tokenizer, sen, CONFIG.LIMIT_SIZE) for sen in sentences]
    sentences_ids = torch.tensor(sentences_ids)

    attention_masks = []
    for seq in sentences_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = torch.tensor(attention_masks)

    return sentences_ids, target, attention_masks


def dataloader_maker(data, masks, labels, valid_ratio=0.3):
    """
    构造 DataLoader

    :param data: 数据集
    :param masks: mask
    :param labels: 标签
    :param valid_ratio: 验证集比例
    :return: train_dataloader, valid_dataloaer(None if valid_ratio = 0)
    """
    if (valid_ratio > 0) & (valid_ratio < 1):
        train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, random_state=1,
                                                                              test_size=0.3)
        train_masks, valid_masks, _, _ = train_test_split(masks, masks, random_state=1, test_size=0.3)

        train_dataset = TensorDataset(train_data, train_masks, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)

        valid_dataset = TensorDataset(valid_data, valid_masks, valid_labels)
        valid_dataloader = DataLoader(valid_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
        return train_dataloader, valid_dataloader
    else:
        train_dataset = TensorDataset(data, masks, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
        return train_dataloader, None


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(CONFIG.CACHE_DIR)
    processing_data(CONFIG.DATA_PATH, tokenizer)