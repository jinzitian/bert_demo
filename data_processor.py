#!/usr/bin/python
# coding:utf8
"""
@author: Jin Zitian
@time: 2020-11-11 16:55
"""

import tokenization
import tensorflow as tf
import numpy as np


#将hive导出的数据变成标准格式，再做后续处理
def make_hive_data_to_normal_test_data(file_path, train_path, dev_path):
    train = open(train_path, 'w', encoding='utf-8')
    dev = open(dev_path, 'w', encoding='utf-8')
    
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    dev_size = int(0.1*len(lines))
    rand_index = np.random.permutation(len(lines))
    num = 0
    for i in rand_index:
        line = lines[i]
        if line.strip() == '' or 'music_id' in line:
            continue
        arr = line.strip().split('|')
        if len(arr) == 8:
            s = ''
            for a in arr:
                s += a.strip()+'\t'
            if num > dev_size:
                train.write(s.strip()+'\n')
            else:
                dev.write(s.strip()+'\n')
        num += 1
        
    train.close()
    dev.close()


def process_one_example(tokenizer, label2id, text, label, max_seq_len=128):
    
    label_size = len(label2id)
    labels = label.split(',')
    label_ids = []
    for label in labels:
        label_ids.append(label2id[label])
        
    tokens = tokenizer.tokenize(text)
    # tokens = tokenizer.tokenize(example.text)  -2 的原因是因为序列需要加一个句首和句尾标志
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        
    while len(label_ids) < label_size:
        label_ids.append(-1)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(label_ids) == label_size

    feature = (input_ids, input_mask, segment_ids, label_ids)
    return feature


def prepare_tf_record_data(tokenizer, max_seq_len, label2id, path, out_path):
    """
        生成训练数据， tf.record, 单标签分类模型, 随机打乱数据
    """
    writer = tf.python_io.TFRecordWriter(out_path)
    example_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            arr = line.strip().split('\t')
            text = arr[5][1:-1]
            labels = arr[4]
            feature = process_one_example(tokenizer, label2id, text, labels, max_seq_len)
            features = {}
            # 序列标注任务
            features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature[0]))
            features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature[1]))
            features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature[2]))
            features["label_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature[3]))

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            example_count += 1

    print("total example:", example_count)
    writer.close()
   
    
def get_label2id(label_file):
    label2id = {}
    i=0
    with open(label_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line.strip() == '':
                break
            category = line.strip().split('\t')[0]
            if category not in label2id:
                label2id[category] = i
                i += 1
    return label2id
        

if __name__ == "__main__":
    vocab_file = "./chinese_L-12_H-768_A-12/vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    label2id = get_label2id('./data/labels.txt')

    max_seq_len = 128
    make_hive_data_to_normal_test_data('./data/train_data.txt', './data/norm_train_data.txt', './data/norm_dev_data.txt')
    prepare_tf_record_data(tokenizer, max_seq_len, label2id, path="./data/norm_train_data.txt",
                           out_path="./data/train.tf_record")
    prepare_tf_record_data(tokenizer, max_seq_len, label2id, path="./data/norm_dev_data.txt",
                           out_path="./data/dev.tf_record")
