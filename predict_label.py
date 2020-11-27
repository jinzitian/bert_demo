#!/usr/bin/python
# coding:utf8
"""
@author: Jin Zitian
@time: 2020-11-11 16:55
"""
import os
import numpy as np
import tensorflow as tf
import tokenization

from data_processor import get_label2id

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vocab_file = "./chinese_L-12_H-768_A-12/vocab.txt"
tokenizer_ = tokenization.FullTokenizer(vocab_file=vocab_file)
label2id = get_label2id('./data/labels.txt')
id2label = {v : k for k, v in label2id.items()}
threshold = 0.5


def process_one_example_p(tokenizer, text, max_seq_len=128):

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
        
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids)
    return feature


def load_model(model_folder):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder, repr(e))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.reset_default_graph()
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We start a session and restore the graph weights
    sess_ = tf.Session()
    saver.restore(sess_, input_checkpoint)

    # opts = sess_.graph.get_operations()
    # for v in opts:
    #     print(v.name)
    return sess_


model_path = "./muti_category_bert_base/"
sess = load_model(model_path)
input_ids = sess.graph.get_tensor_by_name("input_ids:0")
input_mask = sess.graph.get_tensor_by_name("input_mask:0")  # is_training
segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")  # fc/dense/Relu  cnn_block/Reshape
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
prob = sess.graph.get_tensor_by_name("sigmod_loss/Sigmoid:0")


def predict(text, threshold):
    data = [text]
    # 逐个分成 最大62长度的 text 进行 batch 预测
    features = []
    for i in data:
        feature = process_one_example_p(tokenizer_, i, max_seq_len=64)
        features.append(feature)
    feed = {input_ids: [feature[0] for feature in features],
            input_mask: [feature[1] for feature in features],
            segment_ids: [feature[2] for feature in features],
            keep_prob: 1.0
            }

    probs = sess.run(prob, feed)
    pre_labels = [[id2label[i] for i in np.where(data>threshold)[0]] for data in probs]
    return probs, pre_labels



if __name__ == "__main__":
    tag = "怀旧,经典"
    text_ = "在那遥远的地方  - 阿鲁阿卓, 词：王洛宾 曲：哈萨克民歌, 在那遥远的地方, 有位好姑娘, 人们走过她的帐房, 都要回头留恋地张望, 她那粉红的笑脸, 好像红太阳, 她那美丽动人的眼睛, 好像晚上明媚的月亮, 我愿抛弃了财产, 跟她去放羊, 每天看着那粉红的笑脸, 和那美丽金边的衣裳, 我愿做一只小羊, 跟在她身旁, 我愿她拿着细细的皮鞭, 不断轻轻打在我身上, 我愿她拿着细细的皮鞭, 不断轻轻打在我身上"
    (probs, pre_labels) = predict(text_, threshold)
    print(probs)
    print(pre_labels)
