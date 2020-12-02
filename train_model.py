#!/usr/bin/python
# coding:utf8
"""
@author: Jin Zitian
@time: 2020-11-11 16:55
"""
import tensorflow as tf
import numpy as np
import modeling
import optimization as optimization  # _freeze as optimization
import os, math, json
from data_processor import get_label2id

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 100167/64 = 1565.1= 1566 * 5 = 7830
config = {
    "train_data": "./data/train.tf_record",  # 第一个输入为 训练文件
    "dev_data": "./data/dev.tf_record",  # 第二个输入为 验证文件
    "bert_config": "./chinese_L-12_H-768_A-12/bert_config.json",  # bert模型配置文件
    "init_checkpoint": "./chinese_L-12_H-768_A-12/bert_model.ckpt",  # 预训练bert模型
    "train_examples_len": 33423,
    "dev_examples_len": 3741,
    "top_k": 5,
    "threshold": 0.3,
    "num_labels": 70,
    "train_batch_size": 32,
    "dev_batch_size": 64,
    "num_train_epochs": 5,
    "eval_per_step": 20,
    "learning_rate": 1.5e-5,
    "warmup_proportion": 0.1,
    "max_seq_len": 128,  # 输入文本片段的最大 char级别 长度
    "out": "./muti_category_bert_base/",  # 保存模型路径
    "loss_type": "sigmoid" #sigmoid
}


def load_bert_config(path):
    """
    bert 模型配置文件
    """
    return modeling.BertConfig.from_json_file(path)


def f_score(precision, recall):
    if precision == 0:
        return 0.0
    if recall == 0:
        return 0.0
    f_score = 2/(1/precision+ 1/recall)
    return f_score


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, keep_prob, num_labels,
                 use_one_hot_embeddings, loss_type):
    """
    input_ids=[[cls_id,1,2,sep_id,0,0,0],[cls_id,4,5,6,sep_id,0,0]]
    input_mask=[[1,1,1,1,0,0,0],[1,1,1,1,1,0,0]]
    segment_ids=[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    labels=[[1,5,2,-1,-1,-1,-1],[1,0,4,3,-1,-1,-1]]
    """
    
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope='bert'
    )
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    print(output_layer.shape)
    
    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    
    if loss_type == "softmax":
        #通过softmax方式构建损失函数学习，这样应该是趋于各个标签概率均衡的方式学习，但是仍然彼此存在影响，并不好控制标签判定阈值
        with tf.variable_scope("softmax_loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            prob = tf.nn.softmax(logits, axis=-1)

            #batch * labels_fix_len * num_labels, label不足时用-1补全labels_fix_len长度，不会影响计算结果
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            #batch * num_labels 多标签label
            multi_one_hot_labels = tf.reduce_sum(one_hot_labels,axis=1)

            log_probs = tf.nn.log_softmax(logits, axis=-1)

            batch_loss = -tf.reduce_sum(multi_one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(batch_loss)

            return (loss, logits, prob)
    else:
        #通过sigmod方式学习，可以排除各个标签的影响，比较容易找到判定阈值
        with tf.variable_scope("sigmod_loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            prob = tf.nn.sigmoid(logits)

            #batch * labels_fix_len * num_labels, label不足时用-1补全labels_fix_len长度，不会影响计算结果
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            #batch * num_labels 多标签label
            multi_one_hot_labels = tf.reduce_sum(one_hot_labels,axis=1)

            #sigmoid_loss = multi_one_hot_labels * -tf.log(prob) + (1 - multi_one_hot_labels) * -tf.log(1 - prob)
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=multi_one_hot_labels, logits=logits)

            batch_loss = tf.reduce_sum(sigmoid_loss, axis=-1)

            loss = tf.reduce_mean(batch_loss)

            return (loss, logits, prob)


def get_input_data(input_file, seq_length, num_labels, batch_size, is_training=True):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_ids"]
        input_mask = example["input_mask"]
        segment_ids = example["segment_ids"]
        labels = example["label_ids"]
        return input_ids, input_mask, segment_ids, labels

    dataset = tf.data.TFRecordDataset(input_file)
    # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
    if is_training:
        dataset = dataset.map(parser).batch(batch_size).shuffle(buffer_size=3000)
    else:
        dataset = dataset.map(parser).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, labels = iterator.get_next()
    return input_ids, input_mask, segment_ids, labels


def train():
    print("print start load the params...")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(config["out"])
    train_examples_len = config["train_examples_len"]
    dev_examples_len = config["dev_examples_len"]
    learning_rate = config["learning_rate"]
    eval_per_step = config["eval_per_step"]
    num_labels = config["num_labels"]
    num_train_steps = math.ceil(train_examples_len / config["train_batch_size"])
    num_dev_steps = math.ceil(dev_examples_len / config["dev_batch_size"])
    num_warmup_steps = math.ceil(num_train_steps * config["num_train_epochs"] * config["warmup_proportion"])
    print("num_train_steps:{},  num_dev_steps:{},  num_warmup_steps:{}".format(num_train_steps, num_dev_steps,
                                                                               num_warmup_steps))
    use_one_hot_embeddings = False
    is_training = True
    seq_len = config["max_seq_len"]
    init_checkpoint = config["init_checkpoint"]
    print("print start compile the bert model...")
    # 定义输入输出
    input_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_ids')
    input_mask = tf.placeholder(tf.int64, shape=[None, seq_len], name='input_mask')
    segment_ids = tf.placeholder(tf.int64, shape=[None, seq_len], name='segment_ids')
    labels = tf.placeholder(tf.int64, shape=[None, num_labels], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

    bert_config_ = load_bert_config(config["bert_config"])
    (total_loss, logits, prob) = create_model(bert_config_, is_training, input_ids,
                                                                         input_mask, segment_ids, labels, keep_prob,
                                                                         num_labels, use_one_hot_embeddings, config["loss_type"])
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps * config["num_train_epochs"], num_warmup_steps, False)
    print("print start train the bert model...")

    batch_size = config["train_batch_size"]
    dev_batch_size = config["dev_batch_size"]

    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver([v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name],
                           max_to_keep=2)  # 保存最后top3模型

    #动态调整gpu资源
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(init_global)
        print("start load the pre train model")

        if init_checkpoint:
            # tvars = tf.global_variables()
            tvars = tf.trainable_variables()
            print("global_variables", len(tvars))
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            print("initialized_variable_names:", len(initialized_variable_names))
            saver_ = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
            saver_.restore(sess, init_checkpoint)
            tvars = tf.global_variables()
            initialized_vars = [v for v in tvars if v.name in initialized_variable_names]
            not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
            tf.logging.info('--all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            for v in initialized_vars:
                print('--initialized: %s, shape = %s' % (v.name, v.shape))
            for v in not_initialized_vars:
                print('--not initialized: %s, shape = %s' % (v.name, v.shape))
        else:
            sess.run(tf.global_variables_initializer())
        # if init_checkpoint:
        #     saver.restore(sess, init_checkpoint)
        #     print("checkpoint restored from %s" % init_checkpoint)
        print("********* train start *********")

        # tf.summary.FileWriter("output/",sess.graph)
        # albert remove dropout
        def train_step(ids, mask, segment, y, step, epoch):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    labels: y,
                    keep_prob: 0.9}
            _, out_loss, p_ = sess.run([train_op, total_loss, prob], feed_dict=feed)
            #print("epoch :{}, step :{}, lr :{}, loss :{}".format(epoch, step, _[1], out_loss))
            return out_loss, p_, y

        def dev_step(ids, mask, segment, y, threshold, top_k):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    labels: y,
                    keep_prob: 1.0
                    }
            out_loss, prob_pre = sess.run([total_loss, prob], feed_dict=feed)
            
            #计算f值，取top5，再卡阈值
            # f1_total = 0
            # precision_total = 0
            # recall_total = 0
            # for i in range(len(y)):
            #     #获取top_k index
            #     pre_top_k = np.argsort(prob_pre[i])[-top_k:]
            #     pre = set(pre_top_k[np.where(prob_pre[i][pre_top_k] > threshold)])
            #     ori = set(y[i])
            #     if -1 in ori:
            #         ori.remove(-1)
            #     both = pre.intersection(ori)
            #     precision = len(both)*1.0/len(pre) if len(pre) != 0 else 0.0
            #     recall = len(both)*1.0/len(ori) if len(ori) != 0 else 0.0
            #     f1 = f_score(precision, recall)
            #     f1_total += f1
            #     precision_total += precision
            #     recall_total += recall
            #     #print("precision :{}, recall :{}, pre :{}, ori :{}".format(precision, recall, len(pre), len(ori)))
            # f1_avg = f1_total/len(y)
            # precision_avg = precision_total/len(y)
            # recall_avg = recall_total/len(y)
            
            #计算argmax的acc
            # true_num = 0
            # for i in range(len(y)):
            #     class_id = np.argmax(prob_pre[i])
            #     ori = set(y[i])
            #     if class_id in ori:
            #         true_num += 1
            # acc = true_num/len(y)
            
            #计算主要类别的macro-F1值、micro-F1值、argmax的acc
            tags = '伤感,网络,经典,放松,华语,安静,开心,流行,思念,治愈,古风,兴奋,励志,欧美,甜蜜,开车,怀旧,寂寞,校园,影视'
            tag_ids = {label2id[i] for i in tags.split(',')}
            f1_total_m = 0
            precision_total_m = 0
            recall_total_m = 0
            m_s = 0
            for i in range(len(y)):
                #获取top_k index
                pre_top_k = np.argsort(prob_pre[i])[-top_k:]
                pre = set(pre_top_k[np.where(prob_pre[i][pre_top_k] > threshold)])
                ori = set(y[i])
                if -1 in ori:
                    ori.remove(-1)
                if tag_ids.intersection(pre) or tag_ids.intersection(ori):
                    both = pre.intersection(ori)
                    precision = len(both)*1.0/len(pre) if len(pre) != 0 else 0.0
                    recall = len(both)*1.0/len(ori) if len(ori) != 0 else 0.0 
                    f1 = f_score(precision, recall)
                    f1_total_m += f1
                    precision_total_m += precision
                    recall_total_m += recall
                    
                    m_s += 1
                #print("precision :{}, recall :{}, pre :{}, ori :{}".format(precision, recall, len(pre), len(ori)))
            f1_avg = f1_total_m/m_s
            precision_avg = precision_total_m/m_s
            recall_avg = recall_total_m/m_s
            
            #计算主要类别的argmax的acc
            true_num = 0
            all_num = 0
            for i in range(len(y)):
                #class_id = np.argmax(prob_pre[i])
                class_ids = set(np.argsort(prob_pre[i])[-3:])
                ori = set(y[i])
                if tag_ids.intersection(ori):
                    all_num += 1
                    #if class_id in ori:
                    if class_ids.intersection(ori):
                        true_num += 1
            acc = true_num/all_num
            
            
            #print("step :{}, loss :{}, f_score :{}, precision :{}, recall :{}".format(step, out_loss, f1_avg, precision_avg, recall_avg))
            return out_loss, prob_pre, y, f1_avg, precision_avg, recall_avg, acc

        step = 0
        for epoch in range(config["num_train_epochs"]):
            #格式化输出 居中对齐
            print("{:*^100s}".format(("epoch-" + str(epoch)).center(20)))
            # 读取训练数据

            input_ids2, input_mask2, segment_ids2, labels2 = get_input_data(config["train_data"], seq_len, num_labels, batch_size)
            for i in range(num_train_steps):
                step += 1
                ids_train, mask_train, segment_train, y_train = sess.run(
                    [input_ids2, input_mask2, segment_ids2, labels2])
                train_step(ids_train, mask_train, segment_train, y_train, step, epoch)

                if step % eval_per_step == 0:
                    total_loss_dev = 0
                    total_f1 = 0
                    total_precision = 0
                    total_recall = 0
                    total_acc = 0
                    dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2 = get_input_data(config["dev_data"],seq_len,num_labels,dev_batch_size,False)
                    
                    for j in range(num_dev_steps):  # 一个 epoch 的 轮数
                        ids_dev, mask_dev, segment_dev, y_dev = sess.run(
                            [dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2])
                        out_loss, pre, y, f1, precision, recall, acc = dev_step(ids_dev, mask_dev, segment_dev, y_dev, config["threshold"], config["top_k"])
                        total_loss_dev += out_loss
                        total_f1 += f1
                        total_precision += precision
                        total_recall += recall
                        total_acc += acc
                        
                    #print("total_loss_dev:{}".format(total_loss_dev))
                    #print("avg_f1_dev:{}".format(total_f1/num_dev_steps))
                    #print("avg_precision_dev:{}".format(total_precision/num_dev_steps))
                    #print("avg_recall_dev:{}".format(total_recall/num_dev_steps))
                    #print("avg_acc_dev:{}".format(total_acc/num_dev_steps))
                    print("epoch:{:<2}, step:{:<6}, loss:{:<10.6}, acc:{:<10.3}, f1:{:<10.3}, precision:{:<10.3}, recall:{:<10.3}".format(epoch, step, total_loss_dev, total_acc/num_dev_steps, total_f1/num_dev_steps, total_precision/num_dev_steps, total_recall/num_dev_steps))
                    saver.save(sess, config["out"] + 'bert.ckpt', global_step=step)
                

if __name__ == "__main__":
    print("********* seq label start *********")
    label2id = get_label2id('./data/labels.txt')
    train()

