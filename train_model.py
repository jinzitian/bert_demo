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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 100167/64 = 1565.1= 1566 * 5 = 7830
config = {
    "in_1": "./data/train.tf_record",  # 第一个输入为 训练文件
    "in_2": "./data/dev.tf_record",  # 第二个输入为 验证文件
    "bert_config": "./chinese_L-12_H-768_A-12/bert_config.json",  # bert模型配置文件
    "init_checkpoint": "./chinese_L-12_H-768_A-12/bert_model.ckpt",  # 预训练bert模型
    "train_examples_len": 10748,
    "dev_examples_len": 1343,
    "num_labels": 41,
    "train_batch_size": 32,
    "dev_batch_size": 32,
    "num_train_epochs": 5,
    "eval_start_step": 1300,
    "eval_per_step": 100,
    "auto_save": 50,
    "learning_rate": 3e-5,
    "warmup_proportion": 0.1,
    "max_seq_len": 64,  # 输入文本片段的最大 char级别 长度
    "out": "./ner_bert_base/",  # 保存模型路径
    "out_1": "./ner_bert_base/"  # 保存模型路径
}


def load_bert_config(path):
    """
    bert 模型配置文件
    """
    return modeling.BertConfig.from_json_file(path)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, keep_prob, num_labels,
                 use_one_hot_embeddings):
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
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        prob = tf.nn.softmax(logits, axis=-1)
        y_pre = tf.argmax(prob, 1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        
        return (loss, logits, prob, y_pre)


def get_input_data(input_file, seq_length, batch_size, is_training=True):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
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
    labels = tf.placeholder(tf.int64, shape=[None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # , name='is_training'

    bert_config_ = load_bert_config(config["bert_config"])
    (total_loss, logits, prob, y_pre) = create_model(bert_config_, is_training, input_ids,
                                                                         input_mask, segment_ids, labels, keep_prob,
                                                                         num_labels, use_one_hot_embeddings)
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps * config["num_train_epochs"], num_warmup_steps, False)
    print("print start train the bert model...")

    batch_size = config["train_batch_size"]
    dev_batch_size = config["dev_batch_size"]

    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver([v for v in tf.global_variables() if 'adam_v' not in v.name and 'adam_m' not in v.name],
                           max_to_keep=2)  # 保存最后top3模型

    with tf.Session() as sess:
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
        def train_step(ids, mask, segment, y, step):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    labels: y,
                    keep_prob: 0.9}
            _, out_loss, p_ = sess.run([train_op, total_loss, prob], feed_dict=feed)
            print("step :{}, lr:{}, loss :{}".format(step, _[1], out_loss))
            return out_loss, p_, y

        def dev_step(ids, mask, segment, y, step):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: segment,
                    labels: y,
                    keep_prob: 1.0
                    }
            out_loss, p_, y_p = sess.run([total_loss, prob, y_pre], feed_dict=feed)
            
            #计算acc
            accuracy = np.mean(y_p == y)
            print("step :{}, loss :{}, acc :{}".format(step, out_loss, accuracy))
            return out_loss, p_, y, accuracy

        step = 0
        for epoch in range(config["num_train_epochs"]):
            #格式化输出 居中对齐
            print("{:*^100s}".format(("epoch-" + str(epoch)).center(20)))
            # 读取训练数据

            input_ids2, input_mask2, segment_ids2, labels2 = get_input_data(config["in_1"], seq_len, batch_size)
            for i in range(num_train_steps):
                step += 1
                ids_train, mask_train, segment_train, y_train = sess.run(
                    [input_ids2, input_mask2, segment_ids2, labels2])
                out_loss, pre, y = train_step(ids_train, mask_train, segment_train, y_train, step)

                if step % eval_per_step == 0:
                    total_loss_dev = 0
                    total_acc = 0
                    dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2 = get_input_data(config["in_2"],seq_len,dev_batch_size,False)
                    
                    for j in range(num_dev_steps):  # 一个 epoch 的 轮数
                        ids_dev, mask_dev, segment_dev, y_dev = sess.run(
                            [dev_input_ids2, dev_input_mask2, dev_segment_ids2, dev_labels2])
                        out_loss, pre, y, acc = dev_step(ids_dev, mask_dev, segment_dev, y_dev)
                        total_loss_dev += out_loss
                        total_acc += acc
                        
                    print("total_loss_dev:{}".format(total_loss_dev))
                    print("avg_acc_dev:{}".format(total_acc/num_dev_steps))
                    saver.save(sess, config["out"] + 'bert.ckpt', global_step=step)
                

if __name__ == "__main__":
    print("********* seq label start *********")
    train()
