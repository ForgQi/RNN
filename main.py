import random

import functions
import re
import numpy as np
import tensorflow as tf
import os

poetrys = functions.getpoetry()
# print(len(poetrys))
# print(len(titles))
#print(poetrys)
# print(titles)


#获取字频字典，字频字典排序后列表，非汉字字典
sortdict1 = functions.count_char(poetrys)
#print(sortdict1)
#获取汉字数字id与汉字
word_map ,words = functions.get_mapping(sortdict1)
#print(word_map[' '])
#向量化诗
to_num = lambda word: word_map.get(word, len(sortdict1))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

# print(poetrys_vector)
#print(len(poetrys_vector))
batch_size = 64
x_batches,y_batches,n_chunk  = functions.poembatch(poetrys_vector, word_map, batchsize=batch_size)

# print(x_batches)
# print(y_batches)

input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
output_keep_prob = tf.placeholder(tf.float32)

def neural_network(model='lstm', batchsize = batch_size,rnn_size=128, num_layers=3):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
    cell = cell_fun(rnn_size, state_is_tuple=True)

    cells = []
    # 因为是多层RNN，所以在recoll时我们要输入的是一个多层的cell，
    # 根据是否处于训练过程和需要dropout添加dropout层
    for _ in range(num_layers):

        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=output_keep_prob)
        cells.append(cell)
    # MultiRNNCell接受我们之前定义的多层RNNcell列表。
    # state_is_tuple默认为True，表示输入和输出都用tuple存储，将来会丢弃False的选项。

    # 每调用一次这个函数就返回一个BasicRNNCell
    # def get_a_cell():
    #     return cell

    # 用tf.nn.rnn_cell MultiRNNCell创建n层RNN
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)  # n层RNN

    initial_state = cell.zero_state(batchsize, tf.float32)
    #创建变量w，b
    softmax_w , softmax_b = functions.softmax_variable(rnn_size,word_len=len(words))
    inputs = functions.embedding_variable(inputs=input_data,rnn_size=rnn_size,word_len=len(words))


    inputs = tf.nn.dropout(inputs, output_keep_prob)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnn')
    output = tf.reshape(outputs, [-1, rnn_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    #对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，输出就是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率)
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state

def train_neural_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets],
                                                              [tf.ones_like(targets, dtype=tf.float32)]
                                                              )
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    #返回一个长为len(xs)的tensor列表，列表中的每个tensor是ys中每个值对xs[i]求导之和。
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.exists('save/checkpoint'):  # 判断模型是否存在
            saver.restore(sess, 'save/model.ckpt-50')  # 存在就从模型中恢复变量
        else:
            sess.run(tf.global_variables_initializer())


        for epoch in range(51):
            sess.run(tf.assign(learning_rate, 0.003 * (0.97 ** epoch)))
            #n = 0
            for batche in range(n_chunk):
                n = random.randint(1, n_chunk-1)
                for _ in range(1):
                    train_loss, _, _ = sess.run([cost, last_state, train_op],
                                                feed_dict={input_data: x_batches[n], output_targets: y_batches[n],
                                                             output_keep_prob:0.9})

                #n += 1
                print(epoch, batche, train_loss)
            if epoch % 10 == 0:
                print('Saving')
                # train_loss, _, _ = sess.run([cost, last_state, train_op],
                #                             feed_dict={input_data: x_batches[n], output_targets: y_batches[n],
                #                                        output_keep_prob: 1})
                saver.save(sess, 'save/model.ckpt', global_step=epoch)

def gen():
    def to_word(weights):
        #累计求和
        # [1,2,3,4,5]→[1,3,6,10,15]
        t = np.cumsum(weights)
        s = np.sum(weights)
        #数组的插入：np.searchsorted(a,b)将b插入原有序数组a，并返回插入元素的索引值
        #rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        #print(t,s,sample)
        return words[sample]

        # 输入

    # 句子长短不一致 用None自适应
    #inputs = tf.placeholder(tf.int32, shape=(batch_size, None), name='inputs')
    # 防止过拟合
    #keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #lstm_inputs = functions.embedding_variable(inputs,rnn_size=rnn_size, word_len=len(main.words))
    # rnn模型
    _, last_state, probs, cell, initial_state = neural_network(batchsize=1 )


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, 'save/model.ckpt-100')
        state_ = sess.run(cell.zero_state(1, tf.float32))

        x = np.array([list(map(word_map.get, ' '))])
        #print('x1',x)
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        #print(word)
        poem = ''+word
        while poem[-1] != ' ':
            #poem += word
            x = np.zeros((1, 1))
            #print(x)
            x[0, 0] = word_map[word]
            #print('w',word_map[word])
            #print('x2',x)
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
            poem += word
        return poem

if __name__ == '__main__' :
    train_neural_network()
    #print(gen())




