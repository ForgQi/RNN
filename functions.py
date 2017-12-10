import re
import collections
import numpy as np
import tensorflow as tf

def is_chinese(uchar):
    if uchar>= '\u4e00' and uchar<='\u9fa5':     # if uchar not in ["_", "（", "）","【","，","。","\n","？","、","…","《","》","】",":"]:
        return True    #     return True
    else:
        return False

def get_chinese(txtpath):
    with open(txtpath, 'r', encoding='utf') as txt_file:
        file = txt_file.read()
        pattern = re.compile('[\u4e00-\u9fa5]+')
        result = re.findall(pattern, file)
    chinese = ''
    for i in result:
        chinese += i
    return chinese

def count_char(poetrys) :
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    return count_pairs

def count_chinese_word(filepath):
    _dict = {}
    dict = {}
    try:
        with open(filepath, 'r',encoding='utf') as txt_file:
            file = txt_file.read()
            for char in file:
                if is_chinese(char):
                    if _dict.__contains__(char):
                        _dict[char] = _dict[char] + 1
                    else:
                        _dict[char] = 1
                else:
                    if dict.__contains__(char):
                        dict[char] = dict[char] + 1
                    else:
                        dict[char] = 1
    except IOError as ioerr:
        print(ioerr)
        print("文件", filepath, "不存在")
    sdict = sorted(_dict.items(), key=lambda d: d[1], reverse=True)

    return _dict,sdict,dict

def save_word_frequency (_dict,name,file_path):
    with open(r'%s\%s.txt'%(file_path,name), mode='w', encoding='utf-8') as outFile:
        outFile.write('序号\t汉字\t字频\n')
        i = 0
        for char, frequency in sorted(_dict.items(), key=lambda d:d[1], reverse = True):
            i+= 1
            s = '{0}.\t{1}\t{2}\n'.format(i,char, frequency)
            outFile.write(s)


def getpoetry (poetry_file = r'_txt\全唐诗.txt'):
    poetry_file = poetry_file
    poetry = []
    with open(poetry_file, mode='r', encoding='utf')as file:
        for line in file:
            line = line.strip('\n')
            P = re.split('[，。？]+', line)
            #print(P)
            if  len(P)!=9:
                continue

            if len(P[0])!=len(P[3]):
                continue

            if  len(line)!=48 :
                continue
            poetry.append(line+' ')
    return poetry

def poetrys_and_titles (poetry_file = r'_txt\汇总.txt'):
    poetry_file = poetry_file
    poetrys = []
    titles = []
    with open(poetry_file,mode='r',encoding='utf')as file :
        for line in file :
            #print(line)
            pat = re.compile(r'\d*  (.*?)—.*?《(.*?)》')
            if re.match(pat, line) :
                poem , title = re.match(pat, line).groups()
            else:
               continue
            if len(poem) < 5 or len(poem) > 79:
                continue
            poetrys.append(poem)
            titles.append(title)
    return poetrys,titles
# print(len(poetrys))
# print(len(titles))
#print(poetrys)
# print(titles)

def get_mapping (sortdict):
    words, _ = zip(*sortdict)
    word_num_map = dict(zip(words, range(len(words))))
    #word_num_map.update(dict([('·',-1),('，',-2),('。',-3),(' ',-4)]))
    return word_num_map,words


def poembatch (poetrys_vector,word_map,batchsize=64):
    batch_size = batchsize
    n_chunk = len(poetrys_vector) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poetrys_vector[start_index:end_index]
        length = max(map(len, batches))
        xdata = np.full((batch_size, length), word_map[' '], np.int32)
        for row in range(batch_size):
            xdata[row, :len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:, :-1] = xdata[:, 1:]
        """
        xdata             ydata
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(xdata)
        y_batches.append(ydata)
    return x_batches,y_batches ,n_chunk

def embedding_variable(inputs, rnn_size, word_len):
    with tf.variable_scope('rnn'):
        # 这里选择使用cpu进行embedding
        with tf.device("/cpu:0"):
            # 默认使用'glorot_uniform_initializer'初始化，来自源码说明:
            # If initializer is `None` (the default), the default initializer passed in
            # the variable scope will be used. If that one is `None` too, a
            # `glorot_uniform_initializer` will be used.
            # 这里实际上是根据字符数量分别生成state_size长度的向量
            embedding = tf.get_variable('embedding', [word_len, rnn_size])
            # 根据inputs序列中每一个字符对应索引 在embedding中寻找对应向量,即字符转为连续向量:[字]==>[1]==>[0,1,0]
            lstm_inputs = tf.nn.embedding_lookup(embedding, inputs)
    return lstm_inputs

def softmax_variable(rnn_size, word_len):
    with tf.variable_scope('rnn'):
        w = tf.get_variable("w", [rnn_size, word_len])
        b = tf.get_variable("b", [word_len])
    return w, b

if __name__ == "__main__":
    # dict1,dict2 = count_chinese_word('poetry.txt')
    # save_word_frequency(dict1,'frequency')
    # dict3 = collections.Counter(get_chinese('poetry.txt'))
    # save_word_frequency(dict3,'counter.txt')
    # print(dict2)
    getpoetry()
    print(getpoetry())