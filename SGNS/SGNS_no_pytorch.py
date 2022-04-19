import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import argparse

data_file_name = 'corpus.txt'
frequency = None

num_token = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--neg_sample_num', default=3, nargs='?', type=int)
args = parser.parse_args()


# 获取语料
def read_corpus(file_name):
    with open(file_name, mode='r', encoding='utf-8') as f:
        sentences = []
        for sent in f.readlines():
            sentences.append(re.sub(pattern=r'(\d+)', repl=num_token, string=sent.strip()))
        return sentences


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


class Word2Vec:
    def __init__(self, corpus, args, window_half_size=2, table_size=1000000):
        self.word2idx = defaultdict(int)
        self.sample_table = [0] * table_size
        self.table_size = table_size
        self.corpus = [c.strip().split(' ') for c in corpus]
        self.half_window = window_half_size
        self.frequency = None
        self.args = args

    # 获取词表，逆词表，采样表
    def init_vocab_and_sampling(self):
        for sent in self.corpus:
            for word in sent:
                self.word2idx[word] += 1
        self.frequency = np.zeros(shape=(self.word2idx.__len__()), dtype='float')
        for idx, (key, freq) in enumerate(sorted(self.word2idx.items(), key=lambda x: -x[1])):
            self.frequency[idx] = freq ** 0.75
            self.word2idx[key] = idx
        print(f'total word count: {np.sum(self.frequency)}')

        self.frequency /= np.sum(self.frequency)
        freq_mat = self.frequency
        freq_mat = np.cumsum(freq_mat) * self.table_size

        j = 0
        for i in range(self.table_size):
            while i > freq_mat[j]:
                j += 1
            self.sample_table[i] = j

    # 随机取一个窗口, return: center_word, context（context不含center_word）
    def getRandomContext(self):
        # 随机抽取一个句子
        sentID = random.randint(0, len(self.corpus) - 1)
        sent = self.corpus[sentID]
        # 随机选取一个中心词
        wordID = random.randint(0, len(sent) - 1)
        # 计算context下标范围，并获取context（此处context不包含中心词）
        context = sent[max(0, wordID - self.half_window):wordID]
        if wordID + 1 < len(sent):
            context += sent[wordID + 1:min(len(sent), wordID + self.half_window + 1)]
        center_word = sent[wordID]
        context = [w for w in context if w != center_word]
        # 确保context不为空（若句子仅一个词会导致该情况发生），为空则重新抽取
        if len(context) > 0:
            return center_word, context
        else:
            return self.getRandomContext()

    # 采样1个样本(下采样subsampling)
    def subsampling(self):
        idx = self.sample_table[random.randint(0, len(self.sample_table) - 1)]
        p_discard = 1 - np.sqrt((1e-4 / self.frequency[idx]))
        if random.uniform(0, 1) < p_discard:
            return self.subsampling()
        return idx

    # 采样k个不同于outside_word的负样本
    def getNegativeSamples(self, outsideWordIdx, K):
        negSampleWordIndices = [None] * K
        for k in range(K):
            new_idx = self.subsampling()
            while new_idx == outsideWordIdx:
                new_idx = self.subsampling()
            negSampleWordIndices[k] = new_idx
        return negSampleWordIndices

    # 对于给定的center word，进行负采样loss与梯度计算
    def negSamplingLossAndGradient(self, centerWordVec, outsideWordIdx, outsideVectors, K=5):
        """
        Args:
            centerWordVec:  当前词
            outsideWordIdx: context中其他的词的index
            outsideVectors: context中其他词组成的向量 word_num*embedding_len
            K: 采样个数
        """
        # 采样k个负样本
        negSampleWordIndices = self.getNegativeSamples(outsideWordIdx, K)
        # 计算loss
        sig_neg_uvc = sigmoid(-np.dot(outsideVectors[negSampleWordIndices], centerWordVec))  # sigmoid(U·v_w)
        sig_pos = sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))  # sigmoid(u_o·v_w)
        loss = -np.log(sig_pos) - np.sum(np.log(sig_neg_uvc))
        # 计算梯度
        gradCenterVec = - outsideVectors[outsideWordIdx] * (1 - sig_pos) + \
                        np.matmul(1 - sig_neg_uvc, outsideVectors[negSampleWordIndices])  # loss对于v_w的梯度
        gradOutsideVecs = np.zeros(shape=outsideVectors.shape)
        # 由于负采样可能采样到相同的词，故需遍历采样所得词
        for k in negSampleWordIndices:
            gradOutsideVecs[k] += centerWordVec * (1 - sigmoid(-np.dot(outsideVectors[k], centerWordVec)))
        gradOutsideVecs[outsideWordIdx] = - centerWordVec * (1 - sig_pos)  # loss对于u_o的梯度

        return loss, gradCenterVec, gradOutsideVecs

    # 给定中心词，采取skip-gram的策略训练
    def skip_gram(self, currentCenterWord, outsideWords, centerWordVectors, outsideVectors):
        """
        Arguments:
        currentCenterWord -- str
        outsideWords -- list(str)
        centerWordVectors -- np.ndarray(word_num * dim)
        outsideVectors -- np.ndarray(word_num * dim)
        """
        loss = 0.0
        gradCenterVectors = np.zeros(centerWordVectors.shape)
        gradOutsideVectors = np.zeros(outsideVectors.shape)

        t = self.word2idx[currentCenterWord]
        centerWordVec = centerWordVectors[t]
        for outsideWord in outsideWords:
            result = self.negSamplingLossAndGradient(centerWordVec, self.word2idx[outsideWord], outsideVectors,
                                                     K=self.args.neg_sample_num)
            loss += result[0]
            gradCenterVectors[t] += result[1]
            gradOutsideVectors += result[2]
        return loss/outsideWords.__len__(), gradCenterVectors, gradOutsideVectors

    def sgd(self, center_word_vecs, outside_word_vecs, iteration, step, batch_size):
        loss_record = []
        for iter in tqdm(range(iteration)):
            loss = 0.
            grad_center = np.zeros(center_word_vecs.shape)
            grad_outside = np.zeros(outside_word_vecs.shape)
            for i in range(batch_size):
                center_word, context = self.getRandomContext()
                ret = self.skip_gram(center_word, context, center_word_vecs, outside_word_vecs)
                loss += ret[0]
                grad_center += ret[1]
                grad_outside += ret[2]
            if iter % 5 == 0:
                loss_record.append(float(loss / batch_size) / 5)
            else:
                loss_record[-1] += float(loss / batch_size) / 5
            center_word_vecs -= step * grad_center
            outside_word_vecs -= step * grad_outside
        plt.figure()
        x = range(0, loss_record.__len__()*5, 5)
        plt.plot(x, loss_record)
        plt.show()


def main():
    corpus = read_corpus(r'../corpus.txt')
    sgns = Word2Vec(corpus, args=args, window_half_size=2, table_size=4000000)
    sgns.init_vocab_and_sampling()
    wordvec_dim = 35

    center_word_vecs = (np.random.rand(sgns.word2idx.__len__(), wordvec_dim) - 0.5) / wordvec_dim
    outside_word_vecs = (np.random.rand(sgns.word2idx.__len__(), wordvec_dim) - 0.5) / wordvec_dim
    sgns.sgd(center_word_vecs, outside_word_vecs, iteration=2000, step=0.1, batch_size=48)
    np.save('centerword_vecs2000.npy', center_word_vecs)
    np.save('outside_word_vec2000.npy', outside_word_vecs)


if __name__ == '__main__':
    main()
