import numpy as np
from collections import defaultdict
import random
import torch
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import argparse
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.decomposition import TruncatedSVD
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_file_name = 'corpus.txt'
num_token = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--neg_sample_num', default=3, type=int)
parser.add_argument('-d', '--dim', default=100, type=int)
parser.add_argument('-w', '--half_window', default=2, type=int)
parser.add_argument('-e', '--epoch', default=40000, type=int)
parser.add_argument('-b', '--batch_size', default=48, type=int)
parser.add_argument('--subsampling', type=bool, default=False)
parser.add_argument('--record_loss_every', type=int, default=20)

args = parser.parse_args()


# 读取文件
def read_corpus(file_name):
    with open(file_name, mode='r', encoding='utf-8') as f:
        sentences = []
        for sent in f.readlines():
            sentences.append(re.sub(pattern=r'(\d+)', repl=num_token, string=sent.strip()))
        sentences = [sent.split(' ') for sent in sentences]
    return sentences


# 下采样概率
def drop_probability(w, frequecy, word2id):
    t = 1e-4
    return 1-np.sqrt(t/frequecy[word2id[w]])


# 统计词频与词表
def init_vocab_and_sampling(corpus):
    word2idx = defaultdict(int)
    for sent in corpus:
        for word in sent:
            word2idx[word] += 1
    frequency = torch.zeros([word2idx.__len__()])
    sample_freq = torch.zeros([word2idx.__len__()])
    for idx, (key, freq) in enumerate(sorted(word2idx.items(), key=lambda x: -x[1])):
        frequency[idx] = freq
        sample_freq[idx] = freq ** 0.75
        word2idx[key] = idx
    frequency /= torch.sum(frequency)
    sample_freq /= torch.sum(sample_freq)
    return word2idx, frequency, sample_freq


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        init_range = 1. / args.dim
        self.center_vectors = nn.Embedding(self.vocab_size, args.dim)
        self.outside_vectors = nn.Embedding(self.vocab_size, args.dim)
        nn.init.uniform_(self.outside_vectors.weight.data, -init_range, init_range)
        nn.init.uniform_(self.center_vectors.weight.data, -init_range, init_range)

    def forward_center(self, in_id):
        return self.center_vectors(in_id).to(device)

    def forward_outside(self, out_id):
        return self.outside_vectors(out_id).to(device)  # [size, 1, dim]

    def forward_neg(self, sample_freq, size):
        neg_id = torch.multinomial(sample_freq, args.neg_sample_num * size, True)
        # [size, K, embed_size]
        return self.outside_vectors(neg_id).view(size, args.neg_sample_num, args.dim).to(device)

    def forward(self, center_word_vec, outside_word_vec, neg_word_vec):
        size, dim = center_word_vec.shape
        center_vec = center_word_vec.view(size, dim, 1)  # [batch_size, dim, 1]
        outside_word_vec = outside_word_vec.view(size, 1, dim)
        pos_loss = F.logsigmoid((torch.bmm(outside_word_vec, center_vec))).squeeze()
        neg_loss = F.logsigmoid(torch.bmm(neg_word_vec.neg(), center_vec)).squeeze().sum(1)
        return -(pos_loss+neg_loss).mean()


def get_items(idx, sentence):
    # 获取窗口内的outside_word
    center_id = sentence[idx]
    context = list(range(idx - args.half_window, idx)) + list(range(idx + 1, idx + args.half_window + 1))
    context = list(filter(lambda i: 0 <= i < len(sentence), context))  # 过滤，如果索引超出范围，则丢弃
    context_id = [sentence[i] for i in context]  # 周围单词

    return [center_id]*len(context_id), context_id


# 获取一个batch的中心词与外词的id列表
def make_one_batch(batch_size, corpus):
    center_ids = []
    context_ids = []
    for _ in range(batch_size):
        sent_id = random.randint(0, corpus.__len__()-1)
        center_word_id = random.randint(0, len(corpus[sent_id])-1)
        result = get_items(center_word_id, corpus[sent_id])
        center_ids.extend(result[0])
        context_ids.extend(result[1])
    item = center_ids, context_ids
    return item


# 计算余弦相似度
def calculate_similarity(w1, w2, word2idx, model):
    if w1 not in word2idx or w2 not in word2idx:
        return 0  # 若不在词表中，则返回0
    id1, id2 = word2idx[w1], word2idx[w2]  # 词的编号
    with torch.no_grad():  # 不记录梯度
        vec1 = (model.forward_center(torch.IntTensor([id1])) + model.forward_outside(torch.IntTensor([id1])))/2
        vec2 = (model.forward_center(torch.IntTensor([id2])) + model.forward_outside(torch.IntTensor([id2])))/2
        similarity = F.cosine_similarity(vec1, vec2, 1)
    return float(similarity)


# 读取需要计算相似度的词
def target_list(file_name):
    with open(file_name, encoding='utf-8', mode='r') as f:
        word_list = f.readlines()
    word_list = [w.strip().split('\t') for w in word_list]
    return word_list


def main():
    # 读取数据
    corpus = read_corpus(data_file_name)
    word2idx, frequency, sample_freq = init_vocab_and_sampling(corpus)
    if args.subsampling:  # subsampling，以一定概率去除高频词
        for i, sent in enumerate(tqdm(corpus)):
            corpus[i] = [word2idx[w] for w in sent if frequency[word2idx[w]]<1e-4 or
                         random.random() < (1-drop_probability(w, frequency, word2idx))]
    else:  # 不进行subsampling，单纯将语料转为编号序列
        for i, sent in enumerate(tqdm(corpus)):
            corpus[i] = [word2idx[w] for w in sent]
    print(corpus[0])
    sgns = Word2Vec(word2idx.__len__(), embed_size=args.dim)
    optimizer = torch.optim.Adam(sgns.parameters())  # 使用Adam优化
    loss_record = []  # 用于绘制loss变化图像
    for epoch in tqdm(range(args.epoch)):
        # 抽取一个batch的center word及其对应的context
        center_id, context_id = make_one_batch(args.batch_size, corpus)
        center_vec = sgns.forward_outside(torch.LongTensor(center_id))  # 转为tensor
        size = center_vec.shape[0]
        context_vec = sgns.forward_outside(torch.LongTensor(context_id))  # 转为tensor
        neg_vec = sgns.forward_neg(sample_freq, size)   # 负采样并转为tensor

        loss = sgns.forward(center_vec, context_vec, neg_vec)
        if epoch % args.record_loss_every == 0:
            loss_record.append(float(loss/args.batch_size)/args.record_loss_every)
        else:
            loss_record[-1] += float(loss/args.batch_size)/args.record_loss_every
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.no_grad()
    plt.plot(range(0, args.record_loss_every*(loss_record.__len__()), args.record_loss_every), loss_record)
    plt.title('loss')
    plt.savefig('loss-'+str(args.batch_size)+'-'+str(args.epoch)+'.png')

    # 输出相似度
    words = target_list('pku_sim_test.txt')
    with open('2019211420'+str(args.batch_size)+'-'+str(args.epoch)+'.txt', mode='w', encoding='utf-8') as f:
        for w in words:
            f.write(w[0]+'\t'+w[1]+'\t'+str(calculate_similarity(w[0], w[1], word2idx, sgns))+'\n')
            
    with open('words_for_visualization.txt',mode='r', encoding='utf-8') as f:
        words_for_visual = [w.strip() for w in f.readlines()]

    # 效果可视化
    vecs_for_visual = np.zeros(shape=(words_for_visual.__len__(), args.dim))
    for i, w in enumerate(words_for_visual):
        idx = word2idx[w]
        vecs_for_visual[i] = ((sgns.forward_center(torch.IntTensor([idx])) +
                              sgns.forward_outside(torch.IntTensor([idx])))/2).cpu().detach().numpy()
    temp = (vecs_for_visual - np.mean(vecs_for_visual, axis=0))
    covariance = 1.0 / vecs_for_visual.shape[0] * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    vecs_for_visual = temp.dot(U[:, 0:2])
    myfont = FontProperties(fname=r'/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.figure(figsize=(10, 10), dpi=120)
    for i in range(len(words_for_visual)):
        color = 'red'
        plt.text(vecs_for_visual[i, 0], vecs_for_visual[i, 1], words_for_visual[i], bbox=dict(facecolor=color, alpha=0.03),
                 fontsize=12, fontproperties=myfont)  # fontproperties = ChineseFont1
    plt.xlim((np.min(vecs_for_visual[:, 0]) - 0.01, np.max(vecs_for_visual[:, 0]) + 0.01))
    plt.ylim((np.min(vecs_for_visual[:, 1]) - 0.01, np.max(vecs_for_visual[:, 1]) + 0.01))
    plt.savefig('text'+str(args.batch_size)+'-'+str(args.epoch)+'.png')


if __name__ == '__main__':
    main()
