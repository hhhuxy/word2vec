import tqdm
from toolz import sliding_window
from sklearn.metrics.pairwise import cosine_similarity
# SVD分解：    u, sigma, v = torch.svd(A)
import numpy as np
from collections import defaultdict
import torch
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_token = '0'
K = 5  # 窗口大小
end_padding_token = ' <END>'
end_padding_size = K-2
data_file_name = '../corpus.txt'  # 数据文件
word_vocab = defaultdict(int)
inverse_word_vocab = defaultdict(str)


def reduce_to_k_dim(M, k=2):
    print(f"Running SVD over {M.shape[0]} words...")
    u, sigma, v = torch.linalg.svd(M)
    u = u.to("cpu").numpy()
    np.save('u', u)
    sigma = sigma.to("cpu").numpy()
    np.save('sigma', sigma)
    notZero = len(sigma[sigma > 1e-5])
    sig_sum = np.sum(sigma)
    chosen_sum = np.sum(sigma[:k])
    print(f'共有{notZero}个非零奇异值，总和为{sig_sum}\n共选取了{k}个奇异值, 总和为{chosen_sum}，占比{chosen_sum/sig_sum*100}%')
    print('done.')
    print(sigma[:k])
    return np.matmul(u[:, :k], np.diag(sigma[:k]))


def read_corpus(file_name):
    with open(file_name, mode='r', encoding='utf-8') as f:
        sentences = []
        for sent in f.readlines():
            sentences.append(re.sub(pattern=r'(\d+)', repl=num_token, string=sent.strip()))
        return sentences


def get_frequency(data):
    """
    window = K = 5
    Args:
        data: 数字已被替换，[['a', 'b'], [.....], [.....]]
    Returns: None
    """
    array = np.zeros(shape=(word_vocab.__len__(), word_vocab.__len__()), dtype='int')
    for sent in tqdm.tqdm(data):
        for window in sliding_window(K, (sent.strip()+end_padding_token*end_padding_size).split(' ')):
            if window[0] in word_vocab:
                w_c_id = word_vocab[window[0]]
                for word in window[1:]:
                    if word in word_vocab:
                        array[w_c_id][word_vocab[word]] += 1
    array += array.transpose()
    return torch.Tensor(array).to(device)


def read_vocab_from_corpus(lines):
    for l in lines:
        for word in l.split(' '):
            word_vocab[word] += 1


def vocab_process(vocab):
    for id, (key, _) in enumerate(sorted(vocab.items(), key=lambda x: -x[1])):
        vocab[key] = id
        inverse_word_vocab[id] = key


def get_similarity(word1, word2, vecs):
    if word1 not in word_vocab or word2 not in word_vocab:
        return 0
    vec1 = vecs[word_vocab[word1], :].reshape(1, -1)
    vec2 = vecs[word_vocab[word2], :].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def target_list(file_name):
    with open(file_name, encoding='utf-8', mode='r') as f:
        word_list = f.readlines()
    word_list = [w.strip().split('\t') for w in word_list]
    return word_list


def main():
    data = read_corpus(data_file_name)
    read_vocab_from_corpus(data)
    vocab_process(word_vocab)

    vec_sta = get_frequency(data)
    vec_sta = reduce_to_k_dim(vec_sta, 300)
    np.save(file='SVD_vec_result.npy', arr=vec_sta)
    vec_sta = np.load('SVD_vec_result.npy')
    words = target_list('../pku_sim_test.txt')
    with open('2019211420.txt', mode='w', encoding='utf-8') as f:
        for w in words:
            f.write(w[0]+'\t'+w[1]+'\t'+str(get_similarity(w[0], w[1], vec_sta))+'\n')


if __name__ == '__main__':
    main()