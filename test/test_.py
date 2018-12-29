import random
from collections import namedtuple
from os import path
import os

import zipfile
from tqdm import tqdm
from tencentemb.embedding import Embedding
import ipdb


class TencentEmbedding(Embedding):
    """
    Reference: http://nlp.stanford.edu/projects/glove
    """

    Setting = namedtuple('Setting', ['url', 'd_embs', 'size', 'description'])
    settings = {
        'test': Setting(os.getcwd()+'/data/test_corpos.txt',
                           [20], 8, 'A corpus for test')
    }

    def __init__(self, name='test', d_emb=20, default='none'):
        """

        :param name: name of the embedding to retrieve.
        :param d_emb: embedding dimensions.
        :param default: how to embed words that are out of vocabulary. Can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        assert name in self.settings, '{} is not a valid corpus. Valid options: {}'.format(name, self.settings)
        self.setting = self.settings[name]
        assert d_emb in self.setting.d_embs, '{} is not a valid dimension for {}. Valid options: {}'.format(d_emb, name, self.setting)
        assert default in {'none', 'random', 'zero'}

        self.d_emb = d_emb
        self.name = name
        self.db = self.initialize_db(self.path(path.join('tencent', '{}:{}.db'.format(name, d_emb))))
        self.default = default

        if len(self) < self.setting.size:
            self.clear()
            self.load_word2emb()

    def emb(self, word, default=None):
        if default is None:
            default = self.default
        get_default = {
            'none': lambda: None,
            'zero': lambda: 0.,
            'random': lambda: random.uniform(-0.1, 0.1),
        }[default]
        g = self.lookup(word)
        return [get_default() for i in range(self.d_emb)] if g is None else g

    def load_word2emb(self, batch_size=1000):
        seen = set()

        with open(self.setting.url, 'r') as fin:
            batch = []
            for i, line in tqdm(enumerate(fin), total=self.setting.size):
                if i == 0:
                    continue
                elems = line.rstrip().split()
                vec = [float(n) for n in elems[-self.d_emb:]]
                word = ' '.join(elems[:-self.d_emb])
                if word in seen:
                    continue
                seen.add(word)
                batch.append((word, vec))
                if len(batch) == batch_size:
                    self.insert_batch(batch)
                    batch.clear()
            if batch:
                self.insert_batch(batch)


if __name__ == '__main__':
    from time import time
    emb = TencentEmbedding('test', d_emb=20, default='zero')
    for w in ['的', '哈哈', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        print(emb.emb(w))
        print('took {}s'.format(time() - start))
