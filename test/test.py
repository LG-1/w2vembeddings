import unittest
from w2vembeddings.managedb import ManageDB
from w2vembeddings.w2vemb import EMB


class Test_EMB(unittest.TestCase):
    md = ManageDB()

    def list(self):
        self.md.list_db()
        self.md.delete_db('test', 10)
        self.md.list_db()
        self.md.add_file2db('test', 'data/test_corpos.txt', 10, 8)

    def get_v(self):
        from time import time
        emb = EMB(name='test', dimensions=20)
        for w in ['的', '哈哈哈', 'vancouver', 'toronto']:
            start = time()
            print('embedding {}'.format(w))
            print(emb.get_vector(w))
            print('took {}s'.format(time() - start))


if __name__ == '__main__':
    unittest.main()