import unittest
from w2vembeddings.managedb import ManageDB
from w2vembeddings.w2vemb import EMB


class Test_EMB(unittest.TestCase):
    md = ManageDB()

    def test_list_db(self):
        self.md.list_db()
        self.md.add_file2db('test', 'data/test_corpos.txt', 10, 8)
        self.md.delete_db('test', 10)
        self.md.list_db()

    def test_EMB(self):
        from time import time
        self.md.add_file2db('test2', 'data/test_corpos.txt', 20, 8)
        emb = EMB(name='test', dimensions=10)
        for w in ['的', '哈哈哈', 'vancouver', 'toronto']:
            start = time()
            print('embedding {}'.format(w))
            print(emb.get_vector(w))
            print('took {}s'.format(time() - start))


if __name__ == '__main__':
    unittest.main()