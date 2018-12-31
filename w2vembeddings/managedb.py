from tqdm import tqdm
from w2vembeddings.embedding import Embedding
from os import path
import os
import shutil


class ManageDB(Embedding):

    def add_file2db(self, name, file_path, dimensions, size, description='file to sqlite3 db'):
        db_path = self.path(path.join(name, '{}:{}.db'.format(name, dimensions)))
        db = self.initialize_db(db_path)

        batch_size = 1000
        seen = set()

        with open(file_path, 'r') as fin:
            batch = []
            for i, line in tqdm(enumerate(fin), total=size):
                if i == 0:
                    continue
                elems = line.rstrip().split()
                vec = [float(n) for n in elems[-dimensions:]]
                word = ' '.join(elems[:-dimensions])
                if word in seen:
                    continue
                seen.add(word)
                batch.append((word, vec))
                if len(batch) == batch_size:
                    self.insert_batch(db, batch)
                    batch.clear()
            if batch:
                self.insert_batch(db, batch)

    def list_db(self):
        root_path = self.root_path()
        for db_n in [f for f in os.listdir(root_path) if f != '.DS_Store']:
            db_m = [f for f in os.listdir(root_path + '/' + db_n) if f != '.DS_Store']
            print(db_n, db_m)

    def delete_db(self, name, dimensions):
        db_path = self.path(path.join(name, '{}:{}.db'.format(name, dimensions)))
        try:
            os.remove(db_path)
        except Exception as e:
            pass

