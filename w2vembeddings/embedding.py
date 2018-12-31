import sqlite3
from os import path, makedirs, environ
import requests
import logging
from array import array
from io import StringIO


class Embedding:

    def path(self, p):
        """

        Args:
            p (str): relative path.

        Returns:
            str: absolute path to the file, located in the ``$EMBEDDINGS_ROOT`` directory.

        """
        root = self.root_path()
        return path.join(path.abspath(root), p)

    def root_path(self):
        return environ.get('EMBEDDINGS_ROOT', path.join(environ['HOME'], '.embeddings'))

    @staticmethod
    def initialize_db(fname):
        """

        Args:
            fname (str): location of the database.

        Returns:
            db (sqlite3.Connection): a SQLite3 database with an embeddings table.

        """
        if path.dirname(fname) and not path.isdir(path.dirname(fname)):
            makedirs(path.dirname(fname))
        db = sqlite3.connect(fname)
        c = db.cursor()
        c.execute('create table if not exists embeddings(word text primary key, emb blob)')
        db.commit()
        return db

    def load_memory(self):
        # Read database to tempfile
        tempfile = StringIO()
        for line in self.db.iterdump():
            tempfile.write('%s\n' % line)
        self.db.close()
        tempfile.seek(0)

        # Create a database in memory and import from tempfile
        self.db = sqlite3.connect(":memory:")
        self.db.cursor().executescript(tempfile.read())
        self.db.commit()
        self.db.row_factory = sqlite3.Row

    def __len__(self):
        """

        Returns:
            count (int): number of embeddings in the database.

        """
        c = self.db.cursor()
        q = c.execute('select count(*) from embeddings')
        self.db.commit()
        return q.fetchone()[0]

    def insert_batch(self, db, batch):
        """

        Args:
            batch (list): a list of embeddings to insert, each of which is a tuple ``(word, embeddings)``.

        Example:

        .. code-block:: python

            e = Embedding()
            e.db = e.initialize_db(self.e.path('mydb.db'))
            e.insert_batch([
                ('hello', [1, 2, 3]),
                ('world', [2, 3, 4]),
                ('!', [3, 4, 5]),
            ])
        """
        c = db.cursor()
        binarized = [(word, array('f', emb).tobytes()) for word, emb in batch]
        try:
            c.executemany("insert into embeddings values (?, ?)", binarized)
            db.commit()
        except Exception as e:
            print('insert failed\n{}'.format([w for w, e in batch]))
            raise e

    def __contains__(self, w):
        """

        Args:
            w: word to look up.

        Returns:
            whether an embedding for ``w`` exists.

        """
        return self.lookup(w) is not None

    def clear(self):
        """

        Deletes all embeddings from the database.

        """
        c = self.db.cursor()
        c.execute('delete from embeddings')
        self.db.commit()

    def lookup(self, w):
        """

        Args:
            w: word to look up.

        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.

        """
        c = self.db.cursor()
        q = c.execute('select emb from embeddings where word = :word', {'word': w}).fetchone()
        return array('f', q[0]).tolist() if q else None
