import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_tf = [Counter(tokenize(t)) for t in texts]
        self.doc_len = [sum(tf.values()) for tf in self.doc_tf]
        self.N = len(texts)
        self.avgdl = sum(self.doc_len) / self.N

        df = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for w, n in df.items()
        }

    def score(self, query, i):
        s = 0.0
        for w in set(tokenize(query)):
            tf = self.doc_tf[i][w]
            if tf == 0:
                continue
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
            s += self.idf[w] * tf * (self.k1 + 1) / (tf + norm)
        return s

    def search(self, query, k=5):
        scores = [(self.score(query, i), i) for i in range(self.N)]
        scores.sort(reverse=True)
        return scores[:k]