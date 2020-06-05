import numpy as np
import pandas as pd
from build_index import Index
import pickle


class AntiPlag:
    def __init__(self):
        self._index = Index()

    def add_new_doc(self, data: str, label: str):
        # TODO: The method what can add new document into main dict
        pass

    def train_df(self, df: pd.DataFrame) -> None:
        inp_df = pd.DataFrame()
        inp_df['label'] = df['id_post']
        inp_df['data'] = df['post_body']

        self._index.retrive_df(df)
        self._index.tok_lem_stem(type_op='lemmatize')
        self._index.inverted_index_constr()
        self._index.calculate_tf_idf(test_file=query_doc)

        self._index.save_data('dict')

    def get_res(self, data: str) -> float:
        self._index.tfidf_of_query(data)

        final = {}
        for i, key in enumerate(self._index.all_files.keys()):
            wt = []
            u = 0
            for word in self._index.doc_sim_score.keys():
                if(self._index.doc_sim_score[word][i][1] == 0 and self._index.tfidf_query_doc[u] == 0):
                    p1 = 1
                    p2 = 1
                else:
                    p1 = self._index.doc_sim_score[word][i][1] / \
                        (self._index.doc_sim_score[word][i][1]+self._index.tfidf_query_doc[u])
                    p2 = self._index.tfidf_query_doc[u] / \
                        (self._index.doc_sim_score[word][i][1]+self._index.tfidf_query_doc[u])
                wt.append(self._index.doc_sim_score[word][i][1]
                        * p1+self._index.tfidf_query_doc[u]*p2)
                u = u+1
            v = 0
            d1 = 0
            d2 = 0
            sim = 0
            for word in self._index.doc_sim_score.keys():
                d1 += self._index.tfidf_query_doc[v]*((self._index.tfidf_query_doc[v]+1)/(wt[v]+1))
                d2 += wt[v]*np.log((self._index.tfidf_query_doc[v]+1)/(wt[v]+1))
                v = v+1
            #final[key]=d1
            u = 0
            for word in self._index.doc_sim_score.keys():
                if(self._index.doc_sim_score[word][i][1] == 0 and self._index.tfidf_query_doc[u] == 0):
                    p1 = 1
                    p2 = 1
                else:
                    p1 = self._index.doc_sim_score[word][i][1] / \
                        (self._index.doc_sim_score[word][i][1]+self._index.tfidf_query_doc[u])
                    p2 = self._index.tfidf_query_doc[u] / \
                        (self._index.doc_sim_score[word][i][1]+self._index.tfidf_query_doc[u])
                sim = sim+(p1*d1)+(p2*d2)
                u = u+1
            final[key] = sim


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    ap = AntiPlag()
    ap.train_df(df)