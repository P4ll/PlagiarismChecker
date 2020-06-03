from build_index import Index
import numpy as np


def get_result(data):
    query_doc = data
    query_doc1 = data[:-4]
    #print(query_doc1)
    inn = Index()
    inn.retrieve_file()
    inn.tok_lem_stem(type_op='lemmatize')
    inn.inverted_index_constr()
    inn.calculate_tf_idf(test_file=query_doc)
    inn.tfidf_of_query(query_doc1)
    #print(inn.tfidf_query_doc)
    final = {}
    for i, key in enumerate(inn.all_files.keys()):
        wt = []
        u = 0
        for word in inn.doc_sim_score.keys():
            if(inn.doc_sim_score[word][i][1] == 0 and inn.tfidf_query_doc[u] == 0):
                p1 = 1
                p2 = 1
            else:
                p1 = inn.doc_sim_score[word][i][1] / \
                    (inn.doc_sim_score[word][i][1]+inn.tfidf_query_doc[u])
                p2 = inn.tfidf_query_doc[u] / \
                    (inn.doc_sim_score[word][i][1]+inn.tfidf_query_doc[u])
            wt.append(inn.doc_sim_score[word][i][1]
                      * p1+inn.tfidf_query_doc[u]*p2)
            u = u+1
        v = 0
        d1 = 0
        d2 = 0
        sim = 0
        for word in inn.doc_sim_score.keys():
            d1 += inn.tfidf_query_doc[v]*((inn.tfidf_query_doc[v]+1)/(wt[v]+1))
            d2 += wt[v]*np.log((inn.tfidf_query_doc[v]+1)/(wt[v]+1))
            v = v+1
        #final[key]=d1
        u = 0
        for word in inn.doc_sim_score.keys():
            if(inn.doc_sim_score[word][i][1] == 0 and inn.tfidf_query_doc[u] == 0):
                p1 = 1
                p2 = 1
            else:
                p1 = inn.doc_sim_score[word][i][1] / \
                    (inn.doc_sim_score[word][i][1]+inn.tfidf_query_doc[u])
                p2 = inn.tfidf_query_doc[u] / \
                    (inn.doc_sim_score[word][i][1]+inn.tfidf_query_doc[u])
            sim = sim+(p1*d1)+(p2*d2)
            u = u+1
        final[key] = sim

    print('all words')
    print(inn.all_words)
    
    print('inverted index')
    print(inn.inverted_index)

    print('doc_sim_score')
    print(inn.doc_sim_score)

    print('all_files')
    print(inn.all_files)

    print('dict_list')
    print(inn.dict_list)

    print('tfidf query doc')
    print(inn.tfidf_query_doc)

    print('dict_lem_or_stem')
    print(inn.dict_lemm_or_stem)

    print('ultimate_sim')
    print(inn.ultimate_sim)
    return final
