"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@   Instituto Superior Tecnico  @
@@      PRI - 1st Delivery     @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@    Dinis Araújo - 86406     @@
@@    Inês Lacerda - 86436     @@
@@    Maria Duarte - 86474     @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import pairwise_distances
import datetime
import matplotlib.pyplot as plt
from main import *



#########################################################
#      Graph ranking approach: document centrality      #
#########################################################

def min_max_normalization(d, target=1.0):
   top = max(d.values())
   down = min(d.values())
   dic ={}
   for key in d.keys():
       if (top - down) != 0:
            value = (d[key] - down) / (top - down)
       else:
           value = 0
       dic[key] = round(value, 3)
   return dic

# Function that creates the tfidf and tfidf matrix
def tfidf_creation(topic_content, D, compare=False):
    processed = list(dict.fromkeys(topic_content))
    collection = []

    if compare:
        collection.append(' '.join(topic_content))

    for doc in D.keys():
        doc_sentences = ' '.join(D[doc])
        collection.append(doc_sentences)

    tfidf = TfidfVectorizer(vocabulary=processed)
    tfidf_matrix = tfidf.fit_transform(collection)

    return tfidf, tfidf_matrix

def extract_topic_query(q, D, k=3, args=None):
    # @input topic q ∈ Q (identifier), inverted index I, number of top terms for the topic (k), and optional
    # arguments on scoring
    # @behavior selects the top-k informative terms in q against I using parameterizable scoring
    # @output list of k terms (a term can be either a word or phrase)

    tfidf, tfidf_matrix = tfidf_creation(q, D)  # creating the tfif
    features_names = tfidf.get_feature_names()

    dense = tfidf_matrix.todense()
    dense_list = dense.tolist()
    df = pd.DataFrame(dense_list, columns=features_names)
    s_mean = df.mean()  # by doing the mean of every columns in a dataframe we get a series
    s_maximum = s_mean.nlargest(k)  # getting the highest k elements from series

    return list(s_maximum.keys())  # returning the tokens that have the highest relevance

#########################
#     Boolean Query     #
#########################
def boolean_query(q, D, k=3, args=None):
    # @input topic q (identifier), number of top terms k, and index I
    # @behavior maps the inputted topic into a simplified Boolean query using
    # extract topic query and then search for matching* documents using the Boolean IR model
    # @output the filtered collection, specifically an ordered list of document identifiers
    k_terms = extract_topic_query(q, D, k)
    docs_id = {}
    all_docs = []
    all_docs_scores = {}
    for doc in D.keys():
        common_terms = len(list(set(D[doc]).intersection(k_terms)))
        all_docs.append(doc)
        docs_id[doc] = common_terms #took out the 0.8 to always have a result
        all_docs_scores[doc] = common_terms

   # dic = min_max_normalization(docs_id)
    #docs_id_sorted = sorted(dic, key=dic.get, reverse=True)
    #docs = []
    #[docs.append((cosine, dic[cosine])) for cosine in docs_id_sorted]

    return sorted(docs_id, key=docs_id.get, reverse=True)

def ranking(q, D, p=10, args=None):

    _, tfidf_matrix = tfidf_creation(q, D, True)

    cosine_similarities = list(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten())

    dic = {}
    count = 0
    for cosine in cosine_similarities:
        dic[list(D.keys())[count]] = cosine
        count += 1
    dic = min_max_normalization(dic)
    docs_id_sorted = sorted(dic, key=dic.get, reverse=True)

    docs = []
    [docs.append((cosine, dic[cosine])) for cosine in docs_id_sorted]

    return docs[:p], docs_id_sorted, dic

def bm25(topic_content, D, p=10, args=None):

    col = []
    for key in D.keys():
        col.append(D[key])

    bm25 = BM25Okapi(col)
    doc_scores = bm25.get_scores(topic_content)

    dic = {}
    count = 0
    for score in doc_scores:
        dic[list(D.keys())[count]] = score
        count += 1
    dic = min_max_normalization(dic)
    bm25_sorted = sorted(dic, key=dic.get, reverse=True)

    docs = []
    [docs.append((score, dic[score])) for score in bm25_sorted]

    return docs[:p], bm25_sorted, dic

# Function that calculates the rrf score
def rrf_score(doc_ids, docs_rank, docs_bm, p=10):
    rrf_doc_scores = {}
    for doc in doc_ids:
        rrf_doc_scores[doc] = (1 / (50 + docs_rank.index(doc))) + (1 / (50 + docs_bm.index(doc)))

    rrf_sorted = sorted(rrf_doc_scores, key=rrf_doc_scores.get, reverse=True)

    docs = []
    [docs.append((doc, rrf_doc_scores[doc])) for doc in rrf_doc_scores]

    return docs[:p]



#If there is time add fusion metric (extra)
#Implementation Done
def build_graph(D, sim, threshold, args=None):

    # Get documents to build collection
    collection = []
    for doc in D.keys():
        doc_sentences = ""
        for token in D[doc]:
            doc_sentences += " " + token
        collection.append(doc_sentences)

    # Represent documents from the collection as vectors
    vectorizer = TfidfVectorizer(use_idf=False)
    X = vectorizer.fit_transform(collection).todense()

    # Get distances scores for metric sim
    matrix = pairwise_distances(X, metric=sim)

    # Create Graph
    G = nx.Graph()

    # Save scores as edges values in a dictionary
    edges_labeled = {}
    for doc_1 in D.keys(): # Add nodes to graph
        G.add_node(str(doc_1))
        for doc_2 in D.keys():
            index_doc_1 = list(D.keys()).index(doc_1)
            index_doc_2 = list(D.keys()).index(doc_2)
            score = matrix[index_doc_1][index_doc_2]
            if doc_1 != doc_2:
                edges_labeled[(str(doc_1), str(doc_2))] = round(score, 3)

    # Normalize all distance scores with min_max normalization
    # Cosine is a similarity measure, and is already between [0,1]
    if not sim == 'cosine':
        edges_labeled = min_max_normalization(edges_labeled)

   #Add edges to graph
    graph_edges = {}
    for edge in edges_labeled.keys():
        if edges_labeled[edge] < threshold:
            G.add_edge(*edge, weight=edges_labeled[edge])
            graph_edges[edge] = edges_labeled[edge]

    #############  Plotting graph and saving as png only need plot for report
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    pos = nx.spring_layout(G)
    p = nx.drawing.nx_pydot.to_pydot(G)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, labels={node: node for node in G.nodes()}, node_color='red', alpha=0.9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=graph_edges,
                                 font_color='green')

    plt.tight_layout()
    plt.savefig("Graph_" + sim + ".png", format="PNG")
    plt.clf()
    ############### End of plot
    return G
    

def undirected_page_rank(q, D, p, sim, threshold, selection,priors=None):

    topic_content = q_topics_dict[q]
    topic_string = '' + topic_content.title + ' ' + topic_content.desc + ' ' + topic_content.narr
    topic_processed = preprocessing(topic_string)

    # Pre-select p documents to be ranked by page rank
    if selection == 'bm25':
        docs, _, _ = bm25(topic_processed, D, p)
    elif selection == 'svm':
        docs, _, _ = ranking(topic_processed,D,p)
    elif selection == 'rrf':
        docs_bm, all_bm, _ = bm25(topic_processed, D, p)
        docs_svm, all_svm, _ = ranking(topic_processed, D, p)
        docs = rrf_score(list(dict.fromkeys(all_bm + all_svm)), all_svm, all_bm, p)
    elif selection == 'boolean':
        docs = boolean_query(topic_processed,D,3)[:p] #choose to use query of k = 3

    selected_col = {}
    for doc_score in docs:
        if selection != 'boolean':
            doc = doc_score[0]
        else:
            doc = doc_score
        selected_col[doc] = D[doc]

    # Build graph for selected documents
    G = build_graph(selected_col, sim, threshold)


    # Rank documents based on built graph and use corresponding priors
    if priors == 'bm25' and selection == 'bm25':
        page_scores = nx.pagerank(G, personalization=dict(docs), max_iter=100)
    elif priors == 'bm25':
        docs, _, _ = bm25(topic_processed, D, p)
        page_scores = nx.pagerank(G, personalization=dict(docs), max_iter=100)
    elif priors == 'rrf' and selection == 'rrf':
        page_scores = nx.pagerank(G, personalization=dict(docs), max_iter=100)
    elif priors == 'rrf':
        docs_bm, all_bm, _ = bm25(topic_processed, D, p)
        docs_svm, all_svm, _ = ranking(topic_processed, D, p)
        docs = rrf_score(list(dict.fromkeys(all_bm + all_svm)), all_svm, all_bm, p)
        page_scores = nx.pagerank(G, personalization=dict(docs), max_iter=100)
    elif priors == 'degree_cent':
        deg_centrality = nx.degree_centrality(G)
        if sum(deg_centrality.values()) != 0:
            page_scores = nx.pagerank(G, personalization=deg_centrality, max_iter=100)
        else:
            page_scores = nx.pagerank(G)
    elif priors == 'close_cent':
        close_centrality = nx.closeness_centrality(G)
        if sum(close_centrality.values()) != 0:
            page_scores = nx.pagerank(G, personalization=close_centrality, max_iter=100)
        else:
            page_scores = nx.pagerank(G)
    elif priors == 'bet_cent':
        bet_centrality = nx.betweenness_centrality(G, normalized=True,
                                                   endpoints=False)
        if sum(bet_centrality.values()) != 0:
            page_scores = nx.pagerank(G, personalization=bet_centrality, max_iter=100)
        else:
            page_scores = nx.pagerank(G)
    else:
        page_scores = nx.pagerank(G)

    docs_id_sorted = sorted(page_scores, key=page_scores.get, reverse=True)

    res = []
    [res.append((doc_id, page_scores[doc_id])) for doc_id in docs_id_sorted]

    return res[:p]

# todo text processing options
# todo IR models

#########################################################
#                     Main   Code                       #
#########################################################

# Just some input variable to run our experiments with the analyses.py file


stop_words_flag = 'True'
D_PATH = 'rcv1/'
Q_PATH = "topics.txt"
Q_TOPICS_PATH = "topics.txt"
Q_RELS_TEST = "qrels.test.txt"
DATE_TRAIN_UNTIL = datetime.date(1996, 9, 30)
topics_ids = list(range(1, 51))
q_topics_dict = read_topics_file()   # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_test_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic


#folders = os.listdir("rcv1_train/")
#current = os.getcwd()

print('Start reading collection...')
train_xmls, test_xmls, _  = read_xml_files(D_PATH)
print('Done reading collection!')



