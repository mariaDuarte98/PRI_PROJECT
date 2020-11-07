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
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
import os
import datetime
import re

import matplotlib.pyplot as plt

# Creating topic class to represent its structure
class Topic:
    def __init__(self, title, desc, narr):
        self.title = title
        self.desc = desc
        self.narr = narr

# Function that processes qrels file
def red_qrels_file():
    rels_dict = {}
    with open(Q_RELS_TEST) as file:
        for line in file:
            topic_num, doc_id, relevante_bool = line.split(' ')
            relevante_bool = relevante_bool.replace('\n', '')
            if topic_num in rels_dict:
                if relevante_bool == '1':
                    rels_dict[topic_num].append(doc_id)
            else:
                if relevante_bool == '1':
                    rels_dict[topic_num] = [doc_id]
    return rels_dict

# Function that processes topics file
def read_topics_file():
    topics_dic = {}
    with open(Q_TOPICS_PATH, 'r') as file:
        for line in file:
            if '<num>' in line:
                num = (line.replace('<num> Number: ', '')).replace('\n', '')
                num = num.replace(" ", "")
            elif '<title>' in line:
                title = line.replace('<title>', '')
            elif '<desc>' in line:
                desc = ""
                line = file.readline()
                while '<narr>' not in line:
                    if len(line) > 1:
                        desc += " " + line
                    line = file.readline()
                narr = ""
                line = file.readline()
                while '</top>' not in line:
                    if len(line) > 1:
                        narr += " " + line
                    line = file.readline()
             #   if delete_irrel == 'True':
              #      narr = update_narrative(narr)
                topics_dic[num] = Topic(title, desc, narr)
    return topics_dic


#  Function responsible of preprocessing all the tokens:
#  punctuation, lower casing, stopwords, stemming
def preprocessing(content):
    # punctuation
    content = re.sub(r'\W', ' ', content)
    # lower casing
    tokens = nltk.word_tokenize(content.lower())
    # stop words
    if stop_words_flag == 'True':
        stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
    else:
        stop_words = []
    # stemming
    ps = PorterStemmer()
    preprocessed_tokens = []
    for t in tokens:
        if t not in stop_words: # and len(t) > 3:
            preprocessed_tokens.append(ps.stem(t))
    return preprocessed_tokens


######################
# Reading collection #
######################

#   Returns 2 XML Lists:
#       train_xmls, test_xmls
def read_xml_files():
    #folder = os.listdir(D_PATH)
    train_xmls = {}
    test_xmls = {}
    #for folder in folders:
    xml_file_names = os.listdir(D_PATH)
    for xml_file_name in xml_file_names:
        print(xml_file_name)
        if os.path.isfile(os.path.join(D_PATH, xml_file_name)) and xml_file_name.find(
                ".xml") != -1:
            xml_file = ET.parse(D_PATH + xml_file_name)
            year, month, day = [int(x) for x in
                                xml_file.getroot().attrib.get('date').split(
                                    '-')]
            date = datetime.date(year, month, day)
            document = ''
            for tag in ['headline', 'byline', 'dateline']:
                for content in xml_file.getroot().iter(tag):
                    if content.text:
                        document += ' ' + content.text
            for content in xml_file.getroot().iter('text'):
                for paragraph in content:
                    document += ' ' + paragraph.text
            if date <= DATE_TRAIN_UNTIL:
                train_xmls[xml_file.getroot().attrib.get('itemid')] = preprocessing(document)
            else:
                test_xmls[xml_file.getroot().attrib.get('itemid')] = preprocessing(document)

    return train_xmls, test_xmls


###############################################
# Clustering approach: organizing collections #
###############################################

# Note: due to efficiency constraints, please undertake this analysis in Dtrain documents only,
# as opposed to clustering the full RCV1 collection, D.


def clustering(D, args=None):
    # @input document collection D (or topic collection Q), optional arguments on clustering analysis
    # @behavior selects an adequate clustering algorithm and distance criteria to identify the best
    # number of clusters for grouping the D (or Q) documents
    # @output clustering solution given by a list of clusters,
    # each entry is a cluster characterized by the pair (centroid, set of document/topic identifiers)

    '''
    vectorizer = TfidfVectorizer(use_idf=False)
    collection = []
    for doc in D.keys():
        doc_sentences = ""
        for token in D[doc]:
            doc_sentences += " " + token
        collection.append(doc_sentences)
        print(doc_sentences)
        print('next')
    vector_space = vectorizer.fit_transform(collection)
    clustering = AgglomerativeClustering().fit(vector_space.todense())
    print(clustering.labels_) '''

    # k means clustering

    vectorizer = TfidfVectorizer()
    collection = []
    doc_topics =  {}
    for doc in D.keys():
        doc_topics_list = []
        print('docid')
        print(doc)
        doc_sentences = ""
        for token in D[doc]:
            doc_sentences += " " + token
        collection.append(doc_sentences)
        print(doc_sentences)
        print('next')
        doc_topics[doc] = [value for value in q_rels_train_dict.keys() if doc in q_rels_train_dict[value]]

    X = vectorizer.fit_transform(collection)

    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    plt.savefig("plot.png")
    plt.clf()

    # true_k = 4  # number of clusters
    # model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=10)
    # model.fit(X)
    # labels = model.labels_

    # docs_cl = pd.DataFrame(list(zip(D.keys(),doc_topics.values(), labels)), columns=['docId','topicId', 'cluster'])

    # print(docs_cl.loc[docs_cl['cluster'] == 0])
    # print(docs_cl.loc[docs_cl['cluster'] == 1])
    # print(docs_cl.loc[docs_cl['cluster']==2])
    # print(docs_cl.loc[docs_cl['cluster'] == 3])

    exit(0)


def interpret(cluster,D,args):
    # @input cluster and document collection D (or topic collection Q)
    # @behavior describes the documents in the cluster considering both median and me- doid criteria
    # @output cluster description
    exit(0)


def evaluate(q, k, I, args):
    # @input topic q (identifier), number of top terms k, and index I
    # @behavior maps the inputted topic into a simplified Boolean query using
    # extract topic query and then search for matching* documents using the Boolean IR model
    # @output the filtered collection, specifically an ordered list of document identifiers
    exit(0)

#########################################################
# Supervised approach: incorporating relevance feedback #
#########################################################


def training(q,Dtrain,Rtrain,args):
    # @input topic document q ∈ Q, training collection Dtrain,
    # judgments Rtrain, and optional arguments on the classification process
    # @behavior learns a classification model to predict the relevance of
    # documents on the topic q using Dtrain and Rtrain,
    # where the training process is subjected to proper preprocessing,
    # classifier’s selection and hyperparameterization
    # @output q-conditional classification model
    exit(0)


def classify(d,q,M,args):
    # @input document d ∈ Dtest, topic q ∈ Q, and classification model M
    # @behavior classifies the probability of document d to be relevant for topic q given M
    # @output probabilistic classification output on the relevance of document d to the topic t
    exit(0)
    
    
def evaluate(Qtest,Dtest,Rtest,args):
    # @input subset of topics Qtest ⊆ Q, testing document collection Dtest, judgments Rtest,
    # and arguments for the classification and retrieval modules
    # @behavior evaluates the behavior of the IR system in the presence and absence of relevance feedback.
    # In the presence of relevance feedback, training and testing functions are called for each topic
    # in Qtest for a more comprehensive assessment
    # @output performance statistics regarding the underlying classification system
    # and the behavior of the aided IR system
    exit(0)

#########################################################
#      Graph ranking approach: document centrality      #
#########################################################

def build_graph(D,sim,θ,args):
    # @input document collection D, similarity criterion, and minimum similarity threshold θ
    # @behavior computes pairwise similarities for the given document collection
    # using the inputted similarity criterion;
    # maps the pairwise relationships into a weighted undirected graph;
    # and applies the θ threshold in order to remove edges with low similarity scores
    # @output undirected graph capturing document relationships on the basis of their similarity
    exit(0)
    
    
def undirected_page_rank(q,D,p,sim,θ,args):
    # @input topic q, document collection D, number of top documents to return (p),
    # and parameters associated with the graph construction and page rank algorithms
    # @behavior given a topic query q, it applies a modified version of the page rank*
    # prepared to traverse undirected graphs for document ranking
    # @output ordered set of top-p documents – list of pairs (d, score) – in descending order of score
    exit(0)
    
# todo text processing options
# todo IR models

#########################################################
#                     Main   Code                       #
#########################################################

# Just some input variable to run our experiments with the analyses.py file


stop_words_flag = 'True'
D_PATH = "rcv1_rel/"
Q_PATH = "topics.txt"
Q_TOPICS_PATH = "topics.txt"
Q_RELS_TEST = "qrels.train.txt"
DATE_TRAIN_UNTIL = datetime.date(1996, 9, 30)
topics_ids = list(range(1, 5))
q_topics_dict = read_topics_file()   # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_train_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic


rel_docs = list(q_rels_train_dict.values())
print(rel_docs)
#folders = os.listdir("rcv1_train/")
#current = os.getcwd()

train_xmls, test_xmls = read_xml_files()
print(train_xmls)
clustering(train_xmls)


'''

for topic_num in topics_ids:
    if topic_num != 100 and topic_num <= 9:
            topic = "R10" + str(topic_num)  # for example for 3, we have topic R103
    elif topic_num != 100:
        topic = "R1" + str(topic_num)   # for example for 14 we have topic R114
    else:
        topic = "R200"              # for 100 we have topic R200

    D_PATH = "rcv1_r" + str(topic_num) + "/"
    print(topic)
    print(D_PATH)
    DATE_TRAIN_UNTIL = datetime.date(1996, 9, 30)
    train_xmls, test_xmls = read_xml_files()
    clustering(train_xmls)'''




