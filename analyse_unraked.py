from supervised import *
from main import *
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

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

# Function that evaluates boolean query
def eval_boolean_query(t, rel_docs, relevant_len, docs_id):
    f_score = 0

    relevant_retrieved = [doc for doc in docs_id if doc in rel_docs[t]]

    if docs_id:  # there are some topics for each the boolean query might not return anything
        precision = len(relevant_retrieved) / len(docs_id)
    else:
        precision = 0
    recall = len(relevant_retrieved) / relevant_len
    # the weighted harmonic mean of precision and recall
    if precision + recall != 0:
        f_score = (0.5 ** 2 + 1) * ((precision * recall) / (
                    (0.5 ** 2 * precision) + recall))  # 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_score, len(docs_id)

def evaluate(Qtest, Dtest, Rtest, model, args='unranked'):

    # evaluate with relevance feedback
    if args == 'unranked':
        all_precision = []
        all_recall = []
        all_fscore = []
        for topic in Qtest:
            relevant_len = len(Rtest[topic])
            classification_model = training(topic, train_xmls, q_rels_train_dict, model)
            precision, recall, fscore, rel_docs = [], [], [], []
            for doc in Dtest.keys():
                label = classify(Dtest[doc], classification_model)[0]
                if label == 1:
                    rel_docs.append(doc)

            precision, recall, fscore, _ = eval_boolean_query(topic, q_rels_test_dict,relevant_len, rel_docs)
            all_precision.append(precision)
            all_recall.append(recall)
            all_fscore.append(fscore)

        return all_precision, all_recall, all_fscore # accuracy and error rate for all topics
    else:
        return 0

def save_csv(scores, label, args=None):
    if not os.path.isfile("./sec2_Performance.csv"):
        data_frame = pandas.DataFrame([scores],
                                      columns=['precision','recall','fscore'],
                                      index=[label])
        data_frame.to_csv("./sec2_Performance.csv")
    else:
        data_frame = pandas.read_csv("./sec2_Performance.csv", index_col=0)
        data_frame.loc[label] = scores
        data_frame.to_csv("./sec2_Performance.csv")

    return 0

# start of main code



bool_precision = []
bool_recall = []
bool_fscore = []
topics = []

for i in range(1,51):
    if i == 100:
        topic = 'R200'
    elif i <= 9:
        topic = 'R10' + str(i)
    else:
        topic = 'R1' + str(i)
    topics.append(topic)

    p = 15
    print('topic')
    print(topic)
    topic_content = q_topics_dict[topic]
    topic_string = '' + topic_content.title + ' ' + topic_content.desc + ' ' + topic_content.narr
    topic_processed = preprocessing(topic_string)

    relevant_len = len(q_rels_test_dict[topic])

    docs_bool = boolean_query(topic_processed, test_xmls, 3)[:p]
    precision, recall, fscore, _ = eval_boolean_query(topic, q_rels_test_dict, relevant_len, docs_bool)
    bool_precision.append(precision)
    bool_recall.append(recall)
    bool_fscore.append(fscore)


avg_precision = sum(bool_precision) / len(bool_precision)
avg_recall = sum(bool_recall) / len(bool_recall)
avg_fscore = sum(bool_fscore) / len(bool_fscore)
save_csv([avg_precision,avg_recall,avg_fscore],'boolean')

precisions, recalls, fscores = evaluate(topics,test_xmls,q_rels_test_dict, 'RandomForest')
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_fscore = sum(fscores) / len(fscores)
save_csv([avg_precision,avg_recall,avg_fscore],'randomForest')

precisions, recalls, fscores = evaluate(topics,test_xmls, q_rels_test_dict, 'NaiveBayes')
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_fscore = sum(fscores) / len(fscores)
save_csv([avg_precision,avg_recall,avg_fscore],'NaiveBayes')







