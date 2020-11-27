from main import read_xml_files, red_qrels_file, read_topics_file
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.decomposition import PCA

vectorizer = TfidfVectorizer()
pca_model = PCA(n_components=20)


def training(q, Dtrain, Rtrain, args='NaiveBayes'):
    collection = []
    doc_relevance = []
    for doc in Dtrain.keys():
        doc_sentences = ""
        for token in Dtrain[doc]:
            doc_sentences += " " + token
        collection.append(doc_sentences)
        if doc in Rtrain[q]:
            doc_relevance.append(1)
        else:
            doc_relevance.append(0)

    X = vectorizer.fit_transform(collection).toarray()
    X = pca_model.fit_transform(X)
    #X = pca.transform(X)
    print(X)
    if args == 'NaiveBayes':
        gnb = GaussianNB()
        return gnb.fit(X, doc_relevance)

    elif args == 'RandomForest':
        clf = RandomForestClassifier() # entropy also ; numer of tree default is 100, change parameters for the report
        return clf.fit(X, doc_relevance)


def classify(d, q, M, args=None):
    s = ""
    for word in d:
        s += word + " "
    x_test = vectorizer.transform([s]).toarray()
    print(len(x_test))
    print(x_test)
   # pca = pca_model.fit(x_test)
    x_test = pca_model.transform(x_test)
    print(len(x_test))
    print(x_test)
    if args == 'prob':
        return M.predict_proba(x_test) # probabilistic output
    else:
        return M.predict(x_test)


def rank_docs(D,q,M,p):
    prob_docs = {}
    for doc in D.keys():
        prob = classify(D[doc], q, M, 'prob')[0][0]
        prob_docs[doc] = prob


    return sorted(prob_docs, key=prob_docs.get, reverse=True)[:p]

def evaluate(Qtest, Dtest, Rtest, M, args=None):

    # evaluate with relevance feedback
    labels_pred, labels_test, accuracies, error_rate = [], [], [], []
    for i in range(0,1): #just to test with current dataset that only considers 10 topics
        topic = list(Qtest.keys())[i]
        for doc in Dtest.keys():
            label = classify(Dtest[doc], topic, M)[0]
            labels_pred.append(label)
            if doc in Rtest[topic]:
                labels_test.append(1)
            else:
                labels_test.append(0)
        accuracy = metrics.accuracy_score(labels_test, labels_pred)
        accuracies.append(accuracy)
        error_rate.append(1 - accuracy)

    return accuracies, error_rate # accuracy and error rate for all topics



D_PATH = 'rcv1/rcv1_rel10/'
q_topics_dict = read_topics_file()  # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_train_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic
q_rels_test_dict = red_qrels_file("qrels.test.txt")  # dictionary with topic id: relevant document id ,for each topic
train_xmls, test_xmls = read_xml_files(D_PATH)
#for q in q_topics_dict.keys():
q = 'R101'
#print(len(test_xmls.keys()))
classification_model = training(q, train_xmls, q_rels_train_dict)
#print('hedefef')

#print(rank_docs(test_xmls,q,classification_model,10))

#result = classify(test_xmls, q, classification_model,'prob')
#print('hedefef')
#print(result)




print(evaluate(q_topics_dict, test_xmls, q_rels_test_dict, classification_model, args=None))

'''
print(classification_model)
for test_file in test_xmls.keys():
    result = classify(test_xmls[test_file], q, classification_model)
    print('hedefef')
    print(result)
'''
