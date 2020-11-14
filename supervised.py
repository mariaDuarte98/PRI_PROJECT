from main import read_xml_files, red_qrels_file, read_topics_file
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import numpy

vectorizer = TfidfVectorizer()

def training(q, Dtrain, Rtrain, args=None):
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
    gnb = GaussianNB()
    q_classification_model = gnb.fit(X, doc_relevance)
    return q_classification_model


def classify(d, q, M, args=None):
    s = ""
    for word in d:
        s += word + " "
    x_test = vectorizer.transform([s]).toarray()
    return M.predict(x_test)


def evaluate(Qtest, Dtest, Rtest, args=None):
    return


q_topics_dict = read_topics_file()  # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_train_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic
train_xmls, test_xmls = read_xml_files()
for q in q_topics_dict.keys():
    classification_model = training(q, train_xmls, q_rels_train_dict)
    for test_file in test_xmls.keys():
        print(classify(test_xmls[test_file], q, classification_model))
