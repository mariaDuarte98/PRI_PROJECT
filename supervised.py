import PRI_PROJECT.main as primain


def training(q, Dtrain, Rtrain, args=None):
    return


def classify(d, q, M, args=None):
    return


def evaluate(Qtest, Dtest, Rtest, args=None):
    return


q_topics_dict = primain.read_topics_file()  # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_train_dict = primain.red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic
train_xmls, test_xmls = primain.read_xml_files()
training(train_xmls)
