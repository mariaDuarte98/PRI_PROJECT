import os
import shutil

from main import red_qrels_file
import datetime

train = True
test = True
numberOfTopics = 50
# Place rcv1 folder outside project to avoid git mistakes
fullDatasetPath = "../rcv1/"

DATE_TRAIN_UNTIL = datetime.date(1996, 9, 30)

q_rels_train = red_qrels_file()
q_rels_test = red_qrels_file("qrels.test.txt")
topics_train = []
topics_test = []
i = 0

for topic_list in q_rels_train.values():
    topics_train.append(topic_list)
    i += 1
    if i == numberOfTopics:
        break

i = 0
for topic_list in q_rels_test.values():
    topics_test.append(topic_list)
    i += 1
    if i == numberOfTopics:
        break

folders = os.listdir("../rcv1/")
# Loop on every folder inside rcv_dataset
for folder in folders:
    print("Reading folder " + folder)
    if os.path.isdir("../rcv1/" + folder + "/"):
        xml_files = os.listdir("../rcv1/" + folder + "/")
        for xml_file in xml_files:
            if xml_file.find(".xml") != -1:
                if int(xml_file.split("news")[0]) < 86968:
                    if train:
                        for docInTopic in topics_train:
                            # If it has a corresponding topic
                            if xml_file.split("news")[0] in docInTopic:
                                if not os.path.isdir("./rcv1/"):
                                    os.mkdir("rcv1/")
                                shutil.copy("../rcv1/" + folder + "/" + xml_file, "./rcv1/")
                                break
                else:
                    if test:
                        for docInTopic in topics_test:
                            # If it has a corresponding topic
                            if xml_file.split("news")[0] in docInTopic:
                                if not os.path.isdir("./rcv1/"):
                                    os.mkdir("rcv1/")
                                shutil.copy("../rcv1/" + folder + "/" + xml_file, "./rcv1/")
                                break
