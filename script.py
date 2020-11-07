import os
import shutil
from main import *

folders = os.listdir("rcv1_train/")
current = os.getcwd() 

for folder in folders:
    print("hvfgbvdsfgnhg")
    if os.path.isdir("rcv1_train/" + folder + "/"):
        xml = os.listdir("rcv1_train/" + folder + "/")
        for fich in xml:
            print("\n--------" + fich + "----------\n")
            name = fich.replace("newsML.xml","")
            print('nameeeeeeee')
            print(name)
            for i in range(1,5):
                print(name in rel_docs[i-1])
                if  not os.path.isdir("./rcv1_rel/") and name in rel_docs[i-1]:
                    os.mkdir("rcv1_rel/")
                    print("\nnotexists" + name + "\n")
                    shutil.copy("rcv1_train/" + folder + "/" + fich, "rcv1_rel/")
                elif os.path.isdir("./rcv1_rel/")  and name in rel_docs[i-1]:
                    print("\n" + name + "\n")
                    shutil.copy("rcv1_train/" + folder + "/" + fich, "rcv1_rel/")
                
      


                
                
