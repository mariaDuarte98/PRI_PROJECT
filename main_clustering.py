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
from main import read_xml_files, red_qrels_file, read_topics_file
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation
from sklearn_extra.cluster import KMedoids
from kmodes.kmodes import KModes
import hdbscan
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix

import seaborn as sns

###############################################
# Clustering approach: organizing collections #
###############################################

# Note: due to efficiency constraints, please undertake this analysis in Dtrain documents only,
# as opposed to clustering the full RCV1 collection, D.

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    ind = fcluster(linkage_matrix, t = 0.5, criterion = 'distance')
    print(ind)
    n_clusters = len(list(dict.fromkeys(ind).keys()))
    print(n_clusters)

    # Plot the corresponding dendrogram
    den = dendrogram(linkage_matrix, **kwargs)
    # cluster_idxs = {}
    # for c, pi in zip(den['color_list'], den['icoord']):
    #     for leg in pi[1:3]:
    #         i = (leg - 5.0) / 10.0
    #         if abs(i - int(i)) < 1e-5:
    #             cluster_idxs[str(int(i))] = int(c.replace("C", ""))

    return n_clusters


def clustering(D, model):
    # @input document collection D (or topic collection Q), optional arguments on clustering analysis
    # @behavior selects an adequate clustering algorithm and distance criteria to identify the best
    # number of clusters for grouping the D (or Q) documents
    # @output clustering solution given by a list of clusters,
    # each entry is a cluster characterized by the pair (centroid, set of document/topic identifiers)

    vectorizer = TfidfVectorizer(stop_words = "english")
    collection = []
    doc_topics =  {}
    for doc in D.keys():
        doc_topics_list = []
        doc_sentences = ""
        for token in D[doc]:
            doc_sentences += " " + token
        collection.append(doc_sentences)
        doc_topics[doc] = [value for value in q_rels_train_dict.keys() if doc in q_rels_train_dict[value]]

    X = vectorizer.fit_transform(collection).toarray()

    #NAO SEI SE DEVEMOS USAR---------------
    # standard = StandardScaler().fit_transform(X) #Standardizes features by removing the mean and scaling to unit variance
    
    # Find optimal number of components
    pca = PCA().fit(X)
    y = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argwhere(np.diff(np.sign(0.5 - y))).flatten()
    print(idx)
    # reduce the number of dimensions
    pca = PCA(n_components=int(idx)).fit(X)
    pca_2d = pca.transform(X)

    if model == "ap":
        clustering = AffinityPropagation(random_state=None).fit(pca_2d)
        n_clusters = len(clustering.cluster_centers_)

    elif model == "hdbscan":
        clustering = hdbscan.HDBSCAN(algorithm='best', min_cluster_size=2, cluster_selection_epsilon=0.5).fit(pca_2d)
        n_clusters = len(list(dict.fromkeys(clustering.labels_)))
        # clustering.minimum_spanning_tree_.plot(edge_cmap='viridis',
        #                               edge_alpha=0.6,
        #                               node_size=80,
        #                               edge_linewidth=2)
        # plt.title(model + " tree plot")
        # plt.savefig(model + "_tree_plot.png")
        # plt.clf()
        # clustering.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # plt.title(model + " hierarchy plot")
        # plt.savefig(model + "_hierarchy_plot.png")
        # plt.clf()
        # clustering.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        # plt.title(model + " condensed_tree plot")
        # plt.savefig(model + "_condensed_tree_plot.png")
        # plt.clf()

    elif model == "agglomerative":
        # setting distance_threshold=0 ensures we compute the full tree.
        clustering = AgglomerativeClustering(distance_threshold=0, n_clusters = None, linkage = 'complete', affinity = "cosine").fit(pca_2d)

        plt.title('Hierarchical Clustering Dendrogram')
        n_clusters = plot_dendrogram(clustering, truncate_mode='level')
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig("clustering_plots/" + model + "_dendrogram.png")
        plt.clf()
        # print(cluster_indxs)
        # clustering_labels = [cluster_indxs[str(value)] for value in clustering.labels_]
        # n_clusters = len(list(dict.fromkeys(clustering_labels).keys()))

        # clustering = linkage(pca_2d, method='complete')
        # ind = fcluster(clustering, t = 0.5, criterion = 'distance')
        # print(ind)
        # n_clusters = len(list(dict.fromkeys(ind).keys()))
        # print(n_clusters)
        # den = dendrogram(clustering)

        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.savefig("dendrogram2.png")
        # plt.clf()

        clustering = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'complete', affinity = "cosine").fit(pca_2d)

    elif model == "kmeans" or "kmedoids" or "kmodes":
        Sum_of_squared_distances = []
        sil = []
        K = range(2, len(D.keys()))
        for k in K:
            print(k)

            if model == "kmeans":
                clustering = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=5).fit(pca_2d)

            elif model == "kmedoids":
                clustering = KMedoids(n_clusters=k,init='k-medoids++', max_iter=200).fit(pca_2d)

            elif model == "kmodes":
                clustering = KModes(n_clusters=k, init='Cao',max_iter=200, n_init=5).fit(pca_2d)

            # Sum_of_squared_distances.append(clustering.inertia_)

            sil_score = silhouette_score(pca_2d, clustering.labels_, metric = 'cosine')
            sil.append(sil_score)

        plt.plot(K, sil, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method For Optimal k')

        plt.savefig("clustering_plots/" + model + "_sil_plot.png")
        plt.clf()
        
        n_clusters = sil.index(max(sil)) + 2

        if model == "kmeans":
            clustering = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=200, n_init=5).fit(pca_2d)

        elif model == "kmedoids":
            clustering = KMedoids(n_clusters = n_clusters,init='k-medoids++', max_iter=200).fit(pca_2d)

        elif model == "kmodes":
            clustering = KModes(n_clusters = n_clusters, init='Cao', max_iter=200, n_init=5).fit(pca_2d)

    print("\nn_clusterssss\n" + str(n_clusters))

    # Create graph for clustering
    plt.title(model + " plot")
    plt.scatter(pca_2d[:,0],pca_2d[:,1], c=clustering.labels_, cmap='rainbow')

    plt.savefig("clustering_plots/" + model + "_plot.png")
    plt.clf()
    
    docs_cl = pd.DataFrame(list(zip(D.keys(),doc_topics.values(), clustering.labels_)), columns=['docId','topicId', 'cluster'])
    labels = list(sorted(dict.fromkeys(clustering.labels_).keys()))
    docs_cl = docs_cl.sort_values(by=['cluster'])
    docs_cl.to_csv("dataframe.csv")

    for i in labels:
        print("Cluster %d:" % i)
        cluster_docs = docs_cl.loc[docs_cl['cluster'] == i]
        print(cluster_docs)
        # print(cluster_docs.index.tolist())

        interpret(cluster_docs, pca_2d, X, vectorizer)

    evaluate(D, pca_2d, clustering, docs_cl)

    return docs_cl


def interpret(cluster_docs, pca_2d, X, vectorizer):
    # @input cluster and document collection D (or topic collection Q)
    # @behavior describes the documents in the cluster considering both median and medoid criteria
    # @output cluster description

    cluster_doc_ids = cluster_docs.index.tolist()

    # MEDIAN
    # Top terms based on median for all documents

    # Dataframe with tfidf of each term in each document in the cluster
    words_pd = pd.DataFrame(X[cluster_doc_ids], columns=vectorizer.get_feature_names())
    # Calculates median of each term for all documents
    median = words_pd.median()

    print("\nTop 10 terms per cluster:\n")
    top_words = median.nlargest(10)
    print(top_words)
    
    # MEDOID
    # Index of document with the lowest mean distance to the remaining documents in the cluster

    print("Medoid index:\n")
    distances = pairwise_distances(pca_2d[cluster_doc_ids])
    # print(distances)
    mean = []
    for distance in distances:
        mean.append(distance.mean())
    # print(mean)
    medoid_index = np.argmin(mean)
    # print(medoid_index)
    print(cluster_docs.iloc[medoid_index]["docId"])


def evaluate(D, pca_2d, clustering, dataframe):
    # @input document collection D (or topic collection Q), optional arguments on
    # clustering analysis
    # @behavior evaluates a solution produced by the introduced clustering function
    # @output clustering internal (and optionally external) criteria

    sil_score = silhouette_score(pca_2d, clustering.labels_, metric = 'cosine')
    print('silhouette score = {}'.format(sil_score))

    # true_labels = []
    # for label in clustering.labels_:
    #     cluster_docs = docs_cl.loc[docs_cl['cluster'] == label]

    # ar_score = adjusted_rand_score(clustering.labels_, pca_2d)
    # print('adjusted rand score score = {}'.format(ar_score))

    # # compute contingency matrix (also called confusion matrix)
    # contingency_matrix = contingency_matrix(pca_2d, clustering.labels_)
    # # return purity
    # purity =  np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    # print('purity score = {}'.format(purity))

    exit(0)

#########################################################
#                     Main   Code                       #
#########################################################

# Just some input variable to run our experiments with the analyses.py file

folders = os.listdir("rcv1/")

num_topics = [int(folder.replace("rcv1_rel", "")) for folder in folders]

print(sorted(num_topics))
topics = input("Pick number of topics: ")

q_topics_dict = read_topics_file()   # dictionary with topic id: topic(title, desc, narrative) ,for each topic
q_rels_train_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic

D_PATH = "rcv1/rcv1_rel" + topics + "/"
train_xmls, test_xmls = read_xml_files(D_PATH)

models = ["ap", "hdbscan", "agglomerative", "kmeans", "kmedoids", "kmodes"]

print("\n1- ap; \n2- hdbscan; \n3- agglomerative; \n4- kmeans; \n5-kmedoids; \n6- kmodes\n")
model = input("Pick a model (type a number): ")

print("picked " + models[int(model)-1])

if not os.path.isdir("clustering_plots"):
    os.mkdir("clustering_plots")
if not os.path.isdir("clustering_plots/topics_" + topics)

clustering(train_xmls, models[int(model)-1])
