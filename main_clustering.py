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
from main import *
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation
from sklearn_extra.cluster import KMedoids
import hdbscan
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment

###############################################
# Clustering approach: organizing collections #
###############################################

def plot_dendrogram(model):
    # Create linkage matrix and then creates the dendrogram
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
    n_clusters = len(list(dict.fromkeys(ind).keys()))

    # Create the corresponding dendrogram
    dendrogram(linkage_matrix)

    return n_clusters

def clustering(collection, model, ids, eval_criteria=None):
    vectorizer = TfidfVectorizer(stop_words = "english")
    X = vectorizer.fit_transform(collection).toarray()
    # Find optimal number of components
    pca = PCA().fit(X)
    y = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argwhere(np.diff(np.sign(0.8 - y))).flatten()
    # reduce the number of dimensions
    pca = PCA(n_components=int(idx)).fit_transform(X)
    if model == "ap":
        clustering = AffinityPropagation(random_state=None, verbose=True).fit(pca)
        n_clusters = len(clustering.cluster_centers_)

    elif model == "hdbscan":
        clustering = hdbscan.HDBSCAN(algorithm='best', min_cluster_size=2).fit(pca)
        n_clusters = len(list(dict.fromkeys(clustering.labels_)))

    elif model == "agglomerative":
        # setting distance_threshold=0 ensures we compute the full tree.
        clustering = AgglomerativeClustering(distance_threshold=0, n_clusters = None, linkage = 'average', affinity = "cosine").fit(pca)
        # gets optimal number of clusters
        n_clusters = plot_dendrogram(clustering)
        # plots dendrogram
        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.savefig(model + "_dendrogram.png")
        # plt.clf()

        clustering = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'average', affinity = "cosine").fit(pca)

    elif model == "kmeans" or "kmedoids":
        sum_of_squared_distances = []
        sil = []
        K = range(2, len(ids))
        for k in K:
            if model == "kmeans":
                clustering = KMeans(n_clusters=k, init='k-means++', max_iter=200).fit(pca)

            elif model == "kmedoids":
                clustering = KMedoids(n_clusters=k,init='k-medoids++', max_iter=200).fit(pca)

            sum_of_squared_distances.append(clustering.inertia_)

            sil_score = silhouette_score(pca, clustering.labels_, metric = 'cosine')
            sil.append(sil_score)

        # Create elbow graph for clustering
        # plt.plot(K, sum_of_squared_distances, 'bx-')
        # plt.xlabel('k')
        # plt.ylabel('Elbow Score')
        # plt.title('Elbow Method For Optimal k')

        # plt.savefig(model + "_elbow_plot.png")
        # plt.clf()

        # Create silhouette graph for clustering
        # plt.plot(K, sil, 'bx-')
        # plt.xlabel('k')
        # plt.ylabel('Silhouette Score')
        # plt.title('Silhouette Method For Optimal k')

        # plt.savefig(model + "_sil_plot.png")
        # plt.clf()
        
        n_clusters = sil.index(max(sil)) + 2

        if model == "kmeans":
            clustering = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=200).fit(pca)

        elif model == "kmedoids":
            clustering = KMedoids(n_clusters = n_clusters,init='k-medoids++', max_iter=200).fit(pca)

    # Create graph for clustering
    # plt.title(model + " plot" + " num clusters: " + str(n_clusters))
    # plt.scatter(pca[:,0],pca[:,1], c=clustering.labels_, cmap='rainbow')

    # plt.savefig(model + "_plot.png")
    # plt.clf()

    # Create dataframe for collection and corresponding cluster
    docs_cl = pd.DataFrame(list(zip(ids, clustering.labels_)), columns=['Id','cluster'])
    # Add to dataframe, the topics associated with each document
    if eval_criteria != None:
        docs_cl['criteria'] = eval_criteria.values()
    return docs_cl, pca, X, vectorizer


def interpret(cluster_docs, pca, X, vectorizer, criteria, num_terms=10):
    final = {}
    cluster_doc_ids = cluster_docs.index.tolist()

    if criteria == "1" or criteria == "3": # median criteria or both
        # MEDIAN
        # Top terms based on median for all documents
        # Dataframe with tfidf of each term in each document in the cluster
        words_pd = pd.DataFrame(X[cluster_doc_ids], columns=vectorizer.get_feature_names())
        # Calculates median of each term for all documents
        median = words_pd.median()
        top_words = median.nlargest(num_terms)
        # removes terms with 0 median values
        top_words = top_words[top_words != 0]
        final['median_criteria'] = top_words.to_dict()
    
    if criteria == "2" or criteria == "3": # medoid criteria or both
        # MEDOID
        # Index of document with the lowest mean distance to the remaining documents in the cluster
        distances = pairwise_distances(pca[cluster_doc_ids])
        mean = [distance.mean() for distance in distances]
        medoid_index = cluster_docs.iloc[np.argmin(mean)]["Id"]
        final['medoid_criteria'] = medoid_index

    return final

def remove_outliers_function(new_dataframe, pca):
    labels = list(dict.fromkeys(new_dataframe['cluster']).keys())
    deleted = []
    for label in labels:
        cluster_docs = new_dataframe.loc[new_dataframe['cluster'] == label]
        cluster_doc_ids = cluster_docs.index.tolist()
        # Deletes outliers from dataframe
        if len(cluster_docs) == 1 or label == -1:
            new_dataframe.drop(cluster_doc_ids, axis=0, inplace = True)
            # saves deleted outliers
            deleted.append(cluster_doc_ids)
    # Deletes outliers from pca for silhouette score
    pca = np.delete(pca, deleted, axis=0)
    return new_dataframe, pca

def compute_ground_truth(new_dataframe, eval_criteria_opt):
    true_labels = []
    for label in new_dataframe['cluster']:
        cluster_docs = new_dataframe.loc[new_dataframe['cluster'] == label]
        eval_criteria = cluster_docs['criteria']

        if eval_criteria_opt == "1":
            a = eval_criteria.to_numpy()
            if (a[0] == a).all(): # all documents in cluster have the same topic
                true_labels.append(label)
            else:
                true_labels.append(-2)

        elif eval_criteria_opt == "2":
            codes = list(eval_criteria)
            result = set(codes[0])
            [result.intersection_update(s) for s in codes[1:]]
            if len(result) > 0: # all documents in cluster have at least one common category code
                true_labels.append(label)
            else:
                true_labels.append(-2)
    return true_labels


def evaluate(dataframe, pca, remove_outliers, external_flag, eval_criteria_opt = None):
    final = {}
    new_dataframe = dataframe.copy()
    if remove_outliers == "t":
        new_dataframe, pca = remove_outliers_function(new_dataframe, pca)

    # The silhouette value is a measure of how similar an object is to its own cluster -> cohesion
    sil_score = silhouette_score(pca, new_dataframe['cluster'], metric = 'cosine')
    final['internal_criteria'] = sil_score

    if external_flag == "t" and remove_outliers == "t":
        true_labels = compute_ground_truth(new_dataframe, eval_criteria_opt)

        # Return clustering accuracy
        accuracy = accuracy_score(true_labels, new_dataframe['cluster'])
        # Return clusterong adjusted rand index
        ar_score = adjusted_rand_score(true_labels, new_dataframe['cluster'])
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix2 = contingency_matrix(true_labels, new_dataframe['cluster'])
        # Find optimal one-to-one mapping between cluster labels and true labels
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix2)
        # Return clustering purity
        purity =  contingency_matrix2[row_ind, col_ind].sum() / np.sum(contingency_matrix2)
        # Return clustering normalized mutual information
        nmi = normalized_mutual_info_score(true_labels, new_dataframe['cluster'])

        final['external_criteria'] = {'accuracy': accuracy, 'adjusted_rand': ar_score, 'purity': purity, 'normalized_mutual_info': nmi}

    return final

#########################################################
#                     Main   Code                       #
#########################################################

# Just some input variable to run our experiments with the analyses.py file

print("Create cluster of topics or documents: \n")
collect = input("1 - Topics; 2 - Documents: ")

collection = []

models = ["ap", "hdbscan", "agglomerative", "kmeans", "kmedoids"]

print("\n1- ap; \n2- hdbscan; \n3- agglomerative; \n4- kmeans; \n5- kmedoids; \n")
model = input("Pick a model (type a number): ")

if collect == "1":
    q_topics_dict = read_topics_file()   # dictionary with topic id: topic(title, desc, narrative) ,for each topic

    for topic in q_topics_dict.keys():
        topic = q_topics_dict[topic]
        topic_string = topic.title + ' ' + topic.desc + ' ' + topic.narr
        topic_processed = preprocessing(topic_string) # preprocesses topic content
        topic_sentences = ""
        for token in topic_processed:
            topic_sentences += " " + token

        collection.append(topic_sentences)

    clustering_dataframe, pca, X, vectorizer = clustering(collection, models[int(model)-1], q_topics_dict.keys())
    external_flag = "f" # topics evaluation has no external criteria

if collect == "2":
    D_PATH = "rcv1_rel5/"
    train_xmls, test_xmls, codes = read_xml_files(D_PATH)

    q_rels_train_dict = red_qrels_file()  # dictionary with topic id: relevant document id ,for each topic

    external_flag = input("\nAdd external criteria in evaluation: [t or f] ")

    if external_flag == "t": # chooses evaluation criteria --> topics or category codes
        eval_criteria_opt = input("\nEvaluate based on topics associated or category codes: [1 or 2] ")

    eval_criteria =  {}
    for doc in train_xmls.keys():
        if external_flag == "t" and eval_criteria_opt == "1":
            topics_assoc = [value for value in q_rels_train_dict.keys() if doc in q_rels_train_dict[value]]
            #only evaluates doc with one associated topic
            if len(topics_assoc) == 1:
                eval_criteria[doc] = topics_assoc[0]
                doc_sentences = ""
                for token in train_xmls[doc]:
                    doc_sentences += " " + token

                collection.append(doc_sentences)

        if external_flag == "f" or (external_flag == "t" and eval_criteria_opt == "2"):
            eval_criteria[doc] = codes[doc]
            doc_sentences = ""
            for token in train_xmls[doc]:
                doc_sentences += " " + token

            collection.append(doc_sentences)
    clustering_dataframe, pca, X, vectorizer = clustering(collection, models[int(model)-1], eval_criteria.keys(), eval_criteria)

interpret_criteria = input("\nSelect median or medoid or both criteria: [1 or 2 or 3] ")

print("\n---------- INTERPRET ----------\n")
labels = list(dict.fromkeys(clustering_dataframe['cluster']).keys())
for i in labels:
    print("Cluster %d:" % i)
    cluster_docs = clustering_dataframe.loc[clustering_dataframe['cluster'] == i]
    criteria = interpret(cluster_docs, pca, X, vectorizer, interpret_criteria)
    print(criteria)

print("\n---------- EVALUATE ----------\n")
# only internal criteria
if external_flag != "t":
    # chooses to remove outliers or not
    remove_outliers = input("\nRemove outliers for evaluation: [t or f] ")
    criteria = evaluate(clustering_dataframe, pca, remove_outliers, external_flag)
    print(criteria)
else: # both criterias
    # remove outliers for external criteria evaluation
    criteria = evaluate(clustering_dataframe, pca, "t", external_flag, eval_criteria_opt)
    print(criteria)