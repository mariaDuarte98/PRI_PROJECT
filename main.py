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

###############################################
# Clustering approach: organizing collections #
###############################################

# Note: due to efficiency constraints, please undertake this analysis in Dtrain documents only,
# as opposed to clustering the full RCV1 collection, D.


def clustering(D, args):
    #@input document collection D (or topic collection Q), optional arguments on clustering analysis
    # @behavior selects an adequate clustering algorithm and distance criteria to identify the best number of clusters for grouping the D (or Q) documents
    # @output clustering solution given by a list of clusters, each entry is a cluster cha- racterized by the pair (centroid, set of document/topic identifiers)
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
    # @input topic document q ∈ Q, training collection Dtrain, judgments Rtrain, and optional arguments on the classification process
    # @behavior learns a classification model to predict the relevance of documents on the topic q using Dtrain and Rtrain,
    #where the training process is subjected to proper preprocessing, classifier’s selection and hyperparameterization
    # @output q-conditional classification model
    exit(0)


def classify(d,q,M,args):
    # @input document d ∈ Dtest, topic q ∈ Q, and classification model M
    # @behavior classifies the probability of document d to be relevant for topic q given M
    # @output probabilistic classification output on the relevance of document d to the topic t
    exit(0)
    
    
def evaluate(Qtest,Dtest,Rtest,args):
    # @input subset of topics Qtest ⊆ Q, testing document collection Dtest, judgments Rtest, and arguments for the classification and retrieval modules
    # @behavior evaluates the behavior of the IR system in the presence and absence of relevance feedback. In the presence of relevance feedback, training and testing functions are called for each topic in Qtest for a more comprehen- sive assessment
    # @output performance statistics regarding the underlying classification system and the behavior of the aided IR system
    exit(0)

#########################################################
#      Graph ranking approach: document centrality      #
#########################################################

def build_graph(D,sim,θ,args):
    # @input document collection D, similarity criterion, and minimum similarity th- reshold θ
    # @behavior computes pairwise similarities for the given document collection using the inputted similarity criterion;
    # maps the pairwise relationships into a weighted undirected graph;
    # and applies the θ threshold in order to remove edges with low similarity scores
    # @output undirected graph capturing document relationships on the basis of their similarity
    exit(0)
    
    
def undirected_page_rank(q,D,p,sim,θ,args):
    # @input topic q, document collection D, number of top documents to return (p),
    # and parameters associated with the graph construction and page rank algorithms
    # @behavior given a topic query q, it applies a modified version of the page rank* prepared to traverse undirected graphs for document ranking
    # @output ordered set of top-p documents – list of pairs (d, score) – in descending order of score
    exit(0)
    
# todo text processing options
# todo IR models
