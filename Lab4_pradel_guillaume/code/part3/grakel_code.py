import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()

        for i, word1 in enumerate(doc):
            G.add_node(vocab[word1], label = vocab[word1])
            for j in range(i + 1, min(i + window_size, len(doc))):
                word2 = doc[j]
                if not G.has_edge(vocab[word1], vocab[word2]):
                    G.add_edge(vocab[word1], vocab[word2])
        
        graphs.append(G)
    
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)



print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


import grakel
# from grakel.utils import graph_from_networkx
# from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

# Transform networkx graphs to grakel representations
G_train = grakel.utils.graph_from_networkx(G_train_nx, node_labels_tag = 'label')
G_test = grakel.utils.graph_from_networkx(G_test_nx, node_labels_tag = 'label')


# # Initialize a Weisfeiler-Lehman subtree kernel
gk = grakel.kernels.WeisfeilerLehman(base_graph_kernel=grakel.kernels.VertexHistogram) 

# # Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test) 


# #Task 13

# # Train an SVM classifier and make predictions

svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)
y_pred = svm.predict(K_test)

# # Evaluate the predictions
print("Accuracy for Weisfeiler-Lehman :", accuracy_score(y_pred, y_test))


# #Task 14


