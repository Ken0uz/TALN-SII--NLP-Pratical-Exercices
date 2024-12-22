from sklearn.datasets import fetch_20newsgroups
categories = [
'alt.atheism',
'talk.religion.misc',
'comp.graphics',
'sci.space',
]
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
import spacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")
# warnings imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans

#import dataset
dataset = fetch_20newsgroups(subset='all', categories=categories,
shuffle=True, random_state=42)
#save labels
labels = dataset.target
#get the unique labels
true_k = np.unique(labels).shape[0]

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
data = dataset.data
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english',
use_idf=True)
X = vectorizer.fit_transform(data)

print("n_samples: %d, n_features: %d" % X.shape)

#Dimensionality Reduction
# Vectorizer results are normalized, which makes KMeans behave better
    # Since LSA/SVD results are not normalized, we have to redo the normalization.

    #If we do not normalize the data, variables with different scaling 
    # will be weighted differently in the distance formula 
    # that is being optimized during training.
n_components = 5
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
#The final X is the input which we will be using. 
# It has been cleaned, TF-IDF transformed, and its dimensions reduced.
X = lsa.fit_transform(X)

#scikit-learn offers two implementations of kmeans:
# either in mini-batches or without
minibatch = True
if minibatch:
   km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
   init_size=1000, batch_size=1000)
else:
   km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
# top words per cluster
print("Clustering sparse data with %s" % km)

original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
   print("Cluster %d:" % i)
   for ind in order_centroids[i, :10]:
      print(' %s' % terms[ind])
print("First method:")
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f "
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
#Note: You might see different results, as machine learning 
# algorithms do not produce the exact same results each time.
#km.predict(X_test) to test our model

#imports the KMeans algorithm from the scikit-learn library and 
# creates an instance of it with three clusters, a random state of 0, 
# and automatic initialization
#KMeans algorithm is a clustering algorithm that groups 
# similar data points together based on their distance from each other
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
#The fit method is then called on the normalized training data 
# to train the KMeans model on the data.
kmeans.fit(X)
print("Second method:")
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(true_k):
   print("Cluster %d:" % i)
   for ind in order_centroids[i, :10]:
      print(' %s' % terms[ind])
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f "
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

















#dataset = fetch_20newsgroups(download_if_missing=True)
#news_list = dataset.data
#news_list = news_list[-1000:]
#print(news_list[1])

#df = pd.DataFrame(np.array(news_list).reshape(-1,1), columns=['text'])

#df["tokens"]=df["text"].apply(
#     lambda x: [ 
 #    preprocess_token(token)
  #   for token in nlp(x)
   #  if is_token_allowed(token)])

#print(df["tokens"].head)


