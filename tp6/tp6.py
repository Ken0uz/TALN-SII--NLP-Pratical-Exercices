import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer


# Charger les données de formation
categories = [
 'alt.atheism',
 'talk.religion.misc',
 'comp.graphics',
 'sci.space',
]
dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

labels = dataset.target
true_k = np.unique(labels).shape[0]
data = dataset.data

# Vectorisation avec TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)
X = vectorizer.fit_transform(data)

# Vectorisation avec Bag of Words
vectorizer_bow = CountVectorizer(max_df=0.5, min_df=2, stop_words='english')
X_bow = vectorizer_bow.fit_transform(data)

# Réduction de dimensionnalité avec SVD
n_components = 5
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)

lsa = make_pipeline(svd, normalizer)
X_reduced = lsa.fit_transform(X)

lsa_bow = make_pipeline(svd, normalizer)
X_bow_reduced = lsa_bow.fit_transform(X_bow)

# Clustering avec MiniBatchKMeans
minibatch = True
if minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(X_reduced)

# Clustering KMeans avec Bag of Words
km_bow = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km_bow.fit(X_bow_reduced)

# Récupérer les centres des clusters dans l'espace réduit
original_space_centroids = np.dot(km.cluster_centers_, svd.components_)

# Trier les indices des mots les plus importants pour chaque cluster
order_centroids = original_space_centroids.argsort()[:, ::-1]

# Récupérer les mots associés aux indices
terms = vectorizer.get_feature_names_out()

# Afficher les mots les plus importants pour chaque cluster
for i in range(true_k):
    print(f"Cluster {i}:")
    for ind in order_centroids[i, :10]:
        print(f" {terms[ind]}")

# Réduction de la dimensionnalité pour le graphique avec PCA ou SVD
reduced_data = X_reduced  # Projection SVD pour TF-IDF

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=km.labels_, cmap='viridis')
plt.title("KMeans Clustering avec TF-IDF")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar()
plt.show()

# Affichage des résultats pour Bag of Words
reduced_data_bow = X_bow_reduced  # Projection SVD pour BoW

plt.scatter(reduced_data_bow[:, 0], reduced_data_bow[:, 1], c=km_bow.labels_, cmap='viridis')
plt.title("KMeans Clustering avec Bag of Words")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar()
plt.show()

# 1. Charger les données de test
dataset_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
X_test = dataset_test.data
y_test = dataset_test.target  # Les étiquettes réelles pour le test

# 2. Vectoriser les données de test avec le même TfidfVectorizer
X_test_vectorized = vectorizer.transform(X_test)

# 3. Réduire la dimensionnalité avec le même SVD
X_test_reduced = lsa.transform(X_test_vectorized)

# 4. Prédire les labels des clusters avec le modèle KMeans entraîné
y_pred = km.predict(X_test_reduced)

# Clustering avec Bag of Words
X_test_bow = vectorizer_bow.transform(X_test)
X_test_reduced_bow = lsa_bow.transform(X_test_bow)
y_pred_bow = km_bow.predict(X_test_reduced_bow)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=dataset_test.target_names, yticklabels=dataset_test.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix avec TF-IDF')
plt.show()

conf_matrix_bow = confusion_matrix(y_test, y_pred_bow)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_bow, annot=True, fmt="d", cmap="Blues", xticklabels=dataset_test.target_names, yticklabels=dataset_test.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix avec BoW')
plt.show()

# Calcul des scores ARI et V-Measure
ari = adjusted_rand_score(y_test, y_pred)
print(f"Adjusted Rand Index (ARI) pour TF-IDF: {ari}")

ari_bow = adjusted_rand_score(y_test, y_pred_bow)
print(f"Adjusted Rand Index (ARI) pour BoW: {ari_bow}")

v_measure = v_measure_score(y_test, y_pred)
print(f"V-Measure pour TF-IDF: {v_measure}")

v_measure_bow = v_measure_score(y_test, y_pred_bow)
print(f"V-Measure pour BoW: {v_measure_bow}")

# Clustering hiérarchique
dist = 1 - cosine_similarity(X)
linkage_matrix = ward(dist)

fig, ax = plt.subplots(figsize=(10, 15))  # set size
ax = dendrogram(linkage_matrix, orientation="top")
plt.show()

""" from sklearn.naive_bayes import GaussianNB
# order of labels in `target_names` can be different from `categories`
data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)

target_names = dataset.target_names

# split a training set and a test set
y_train, y_test = dataset.target, data_test.target

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(data_test.data)
X_test = lsa.fit_transform(X_test)

X_test_bow = vectorizer_bow.transform(data_test.data)
X_test_bow = lsa_bow.fit_transform(X_test_bow)

gnb = GaussianNB()
y_pred_NB = gnb.fit(X, y_train).predict(X_test)

gnb_bow = GaussianNB()
y_pred_NB_bow = gnb.fit(X_bow_reduced, y_train).predict(X_test_bow)

from sklearn.svm import SVC
svm = SVC()
y_pred_SVM = svm.fit(X, y_train).predict(X_test)
#y_pred_SVM_bow = svm.fit(X_bow, y_train).predict(X_test_bow)

y_pred_SVM_bow = svm.fit(X_bow, y_train).predict(X_test_bow) """


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# order of labels in `target_names` can be different from `categories`
data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)

target_names = dataset.target_names

# split a training set and a test set
y_train, y_test = dataset.target, data_test.target

# Réduction de dimensionnalité pour les données d'entraînement
X_train_reduced = lsa.fit_transform(X)  # Transformation SVD pour les données d'entraînement
X_train_reduced_bow = lsa_bow.fit_transform(X_bow)  # Transformation SVD pour les données BoW d'entraînement

# Appliquer la réduction de dimensionnalité sur les données de test
X_test_reduced = lsa.transform(X_test_vectorized)  # Transformation SVD pour les données de test TF-IDF
X_test_reduced_bow = lsa_bow.transform(X_test_bow)  # Transformation SVD pour les données de test BoW

# Naive Bayes - TF-IDF
gnb = GaussianNB()
gnb.fit(X_train_reduced, y_train)  # Entraînement sur les données réduites
y_pred_NB = gnb.predict(X_test_reduced)  # Prédiction sur les données de test

# Naive Bayes - BoW
gnb_bow = GaussianNB()
gnb_bow.fit(X_train_reduced_bow, y_train)  # Entraînement sur les données réduites BoW
y_pred_NB_bow = gnb_bow.predict(X_test_reduced_bow)  # Prédiction sur les données de test BoW

# Support Vector Machine - TF-IDF
svm = SVC()
svm.fit(X_train_reduced, y_train)  # Entraînement sur les données réduites
y_pred_SVM = svm.predict(X_test_reduced)  # Prédiction sur les données de test

# Support Vector Machine - BoW
svm_bow = SVC()
svm_bow.fit(X_train_reduced_bow, y_train)  # Entraînement sur les données réduites BoW
y_pred_SVM_bow = svm_bow.predict(X_test_reduced_bow)  # Prédiction sur les données de test BoW

# Évaluer les performances
print(f"Accuracy Naive Bayes (TF-IDF): {accuracy_score(y_test, y_pred_NB)}")
print(f"Accuracy Naive Bayes (BoW): {accuracy_score(y_test, y_pred_NB_bow)}")
print(f"Accuracy SVM (TF-IDF): {accuracy_score(y_test, y_pred_SVM)}")
print(f"Accuracy SVM (BoW): {accuracy_score(y_test, y_pred_SVM_bow)}")
