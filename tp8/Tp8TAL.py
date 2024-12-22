from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

# Étape 1 : Chargement et prétraitement
print("Chargement et prétraitement du dataset...")
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))  # Supprime les métadonnées
documents = newsgroups_data.data[:5000]  # Limiter à 5000 documents pour accélérer les calculs
preprocessed_docs = [simple_preprocess(doc) for doc in documents]

# Étape 2 : Entraînement du modèle Word2Vec
print("Entraînement du modèle Word2Vec...")
word2vec_model = Word2Vec(
    sentences=preprocessed_docs, 
    vector_size=50,  #  dimensions des vecteurs
    window=3,        # Réduit la fenêtre contextuelle
    min_count=5,     # Filtre les mots peu fréquents
    sg=1,            # Skip-gram
    epochs=10,        # Réduit le nombre d'itérations
    workers=4        # Multithreading
)

# Word2Vec - Évaluation des mots similaires
target_word = "data"
try:
    print(f"Mots similaires à '{target_word}':", word2vec_model.wv.most_similar(target_word))
except KeyError:
    print(f"Le mot '{target_word}' n'est pas dans le vocabulaire.")

# Word2Vec - Identifier l'intrus
word_list = ["data", "analysis", "cat"]
try:
    print("Intrus détecté :", word2vec_model.wv.doesnt_match(word_list))
except KeyError:
    print("Certains mots de la liste ne sont pas dans le vocabulaire.")

# Étape 3 : Entraînement du modèle Doc2Vec
print("Entraînement du modèle Doc2Vec...")
tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(preprocessed_docs)]
doc2vec_model = Doc2Vec(
    tagged_docs, 
    vector_size=50,  # Diminue les dimensions des vecteurs
    window=3,        # Réduit la fenêtre contextuelle
    min_count=5,     # Filtre les mots peu fréquents
    epochs=10,       # Réduit le nombre d'itérations
    workers=4        # Multithreading
)

# Calculer les embeddings des documents
doc_vectors = np.array([doc2vec_model.dv[str(i)] for i in range(len(tagged_docs))])

# Étape 4 : Clustering avec KMeans
print("Clustering des documents avec KMeans...")
n_clusters = 3  # Diminue le nombre de clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(doc_vectors)

# Étape 5 : Évaluation des clusters
print("Évaluation des clusters...")
silhouette_avg = silhouette_score(doc_vectors, clusters)
print(f"Score de silhouette (Silhouette Score) : {silhouette_avg}")

true_labels = newsgroups_data.target[:5000]  # Limite les étiquettes aux documents utilisés
ari_score = adjusted_rand_score(true_labels, clusters)
print(f"Score ARI (Adjusted Rand Index) : {ari_score}")

# Étape 6 : Comparaison Word2Vec vs Doc2Vec
print("Comparaison entre Word2Vec et Doc2Vec...")
# Word2Vec - Similarité entre deux mots
try:
    similarity_word2vec = word2vec_model.wv.similarity("data", "information")
    print(f"Similarité Word2Vec (data, information) : {similarity_word2vec}")
except KeyError:
    print("Un des mots n'est pas dans le vocabulaire.")

# Doc2Vec - Similarité entre documents
new_doc = "This is a document about machine learning."
inferred_vector = doc2vec_model.infer_vector(simple_preprocess(new_doc))
most_similar_docs = doc2vec_model.dv.most_similar([inferred_vector], topn=3)
print("Documents les plus similaires au nouveau document (Doc2Vec) :", most_similar_docs)
