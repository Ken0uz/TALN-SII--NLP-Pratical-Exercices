from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from gensim.matutils import kullback_leibler, hellinger
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models, similarities
#cats loves to play in the sun
# 1. Collection des documents
documents = [
    "Cats love to lounge in the sun.",
    "Dogs are loyal and like to play.",
    "Birds sing early in the morning in the trees.",
    "Fish swim in the calm waters of the ponds.",
    "Horses gallop in the green meadows."
]

# Nettoyage et tokenisation des documents
texts = [[word.lower() for word in doc.split()] for doc in documents]

# 2. Création des représentations vectorielles
dictionary = corpora.Dictionary(texts)
corpus_bow = [dictionary.doc2bow(text) for text in texts]

# TF-IDF
tfidf_model = models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf_model[corpus_bow]

# LDA
lda_model = models.LdaModel(corpus_bow, id2word=dictionary, num_topics=5, random_state=30)
corpus_lda = lda_model[corpus_bow]

# Affichage des topics LDA
print("\n### Topics LDA ###")
topics = lda_model.print_topics(num_words=5)  # Afficher les 5 mots principaux pour chaque topic
for topic in topics:
    print(f"Topic {topic[0]}: {topic[1]}")

# Création des matrices de similarité
similarities_bow = similarities.MatrixSimilarity(corpus_bow)
similarities_tfidf = similarities.MatrixSimilarity(corpus_tfidf)
similarities_lda = similarities.MatrixSimilarity(corpus_lda)

# Affichage des similarités
def print_similarity_results(similarity_matrix, representation_name):
    print(f"\nSimilarités ({representation_name}):")
    for i, sims in enumerate(similarity_matrix):
        sims_list = list(enumerate(sims))
        print(f"Document {i} similarités: {sims_list}")

print_similarity_results(similarities_bow, "BoW")
print_similarity_results(similarities_tfidf, "TF-IDF")
print_similarity_results(similarities_lda, "LDA")

# 3. Calcul des distances entre documents
def jaccard_index(set1, set2):
    """Calcul du Jaccard Index entre deux ensembles."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def calculate_all_distances(corpus, representation_name):
    print(f"\n### Distances pour {representation_name} ###")
    num_docs = len(corpus)
    distances = []

    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            doc1, doc2 = corpus[i], corpus[j]
            h_dist, kl_dist, cosine_dist, euclidean_dist, jaccard_sim = None, None, None, None, None

            if representation_name == "LDA":
                # Hellinger distance
                h_dist = hellinger(doc1, doc2)
                # KL divergence (prend en charge uniquement des distributions non nulles)
                kl_dist = kullback_leibler(doc1, doc2)

            # Cosine similarity
            if representation_name in ["BoW", "TF-IDF", "LDA"]:
                #intialisation des vect denses(similarites de dimensions (le nombres tottal de mots dans le dictionaire globale))
                dense1 = np.zeros(len(dictionary))
                dense2 = np.zeros(len(dictionary))
                #Pour chaque mot dans le document :
                for word_id, count in doc1:
                    dense1[word_id] = count #indice du mot unique dans le doc count poids(freq)
                for word_id, count in doc2:
                    dense2[word_id] = count
                cosine_dist = cosine_similarity([dense1], [dense2])[0][0]

            # Euclidean Distance
            # Create dense vectors with the same length for both documents
            dense1 = np.zeros(len(dictionary))
            dense2 = np.zeros(len(dictionary))
            for word_id, count in doc1:
                dense1[word_id] = count
            for word_id, count in doc2:
                dense2[word_id] = count
            euclidean_dist = np.linalg.norm(dense1 - dense2)

            # Jaccard Index
            set1 = set(word_id for word_id, _ in doc1)
            set2 = set(word_id for word_id, _ in doc2)
            jaccard_sim = jaccard_index(set1, set2)

            distances.append({
                "Paire de documents": f"Doc{i}-Doc{j}",
                "Hellinger Distance": h_dist,
                "KL Divergence": kl_dist,
                "Cosine Similarity": cosine_dist,
                "Euclidean Distance": euclidean_dist,
                "Jaccard Index": jaccard_sim
            })

    return distances


bow_distances = calculate_all_distances(corpus_bow, "BoW")
tfidf_distances = calculate_all_distances(corpus_tfidf, "TF-IDF")
lda_distances = calculate_all_distances(corpus_lda, "LDA")

# Afficher les résultats
print("\n### Distances entre documents pour BoW ###")
print(pd.DataFrame(bow_distances))
print("\n### Distances entre documents pour TF-IDF ###")
print(pd.DataFrame(tfidf_distances))
print("\n### Distances entre documents pour LDA ###")
print(pd.DataFrame(lda_distances))

# 4. Recherche de similarité pour un nouveau document
query_doc = "Cats love to play in the sun.".lower().split()
query_bow = dictionary.doc2bow(query_doc)
query_tfidf = tfidf_model[query_bow]
query_lda = lda_model[query_bow]

print("\n### Recherche de similarités pour un nouveau document ###")
print("\nReprésentation BoW:")
print(sorted(enumerate(similarities_bow[query_bow]), key=lambda x: x[1], reverse=True))
print("\nReprésentation TF-IDF:")
print(sorted(enumerate(similarities_tfidf[query_tfidf]), key=lambda x: x[1], reverse=True))
print("\nReprésentation LDA:")
print(sorted(enumerate(similarities_lda[query_lda]), key=lambda x: x[1], reverse=True))
