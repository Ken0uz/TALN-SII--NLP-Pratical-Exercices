import spacy
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.hdpmodel import HdpModel
from sklearn.decomposition import NMF
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Load spaCy's language model
nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        doc = nlp(text)
        processed_texts.append([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token) > 2])
    return processed_texts

def main():
    # corpus 'paraphraphes'
    texts = [
        #climate change
        """
        Climate change is a pressing issue that affects all countries. 
        Scientists are working tirelessly to understand its causes and effects. 
        Governments around the world are implementing policies to mitigate the impact of global warming.
        """,
        #Technology and communication
        """
        Technology has revolutionized communication, enabling people to connect across the globe. 
        Smartphones and social media platforms have transformed how we interact and share information. 
        However, concerns about privacy and misinformation remain significant challenges.
        """,
        #IA
        """
        The field of artificial intelligence is advancing rapidly. 
        Machine learning algorithms are being applied in healthcare, education, and finance. 
        Despite these advancements, ethical concerns about AI's impact on employment and decision-making continue to be discussed.
        """,
         #Renewable energy 
        """
        Renewable energy sources such as solar, wind, and hydroelectric power are becoming increasingly important. 
        As fossil fuel reserves dwindle, countries are investing heavily in green energy solutions. 
        These initiatives aim to create a sustainable future while reducing greenhouse gas emissions.
        """,
        #Space exploration 
        """
        Space exploration is entering a new era with advancements in rocket technology. 
        Companies like SpaceX and Blue Origin are making space travel more accessible. 
        Missions to the Moon and Mars are inspiring a new generation of scientists and engineers.
        """
    ]

    # Preprocess the texts
    processed_texts = preprocess_text(texts)

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Number of topics
    num_topics = 5

    # LDA Model
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=30)
    print("LDA Topics:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")

    # Coherence Scores
    print("\nCoherence Scores:")

    lda_coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    lda_coherence = lda_coherence_model.get_coherence()
    print(f"LDA Coherence Score: {lda_coherence}")

    

    # LSA Model
    lsa_model = gensim.models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    print("\nLSA Topics:")
    for idx, topic in lsa_model.print_topics(-1):
     print(f"Topic {idx}: {topic}")
    
    print("\nCoherence Score:")
    lsa_coherence_model = CoherenceModel(model=lsa_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    lsa_coherence = lsa_coherence_model.get_coherence()
    print(f"LSA Coherence Score: {lsa_coherence}")

    # Assuming 'corpus' and 'dictionary' are already created

    # HDP Model
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)

    print("\nHDP Topics:")
    num_topics = 6 # Define number of topics to extract

    # Print the top words in each topic
    for idx, topic in hdp_model.print_topics(num_topics=num_topics):
     print(f"Topic {idx}: {topic}")

    # Extract top words for each topic
    num_top_words = 20# Number of words to display
    top_words = []

    for topic in hdp_model.show_topics(formatted=False, num_words=num_top_words):
     top_words.append([word for word, _ in topic[1]])

    # Coherence Score calculation
    coherence_model = CoherenceModel(topics=top_words, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    # Print coherence score
    print("\nCoherence Score:")
    print(f"HDP Coherence Score: {coherence_score}")

    
    # After generating the LDA visualization:
    lda_visualization = gensimvis.prepare(lda_model, corpus, dictionary)

    # Save the visualization to an HTML file using pyLDAvis.save_html
    pyLDAvis.save_html(lda_visualization, 'lda_visualization.html')

if __name__ == '__main__':
    main()