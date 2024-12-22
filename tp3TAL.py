import spacy
import random
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags

# Initializser le blank English model
nlp = spacy.blank("en")

#  ajout du  tagger au pipeline si il n'exist pas 
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")

# Ajout du label "TECH"
ner.add_label("TECH")
# definir l'ensemble d'entrainement 
TRAIN_DATA = [
    ("I'm learning Python daily", {"entities": [(13, 19, "TECH")]}),
    ("Python is a powerful programming language", {"entities": [(0, 6, "TECH")]}),
    ("Many developers use Python for web development", {"entities": [(21, 27, "TECH")]}),
    ("Python has many libraries for data science", {"entities": [(0, 6, "TECH")]}),
    ("Artificial intelligence can be implemented using Python", {"entities": [(42, 48, "TECH")]}),
    ("She writes Python code every day", {"entities": [(12, 18, "TECH")]}),
    ("Python is popular in machine learning", {"entities": [(0, 6, "TECH")]}),
    ("I created my first app with Python", {"entities": [(27, 33, "TECH")]}),
    ("Python can handle large data sets easily", {"entities": [(0, 6, "TECH")]}),
    ("We use Python for our backend services", {"entities": [(7, 13, "TECH")]}),
    ("Python supports multiple programming paradigms", {"entities": [(0, 6, "TECH")]}),
    ("Python scripts are widely used in automation", {"entities": [(0, 6, "TECH")]}),
    ("My favorite language is Python", {"entities": [(23, 29, "TECH")]}),
    ("The company decided to switch to Python for analytics", {"entities": [(29, 35, "TECH")]}),
    ("Python is used in data analysis and visualization", {"entities": [(0, 6, "TECH")]}),
    ("Python has a simple syntax which is easy to learn", {"entities": [(0, 6, "TECH")]}),
    ("Learning Python can boost your career", {"entities": [(9, 15, "TECH")]}),
    ("Python is an essential skill for data scientists", {"entities": [(0, 6, "TECH")]}),
    ("Python and R are both popular languages for data analysis", {"entities": [(0, 6, "TECH")]}),
    ("Developers often choose Python for rapid prototyping", {"entities": [(24, 30, "TECH")]}),
    ("Python excels in data science and AI applications", {"entities": [(0, 6, "TECH")]}),
    ("Many scientists prefer Python for research", {"entities": [(19, 25, "TECH")]}),
]

# entrainement du NER tagger
optimizer = nlp.initialize() #ajuster les poids du model pour min les losses
for i in range(50):  # 50 iterations 
    random.shuffle(TRAIN_DATA)#melanger a chque it l'ordre des exemples
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        # convertir les entités annotées en un format compatible avec le modèle NER,
        tags = offsets_to_biluo_tags(doc, annotations["entities"])
        example = Example.from_dict(doc, {"entities": tags})
        nlp.update([example], sgd=optimizer, losses=losses)
    #print(f"Iteration {i+1}, Losses: {losses}")

# Sauvgarder le model entrainer 
nlp.to_disk("ner_model_tech")

# Tester sur de nouvelles 
test_sentences = [
    "I'm learning Python",
    "Python is great for data science",
    "Many tech companies use Python for machine learning",
    "She loves coding in Python every day",
    "Python is widely used in artificial intelligence",
]

for sentence in test_sentences:
    doc = nlp(sentence)
    print("Sentence:", sentence)
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print()
