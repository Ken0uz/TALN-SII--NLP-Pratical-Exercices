# **NLP Practical Exercises**  

This repository contains a set of practical exercises (TPs) designed to teach key concepts in Natural Language Processing (NLP). Each exercise involves hands-on implementation in Python using libraries such as **spaCy**, **scikit-learn**, **Gensim**, and others.  

Additionally, the repository includes a reference book to complement your learning:  
**"Natural Language Processing and Computational Linguistics: A Practical Guide to Text Analysis with Python, Gensim, spaCy, and Keras"** by Bhargav Srinivasa-Desikan (2018).  

---

## **Repository Structure**  

The repository is organized into folders corresponding to each exercise:  

### **1. TP1: Preprocessing**  
- **File:** `preprocess.ipynb`  
- **Objective:**  
  Implement a function to preprocess a given text by performing the following:  
  - Lower casing  
  - Removal of punctuation  
  - Removal of stopwords  
  - Removal of frequent and rare words  
  - Lemmatization  
- **Input:** A sample text composed of multiple sentences.  

---

### **2. TP2: Bigrams and Trigrams**  
- **File:** `bigrams_trigrams.ipynb`  
- **Objective:**  
  Test different vector representation models on a corpus and extract bigrams and trigrams.  

---

### **3. TP4: Dependency Parsing**  
- **File:** `dependency_parsing.ipynb`  
- **Objective:**  
  Train a dependency parser to recognize a custom dependency called “Quantity.”  
  - Tag adjectives like **few, many, some, all, half, whole, enough, numerous** as “Quantity.”  
  - Create training data to include these adjectives.  

---

### **4. TP6: Clustering and Classification**  
- **Files:**  
  - `clustering.py`  
  - `clusteringAndClassifying.ipynb`  
- **Objective:**  
  - Compare **KMeans** and **Naïve Bayes** algorithms for clustering and classification on a dataset of newspapers.  
  - Analyze the performance of models using **Bag of Words (BoW)** and **TF-IDF**.  
  - Test the best-performing model on new text inputs.  

---

### **5. TP8: Word2Vec and Doc2Vec**  
- **File:** `tp8TAL.py`  
- **Objective:**  
  - Preprocess a real-world dataset (e.g., 20 Newsgroups Dataset).  
  - Train a **Word2Vec** model to:  
    - Identify similar words.  
    - Find the odd one out in a list of words.  
  - Train a **Doc2Vec** model to:  
    - Compute document embeddings.  
    - Perform clustering using embeddings.  
  - Compare Word2Vec and Doc2Vec for document similarity.  

---

### **6. TP9: Text Classification Using Logistic Regression**  
- **File:** `tp9.ipynb`  
- **Objective:**  
  - Use the given spam dataset to classify whether a text is spam or not.  
  - Apply preprocessing steps such as stopword removal and lemmatization.  
  - Test various vectorization techniques: **BoW**, **TF-IDF**, **Word2Vec**, and **Doc2Vec**.  
  - Evaluate the performance of each model using metrics like confusion matrix and accuracy.  
  - Discuss how to improve the performance.  

---

## **Additional Resources**  

### **Reference Book**  
A recommended book for understanding the theoretical and practical aspects of NLP:  
- **"Natural Language Processing and Computational Linguistics"** by Bhargav Srinivasa-Desikan (2018)  
  - A practical guide to text analysis with Python, Gensim, spaCy, and Keras.  

---

## **How to Use**  

1. Clone the repository:  
   ```bash
   git clone https://github.com/Ken0uz/NLP-Practical-Exercises.git
