Multi-Model Information Retrieval System (VSM vs QLM)
Project Overview
This group project involved developing a sophisticated search engine that compares two major IR models: the Vector Space Model (VSM) and the Query Likelihood Model (QLM). The system retrieves relevant education-related documents from a merged index of Coursera, Udemy, and W3Schools.

Technical Implementation
Index Merging: Integrated separate inverted indices from three different sources into a single unified master index.

Vector Space Model (VSM): * Implemented TF-IDF (Term Frequency-Inverse Document Frequency) scoring.

Used Cosine Similarity to rank document vectors against user query vectors.

Query Likelihood Model (QLM): * Built language models for each document based on term probabilities.

Applied Dirichlet/Jelinek-Mercer smoothing to handle zero-probability terms.

User Interface: Developed a Streamlit dashboard with a dual-tab layout to compare search results from both models side-by-side.

Technical Stack
Interface: Streamlit

Analysis: Python (Math, NumPy, Pandas)

Data Sources: Coursera, Udemy, W3Schools (Crawled data)
