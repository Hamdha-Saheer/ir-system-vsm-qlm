import os
import json
import math
import streamlit as st
from collections import defaultdict, Counter

# NLTK Imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# -------------------- NLTK SETUP --------------------
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def normalize(text):
    """Consistent normalization as used in w3school.py, main.py, and coursera_crawler.py."""
    tokens = word_tokenize(text)
    return [
        stemmer.stem(token.lower())
        for token in tokens
        if token.isalpha() and token.lower() not in stop_words
    ]

# =========================================================
# REGION 1: INDEX MERGING (Section 1)
# =========================================================
SOURCE_DIRECTORIES = {
    "w3schools": r"C:\Users\hamdh\Desktop\Lakshika[014]\web_CRAWLER\w3school_pages",
    "udemy": r"C:\Users\hamdh\Desktop\Hemara[012]\web_CRAWLER_pages\web_CRAWLER",
    "coursera": r"C:\Users\hamdh\Desktop\Hamdha[013]\web_CRAWLER\coursera_pages"
}

def merge_indices():
    final_inverted_index = defaultdict(list)
    master_url_mapping = {}
    global_doc_id = 1

    for source_name, folder_path in SOURCE_DIRECTORIES.items():
        if not os.path.exists(folder_path):
            continue

        mapping_path = os.path.join(folder_path, "url_mapping.json")
        local_mapping = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                local_mapping = json.load(f)

        for file_name in os.listdir(folder_path):
            if file_name.startswith("doc_") and file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                local_id = file_name.replace("doc_", "").replace(".txt", "")
                
                # Retrieve original URL from the source's mapping file
                original_url = next((url for url, id in local_mapping.items() if str(id) == local_id), "Unknown URL")

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    tokens = set(normalize(content))
                    for token in tokens:
                        final_inverted_index[token].append(global_doc_id)

                master_url_mapping[global_doc_id] = {
                    "url": original_url,
                    "source": source_name,
                    "local_path": file_path
                }
                global_doc_id += 1

    return final_inverted_index, master_url_mapping

# =========================================================
# REGION 2: VSM IMPLEMENTATION (Section 2)
# =========================================================
def get_vsm_rankings(query, inverted_index, master_mapping):
    N = len(master_mapping)
    query_tokens = normalize(query)
    query_tf = Counter(query_tokens)
    
    query_vec = {}
    for term, count in query_tf.items():
        if term in inverted_index:
            df = len(set(inverted_index[term]))
            idf = math.log10(N / df)
            query_vec[term] = (1 + math.log10(count)) * idf

    scores = {}
    query_norm = math.sqrt(sum(v**2 for v in query_vec.values()))
    if query_norm == 0: return []

    for doc_id, info in master_mapping.items():
        with open(info['local_path'], "r", encoding="utf-8") as f:
            doc_tokens = normalize(f.read())
            doc_tf = Counter(doc_tokens)
            dot_product, doc_norm_sq = 0, 0
            
            for term, count in doc_tf.items():
                if term in inverted_index:
                    df = len(set(inverted_index[term]))
                    idf = math.log10(N / df)
                    weight = (1 + math.log10(count)) * idf
                    doc_norm_sq += weight**2
                    if term in query_vec:
                        dot_product += query_vec[term] * weight
            
            doc_norm = math.sqrt(doc_norm_sq)
            if doc_norm > 0:
                scores[doc_id] = dot_product / (query_norm * doc_norm)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# =========================================================
# REGION 3: QLM IMPLEMENTATION (Section 3)
# =========================================================
def get_qlm_rankings(query, inverted_index, master_mapping, lambd=0.1):
    query_tokens = normalize(query)
    if not query_tokens: return []
    
    total_terms_collection = sum(len(postings) for postings in inverted_index.values())
    collection_probs = {t: len(inverted_index[t]) / total_terms_collection for t in query_tokens if t in inverted_index}
    
    scores = []
    for doc_id, info in master_mapping.items():
        with open(info['local_path'], "r", encoding="utf-8") as f:
            doc_tokens = normalize(f.read())
            doc_len = len(doc_tokens)
            if doc_len == 0: continue
            
            doc_tf = Counter(doc_tokens)
            log_likelihood = 0
            for term in query_tokens:
                p_mle = doc_tf.get(term, 0) / doc_len
                p_coll = collection_probs.get(term, 1e-9)
                smoothed_prob = ((1 - lambd) * p_mle) + (lambd * p_coll)
                log_likelihood += math.log(smoothed_prob)
            scores.append((doc_id, log_likelihood))
            
    return sorted(scores, key=lambda x: x[1], reverse=True)

# =========================================================
# REGION 4: FINAL APPLICATION (Section 4)
# =========================================================
def main():
    st.set_page_config(page_title="Course Search Engine", layout="wide")
    st.title("🔍 Course Search Engine")
    st.markdown("### Integrated Information Retrieval System")

    # Load and merge index
    @st.cache_resource
    def load_engine():
        return merge_indices()

    inverted_index, master_mapping = load_engine()

    query = st.text_input("Enter your search query (e.g., 'python basics'):")

    if query:
        # Requirement: Two Tabs for Output
        tab1, tab2 = st.tabs(["Vector Space Model (VSM)", "Query Likelihood Model (QLM)"])

        with tab1:
            st.subheader("VSM Ranking Results")
            vsm_results = get_vsm_rankings(query, inverted_index, master_mapping)
            if vsm_results:
                for doc_id, score in vsm_results[:10]:
                    data = master_mapping[int(doc_id)]
                    st.write(f"**Score:** {round(score, 4)} | **Source:** {data['source']} | [Course Link]({data['url']})")
            else:
                st.info("No VSM results found.")

        with tab2:
            st.subheader("QLM Ranking Results")
            qlm_results = get_qlm_rankings(query, inverted_index, master_mapping)
            if qlm_results:
                for doc_id, score in qlm_results[:10]:
                    data = master_mapping[int(doc_id)]
                    st.write(f"**Log-Likelihood:** {round(score, 4)} | **Source:** {data['source']} | [Course Link]({data['url']})")
            else:
                st.info("No QLM results found.")

if __name__ == "__main__":
    main()