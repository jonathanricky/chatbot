import streamlit as st
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

CONTENT_LIST_PATHS = [
    "C:/Coding/VSCode/chatbot/output/abus_elektro-seilzug_programm/auto/abus_elektro-seilzug_programm_content_list.json",
    "C:/Coding/VSCode/chatbot/output/abus_hb-system_programm/auto/abus_hb-system_programm_content_list.json",
    "C:/Coding/VSCode/chatbot/output/abus_laufkran_programm/auto/abus_laufkran_programm_content_list.json"
]
IMAGES_BASE_DIRS = [
    "C:/Coding/VSCode/chatbot/output/abus_elektro-seilzug_programm/auto/images",
    "C:/Coding/VSCode/chatbot/output/abus_hb-system_programm/auto/images",
    "C:/Coding/VSCode/chatbot/output/abus_laufkran_programm/auto/images"
]

@st.cache_data(show_spinner=False)
def load_content(paths):
    """Load JSON content from the given paths."""
    content = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as file:
                content.extend(json.load(file))
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
    return content

@st.cache_data(show_spinner=False)
def extract_content(content_list, image_dirs):
    """Extract texts and image paths from the content list."""
    texts, image_paths = [], []
    image_dir_mapping = {os.path.basename(dir): dir for dir in image_dirs}
    for item in content_list:
        if item["type"] == "text" and item["text"].strip():
            texts.append(item["text"].strip())
            image_paths.append(None)
        elif item["type"] == "image":
            caption = " ".join(item.get("img_caption", [])).strip() or "No caption available"
            texts.append(caption)
            img_dir = os.path.dirname(item["img_path"])
            base_dir = image_dir_mapping.get(os.path.basename(img_dir), image_dirs[0])
            image_paths.append(os.path.normpath(os.path.join(base_dir, item["img_path"])))
    return texts, image_paths

def initialize_faiss_index(embeddings):
    """Initialize and return a FAISS index with the given embeddings."""
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    return index

@st.cache_resource(show_spinner=False)
def load_models_and_index(texts):
    """Load NLP models, compute embeddings, and initialize the FAISS index."""
    # Load models
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qa_model = pipeline("question-answering", model="deepset/gelectra-base-germanquad")
    
    # Compute embeddings and create FAISS index
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = initialize_faiss_index(embeddings)
    return embedding_model, reranker, qa_model, index

def search_and_answer(query, texts, embedding_model, reranker, qa_model, index, k=5, confidence_threshold=0.01):
    """Search for the query in the FAISS index and return an answer using the QA model."""
    st.write("-" * 50)
    st.write(f"Searching for: **{query}** (retrieving top {k * 2} results)")
    
    # Encode query and search index
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k * 2)
    
    # Retrieve texts from indices
    retrieved_texts = [texts[idx] for idx in indices[0] if idx < len(texts)]
    
    # Rank the retrieved texts using the reranker
    pairs = [[query, text] for text in retrieved_texts]
    scores = reranker.predict(pairs)
    # Get top k texts sorted by score
    sorted_texts = [text for _, text in sorted(zip(scores, retrieved_texts), reverse=True)[:k]]
    context = " ".join(sorted_texts)
    
    st.write("**Retrieved texts:**")
    for text in retrieved_texts:
        st.write(f"- {text}")
    st.write("**Ranked texts used as context:**")
    for text in sorted_texts:
        st.write(f"- {text}")
    st.write("**Combined context:**")
    st.write(context)
    
    # Get answer from the QA model
    result = qa_model(question=query, context=context)
    if result.get('score', 0) < confidence_threshold:
        st.write("The model is unsure about the answer. Returning context instead.")
        return context
    else:
        st.write(f"**Answer:** {result['answer']}")
        return result['answer']

def main():
    st.title("Crane Systems Chatbot")
    st.write("Ask your questions about crane systems and get answers based on the provided context.")

    # Load content and extract texts (cached)
    content_list = load_content(CONTENT_LIST_PATHS)
    texts, image_paths = extract_content(content_list, IMAGES_BASE_DIRS)
    st.write(f"Loaded **{len(content_list)}** items with **{len(texts)}** texts extracted.")

    # Load models and create FAISS index (cached)
    with st.spinner("Loading models and creating index..."):
        embedding_model, reranker, qa_model, index = load_models_and_index(texts)
    st.write(f"Models loaded and FAISS index created with **{index.ntotal}** items.")

    # User input for query
    query = st.text_input("Enter your question:")
    if st.button("Submit") and query:
        with st.spinner("Searching for an answer..."):
            answer = search_and_answer(query, texts, embedding_model, reranker, qa_model, index)
        st.markdown("### Final Answer")
        st.write(answer)

if __name__ == "__main__":
    main()