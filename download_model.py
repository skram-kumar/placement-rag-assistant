"""
Run this once before launching the app to pre-download the embedding model.
Usage:  python download_model.py
"""
 
import os
 
CACHE_FOLDER = "./model_cache"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
 
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_FOLDER
os.makedirs(CACHE_FOLDER, exist_ok=True)
 
print(f"Downloading / verifying model: {MODEL_NAME}")
print(f"Cache folder: {os.path.abspath(CACHE_FOLDER)}")
 
# Use HuggingFaceEmbeddings (same class used in rag_engine) so the cache path
# is identical — avoids a second download when the app starts.
from langchain_huggingface import HuggingFaceEmbeddings
 
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder=CACHE_FOLDER,
)
 
# Quick smoke-test
test_embedding = embedding_model.embed_query("hello world")
print(f"\n✅ Model loaded and working. Embedding dimension: {len(test_embedding)}")
print("You can now run:  streamlit run app.py")