import os
import pandas as pd
from dotenv import load_dotenv
 
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
 
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings
 
load_dotenv()
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "YOUR_GROQ_API_KEY_HERE"
CHROMA_DB_DIR = "chroma_db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "data", "placement_dataset.xlsx")
 
 
class ChromaDefaultEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper around ChromaDB's DefaultEmbeddingFunction.
    Uses the all-MiniLM-L6-v2 ONNX model bundled with chromadb.
    No separate HuggingFace download needed.
    """
    def __init__(self):
        self._fn = DefaultEmbeddingFunction()
 
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)
 
    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]
 
 
def load_excel_as_documents(path: str) -> list[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Make sure placement_dataset.xlsx is inside the data/ folder."
        )
    print("Reading Excel file...")
    df = pd.read_excel(path)
    documents = []
    for _, row in df.iterrows():
        text = (
            f"Company: {row['Company Name']}. "
            f"Role: {row['Role']}. "
            f"CTC: {row['CTC (LPA)']} LPA. "
            f"Number of Openings: {row['No. of Openings']}. "
            f"Internship Available: {row['Internship']}. "
            f"Minimum CGPA Required: {row['Min CGPA']}. "
            f"Sector: {row['Sector']}."
        )
        metadata = {
            "company":    str(row["Company Name"]),
            "role":       str(row["Role"]),
            "ctc":        str(row["CTC (LPA)"]),
            "openings":   str(row["No. of Openings"]),
            "internship": str(row["Internship"]),
            "cgpa":       str(row["Min CGPA"]),
            "sector":     str(row["Sector"]),
        }
        documents.append(Document(page_content=text, metadata=metadata))
    print(f"Loaded {len(documents)} company records.")
    return documents
 
 
def build_vectorstore(documents: list[Document]) -> Chroma:
    print("Building ChromaDB...")
    embedding = ChromaDefaultEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_DB_DIR,
    )
    print("ChromaDB built and saved.")
    return vectorstore
 
 
def load_vectorstore() -> Chroma:
    print("Loading existing ChromaDB from disk...")
    embedding = ChromaDefaultEmbeddings()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding,
    )
 
 
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
 
 
def initialize_rag():
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Create a .env file with:  GROQ_API_KEY=your_key\n"
            "Or paste it directly in rag_engine.py where it says YOUR_GROQ_API_KEY_HERE"
        )
 
    if not os.path.exists(CHROMA_DB_DIR):
        print("First run — building vector database...")
        documents = load_excel_as_documents(EXCEL_PATH)
        vectorstore = build_vectorstore(documents)
    else:
        vectorstore = load_vectorstore()
 
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
 
    print("Connecting to Groq...")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )
 
    prompt = PromptTemplate.from_template("""
You are a helpful placement assistant. Use ONLY the context below to answer.
Context has: company name, role, CTC (LPA), openings, internship (Yes/No), min CGPA, sector.
 
Rules:
- Answer accurately from context only.
- For lists/comparisons, use bullet points.
- If not found, say: "I could not find that information in the placement dataset."
- Never make up data.
 
Context:
{context}
 
Question: {question}
 
Answer:""")
 
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
            source_docs=retriever,
        )
        | RunnableParallel(
            answer=prompt | llm | StrOutputParser(),
            source_docs=lambda x: x["source_docs"],
        )
    )
 
    print("RAG chain ready!")
    return rag_chain, retriever