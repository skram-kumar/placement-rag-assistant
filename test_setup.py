"""
Run this first to verify everything works before launching the app.
Usage:  python test_setup.py
"""
 
print("Step 1: Testing ChromaDB default embedding...")
try:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    fn = DefaultEmbeddingFunction()
    result = fn(["hello world"])
    print(f"  ✅ Embedding works! Dimension: {len(result[0])}")
except Exception as e:
    print(f"  ❌ Embedding failed: {e}")
    print("  Fix: pip install chromadb --upgrade")
    exit(1)
 
print("\nStep 2: Testing Groq API key...")
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if key:
        print(f"  ✅ GROQ_API_KEY found in .env ({key[:8]}...)")
    else:
        print("  ⚠️  GROQ_API_KEY not found in .env")
        print("  Create a .env file with:  GROQ_API_KEY=your_key_here")
except Exception as e:
    print(f"  ❌ Error: {e}")
 
print("\nStep 3: Testing Excel file...")
try:
    import pandas as pd
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "placement_dataset.xlsx")
    df = pd.read_excel(path)
    print(f"  ✅ Excel loaded: {len(df)} rows, columns: {list(df.columns)}")
except Exception as e:
    print(f"  ❌ Excel failed: {e}")
    print("  Fix: make sure data/placement_dataset.xlsx exists")
    exit(1)
 
print("\nStep 4: Testing LangChain imports...")
try:
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    print("  ✅ All LangChain imports OK")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    print("  Fix: pip install -r requirements.txt")
    exit(1)
 
print("\n" + "="*50)
print("✅ All checks passed! Run:  streamlit run app.py")
print("="*50)