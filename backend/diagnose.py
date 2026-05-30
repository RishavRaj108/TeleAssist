import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

print("Loading embeddings...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("  Embeddings loaded.")

print("Loading CSV...")
df = pd.read_csv("data/faq.csv")
print(f"  {len(df)} rows found")

docs = []
for _, row in df.iterrows():
    docs.append(Document(
        page_content=f"Q: {row['question']}\nA: {row['answer']}",
        metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])}
    ))

print("Deleting old chroma_store...")
if os.path.exists("chroma_store"):
    shutil.rmtree("chroma_store")
    print("  Deleted.")

print("Writing to Chroma...")
vs = Chroma.from_documents(docs, emb, collection_name="faq", persist_directory="chroma_store")
count = vs._collection.count()
print(f"  Done. Count: {count}")

if count == 0:
    print("PROBLEM: Still 0 vectors — chromadb version may be incompatible.")
    print("Run: pip install langchain-chroma==0.1.4 chromadb==0.5.3")
else:
    print("SUCCESS: Vectors written correctly.")
