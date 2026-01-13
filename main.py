import os
import ssl
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- THE SIMPLEST FIX FOR NLTK/SSL ERRORS ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Configuration
DOCS_FOLDER = "../docs"
CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5433/rag_db"
#postgresql+psycopg://user:password@localhost:5432/rag_db
COLLECTION_NAME = "my_docs_collection"

def main():
    # 1. Initialize Embedding Model
    embeddings = OllamaEmbeddings(model="bge-m3")

    # 2. Load all .docx files
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    if not os.path.exists(DOCS_FOLDER):
        print(f"❌ Folder not found at {DOCS_FOLDER}")
        return

    for file in os.listdir(DOCS_FOLDER):
        # Ignore hidden files and temporary Word files
        if file.endswith(".docx") and not file.startswith("~$"):
            print(f"Processing {file}...")
            try:
                loader = UnstructuredWordDocumentLoader(os.path.join(DOCS_FOLDER, file))
                data = loader.load()
                all_chunks.extend(text_splitter.split_documents(data))
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")

    if not all_chunks:
        print("No valid .docx files found!")
        return

    # 3. Create Vector Store & Index
    print(f"Creating embeddings for {len(all_chunks)} chunks...")
    PGVector.from_documents(
        embedding=embeddings,
        documents=all_chunks,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    print("✅ Success! Your local database is now vectorized.")

if __name__ == "__main__":
    main()