import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_FOLDER = "data/pdfs"
DB_PATH = "faiss_index"

print("Starting PDF ingestion...")
documents = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        print(f"Loading {file}")
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

print("Creating embeddings... This takes 1-2 mins")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_documents(chunks, embeddings)
db.save_local(DB_PATH)
print("Done! Vector DB saved to faiss_index")