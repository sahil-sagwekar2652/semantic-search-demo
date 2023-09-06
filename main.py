from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from get_embds import LocalLlamaEmbeddings
import sys
import os
import time


# Load environment variables
load_dotenv()

loader = DirectoryLoader(
    "./consume",
    glob="*.pdf",
    use_multithreading=True,
    loader_cls=PyPDFLoader,
    show_progress=True,
)

docs = loader.load_and_split(
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
)

print("-" * 20 + "DOCUMENT LOAD AND SPLIT COMPLETE" + "-" * 20)

embeddings = LocalLlamaEmbeddings(
    os.environ.get("EMBEDDINGS_URL"),
    {"accept": "application/json", "Content-Type": "application/json"},
)

# save to disk
start = time.time()
db = Chroma.from_documents([docs[0]], embeddings, persist_directory="./chroma_db")
print(f"\nEmbeddings created in {time.time()-start}\n")

print("-" * 20 + "EMBEDDINGS CREATED" + "-" * 20)

# load from disk
# db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

query = str(sys.argv[1])

# query_embedding = embeddings.embed_query(query)
# fdocs = db.similarity_search_by_vector(query_embedding)

fdocs = db.search(query=query, search_type='similarity')

for i in fdocs:
    print(i.metadata)
    print("\n\n")
    print(i.page_content)
    print("\n\n")

print("-" * 20 + "SEARCH COMPLETE" + "-" * 20)
