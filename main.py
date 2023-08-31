from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st
from get_embds import LocalLlamaEmbeddings
import sys

# from langchain.storage import RedisStore


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

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

embeddings = LocalLlamaEmbeddings(
    "http://100.105.38.57:3001/v1/embeddings",
    {"accept": "application/json", "Content-Type": "application/json"},
)

# save to disk
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

print("-" * 20 + "EMBEDDINGS CREATED" + "-" * 20)

# load from disk
# db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

query = input("Enter your query: ")

query_embedding = embeddings.embed_query(query)

fdocs = db.similarity_search_by_vector(query_embedding)


for i in fdocs:
    print(i.metadata)
    print("\n\n")
    print(i.page_content)
    print("\n\n")
