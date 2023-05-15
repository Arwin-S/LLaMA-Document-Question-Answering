from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = PyPDFLoader("docs/Nokia_Annual_Report_2021.pdf")
pages = loader.load_and_split()
pagesList = loader.load()


llama = LlamaCppEmbeddings(model_path="./models/ggml-vic13b-q5_1.bin")
query_result = llama.embed_documents(pages)
