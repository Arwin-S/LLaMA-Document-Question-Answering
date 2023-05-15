from langchain.embeddings import LlamaCppEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader('docs/acronyms_small.pdf')
pages = loader.load_and_split(
    # text_splitter=RecursiveCharacterTextSplitter(
    #     chunk_size = 500,
    #     chunk_overlap = 50,
    #     length_function = len,
    # )
)

embeddings_llama = LlamaCppEmbeddings(model_path="models/ggml-vic13b-q5_1.bin", n_ctx=2048)
faiss_index = FAISS.from_documents(pages, embeddings_llama)
faiss_index.save_local('faiss_index')

