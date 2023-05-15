from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# loader = PyPDFLoader('docs/speech.pdf')
# pages = loader.load_and_split(
#     text_splitter=RecursiveCharacterTextSplitter(
#         chunk_size = 2000,
#         chunk_overlap = 0,
#         length_function = len,
#     )
# )

loader = TextLoader("docs/trends.txt")
pages = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 50,
        length_function = len,
    )
)

# print(pages[0].page_content)
# print("\n")
# print(pages[1].page_content)


embeddings_llama = LlamaCppEmbeddings(model_path="models/ggml-vic13b-q5_1.bin", n_ctx=2048)
chroma_index = Chroma.from_documents(pages, embeddings_llama, persist_directory="chroma_index")

chroma_index.persist()

