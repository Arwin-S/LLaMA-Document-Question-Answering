from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

query = "What does CCM mean?"
embeddings_llama = LlamaCppEmbeddings(model_path="models/ggml-vic13b-q5_1.bin", n_ctx=2048)


# 1: Exracting string from db

saved_db = FAISS.load_local('faiss_index', embeddings_llama)
docs = saved_db.similarity_search(query)

# print(docs[0].page_content)

text_file = open("output_docs/faiss_out.txt", "w")
n = text_file.write(docs[0].page_content)
text_file.close()



# 2: Prompting String

loader = TextLoader("output_docs/faiss_out.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager


docsearch = Chroma.from_documents(texts, embeddings_llama)

qa = RetrievalQA.from_chain_type(llm = LlamaCpp(model_path="./models/ggml-vic13b-q5_1.bin",
                                    callback_manager=callback_manager, verbose=True),
                                 chain_type="stuff", retriever=docsearch.as_retriever())


qa.run(query)