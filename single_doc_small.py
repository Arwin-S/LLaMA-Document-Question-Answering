from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.document_loaders import TextLoader

from langchain.chains import RetrievalQA

loader = TextLoader("docs/simple.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager


llama = LlamaCppEmbeddings(model_path="models/ggml-vic13b-q5_1.bin")
docsearch = Chroma.from_documents(texts, llama)

qa = RetrievalQA.from_chain_type(llm = LlamaCpp(model_path="./models/ggml-vic13b-q5_1.bin",
                                    callback_manager=callback_manager, verbose=True),
                                 chain_type="stuff", retriever=docsearch.as_retriever())

query = "What type of restaurants would you recommend?"

qa.run(query)

