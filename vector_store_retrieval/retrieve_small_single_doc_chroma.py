from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

query = "What is the 1st trend?"
embeddings_llama = LlamaCppEmbeddings(model_path="models/ggml-vic13b-q5_1.bin")

chroma_index = Chroma(persist_directory="chroma_index", embedding_function=embeddings_llama)

docs = chroma_index.similarity_search(query)

print(docs[0].page_content)

    

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

chain = load_qa_chain(llm = LlamaCpp(model_path="./models/ggml-vic13b-q5_1.bin",
                                    callback_manager=callback_manager, verbose=True, n_ctx=2048),
                                 chain_type="stuff")

chain.run(input_documents=docs, question=query)