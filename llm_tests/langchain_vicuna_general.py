from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="models/ggml-vic13b-q5_1.bin", callback_manager=callback_manager, verbose=True
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What does the acronym 7705 SAR mean?"

llm_chain.run(question)