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
    model_path="../models/ggml-vic13b-uncensored-q5_1.bin", callback_manager=callback_manager, verbose=True, n_ctx=2048
)

# llm = LlamaCpp(
#     model_path="../models/ggml-vic13b-uncensored-q5_1.bin", n_ctx=2048, max_tokens=50
# )

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Why is the sky blue?"


import sys
f = open("out.txt", 'w')
sys.stdout = f

llm_chain.run(question)

f.close()




# out = ""
# out = llm_chain.run(question)
# with open('out.txt', 'w') as file:
#     file.write(out)

# print(out)
