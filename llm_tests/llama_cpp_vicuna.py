import json
from llama_cpp import Llama

#load model
print("started loading model")

llm= Llama(model_path="../models/ggml-vic13b-uncensored-q5_1.bin", n_ctx=2048)

print("model done loading")

output = llm(
    "Question: What is Generative AI and how far until we reach it?. Answer:",
    max_tokens= 100,
    echo=True,
)

#print output
print(json.dumps(output, indent=2))