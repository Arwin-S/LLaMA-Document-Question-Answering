import json
from llama_cpp import Llama

#load model
print("started loading model")

llm= Llama(model_path="./models/ggml-vic13b-q5_1.bin")

print("model done loading")

output = llm(
    "Question: What is the transistor? Answer:",
    max_tokens= 100,
    stop=["\n", "Question:", "Q:"],
    echo=True,
)

#print output
print(json.dumps(output, indent=2))