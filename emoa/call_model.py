from openai import OpenAI
import time
import os
client = OpenAI(
    api_key=os.getenv('YOUR_API_KEY'),
    # base_url="http://127.0.0.1:10420/v1"  
    # base_url="http://127.0.0.1:8000/v1"  
    base_url="http://127.0.0.1:10406/v1"  
    # base_url="http://127.0.0.1:10416/v1"  
    # base_url="http://127.0.0.1:10402/v1"  #mistral
    # api_key=os.getenv('OPENAI_API_KEY'),
    # base_url="https://api.openai.com/v1"
    # api_key='YOUR_API_KEY',
    # base_url="https://openrouter.ai/api/v1"
)

model_name = "ContactDoctor/Bio-Medical-Llama-3-8B"
# model_name = "mistralai/Ministral-8B-Instruct-2410"
# model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
# model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
# model_name = "google/gemma-2-9b-it"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "deepseek/deepseek-r1-distill-llama-70b:free"
# model_name = "nvidia/llama-3.1-nemotron-70b-instruct:free"
start = time.time()
# question = "Which of the following is not a way to form recombinant DNA?\n\nA) Translation\nB) Conjugation\nC) Specialized transduction\nD) Transformation"
# question = "Donald drinks 3 more than twice the number of juice bottles Paul drinks in one day. If Paul drinks 3 bottles of juice per day, how many bottles does Donald drink per day?"
# question = "Write a function to get the sum of the digits of a non-negative integer.\nYour code should pass these tests:\n\nassert sum_digits(345)==12\nassert sum_digits(12)==3\nassert sum_digits(97)==16"
response = client.chat.completions.create(
  model=model_name,
  messages=[
    # {"role": "system", "content": "You are a helpful assistant."},
    #{"role": "user", "content": "Translate "Beauty is in the eye of the beholder" into Chinese"},
    {"role": "user", "content": "In an antique car show, there are seven vehicles: a limousine, a convertible, a station wagon, a minivan, a bus, a tractor, and a truck. The minivan is the oldest. The truck is older than the station wagon. The truck is newer than the limousine. The bus is newer than the convertible. The bus is older than the tractor. The limousine is newer than the tractor. Please sort the vehicles from old to new."}
    # {"role": "system", "content": "You are a creative assistant skilled in rephrasing text while preserving its meaning."},
    # {"role": "user", "content": f"Rephrase the following question in a different way, using varied sentence structures and synonyms, question format (e.g. multiple choice, quiz, fill-in-the-blank, etc), different order of answers, but keep the original meaning intact. Please only return your rephrased question, without additional words:\n\n{question}"},
    # {"role": "user", "content": f"I am training a router that can predict the scores of various large models in answering a user's question. However, I am currently facing a problem of data imbalance, as most of the data are questions that all models can answer correctly, which hinders the router's training. To address this, I want to perform data augmentation on difficult samples. The specific method is to rephrase the difficult questions without changing their knowledge scope or complexity, but altering their wording or format. If the question is a multiple-choice question, the options will also be rephrased (if there are fixed technical terms in the options, they do not need to be rephrased), and their order may be changed. Additionally, the question format or type may also be altered during the rephrasing process, such as converting it into a multiple-choice question, a fill-in-the-blank question, or an open-ended question, among others. Please output only the rephrased question without any redundant content. Original question: {question}. Rephrased question:"}
  ],
    temperature=1,
    top_p=0.8,
    timeout=100
)

# response = client.chat.completions.create(
#   model=model_name,
#   # reasoning_effort="medium",
#   messages=[
#     # {"role": "system", "content": "You are a helpful assistant."},
#     # {"role": "user", "content": "Translate "Beauty is in the eye of the beholder" into Chinese"},
#     {"role": "user", "content": """Write a bash script that takes a matrix represented as a string with 
# format '[1,2],[3,4],[5,6]' and prints the transpose in the same format."""}
#   ]
# )
end = time.time()

# print(response)
print(response.choices[0].message.content)
print("total time: {}".format(end-start))
