import requests
url = 'http://192.18.150.40:8000/nodes/add' # proxy
# url = 'http://127.0.0.1:8001/nodes/add' # proxy1
# url = 'http://127.0.0.1:8001/nodes/add' # proxy2
# url = 'http://127.0.0.1:8002/nodes/add' # proxy3


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
node_list = [
    # "http://127.0.0.1:10401", # gemma
    # "http://127.0.0.1:10011", # gemma
    # "http://127.0.0.1:10411",
    "http://192.18.150.40:10700",
    # "http://127.0.0.1:10402", # ministral
    # "http://127.0.0.1:10073", # ministral
    # "http://127.0.0.1:10412",
    "http://192.18.150.40:10400",
    # "http://127.0.0.1:10404", # qwen-coder
    # "http://127.0.0.1:10088", # qwen-coder
    "http://192.18.150.40:10788",
    # "http://127.0.0.1:10414",
    # "http://127.0.0.1:10424",
    # "http://127.0.0.1:10405", # qwen-math
    # "http://127.0.0.1:10022", # qwen-math
    # "http://127.0.0.1:10415",
    "http://192.18.150.40:10300",
    # "http://127.0.0.1:10406", # lma-biomed
    # "http://127.0.0.1:10033", # lma-biomed
    # "http://127.0.0.1:10416",
    "http://192.18.150.40:10200",
]

for node in node_list:
    data = {"url": node}
    response = requests.post(url, headers=headers, json=data)
    print(f"url: {node}, msg: {response.text}")

# data = {"url": "http://127.0.0.1:10800"}
# response = requests.post(url, headers=headers, json=data)
# print(response.text)

# # delete a node
# import requests
# url = 'http://127.0.0.1:8000/nodes/remove'
# headers = {'accept': 'application/json',}
# params = {'node_url': "http://127.0.0.1:10406",}
# response = requests.post(url, headers=headers, data='', params=params)
# print(response.text)