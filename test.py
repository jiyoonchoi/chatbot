import requests

response_main = requests.get("https://balanced-bess-jiyoon-648a0821.koyeb.app")
print('Web Application Response:\n', response_main.text, '\n\n')


data = {"text":"Hi there! How can I help you today?"}
response_llmproxy = requests.post("hhttps://balanced-bess-jiyoon-648a0821.koyeb.app/query", json=data)
print('LLMProxy Response:\n', response_llmproxy.text)