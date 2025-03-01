import requests

response_main = requests.get("https://balanced-bess-jiyoon-648a0821.koyeb.app/")
print('Web Application Response:\n', response_main.text, '\n\n')


data = {"text":"tell me about tufts", "user_name":"Aya", "bot":False}
response_llmproxy = requests.post("https://balanced-bess-jiyoon-648a0821.koyeb.app/query", json=data)
print('LLMProxy Response:\n', response_llmproxy.text)