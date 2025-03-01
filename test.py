import requests

response_main = requests.get("https://mini-proj-bot.onrender.com")
print('Web Application Response:\n', response_main.text, '\n\n')


data = {"text":"what is the motivation of the paper?", "user_name":"Aya", "bot":False}
response_llmproxy = requests.post("https://mini-proj-bot.onrender.com/query", json=data)
print('LLMProxy Response:\n', response_llmproxy.text)