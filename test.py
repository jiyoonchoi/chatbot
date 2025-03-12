import requests

response_main = requests.get("https://mini-proj-bot.onrender.com")
print('Web Application Response:\n', response_main.text, '\n\n')


data = {"text":"tell me about tufts"}
response_llmproxy = requests.post("https://mini-proj-bot.onrender.com/query", json=data)
print('LLMProxy Response:\n', response_llmproxy.text)