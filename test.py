import requests

<<<<<<< Updated upstream
response_main = requests.post("https://replace_with_your_web_server_link")
=======
response_main = requests.get("https://balanced-bess-jiyoon-648a0821.koyeb.app/")
>>>>>>> Stashed changes
print('Web Application Response:\n', response_main.text, '\n\n')


data = {"text":"tell me about tufts"}
response_llmproxy = requests.post("https://balanced-bess-jiyoon-648a0821.koyeb.app/query", json=data)
print('LLMProxy Response:\n', response_llmproxy.text)
