# import os
# from flask import Flask, request, jsonify
# from llmproxy import generate
# import requests

# app = Flask(__name__)

# API_KEY = os.environ.get("googleApiKey")
# CSE_ID = os.environ.get("googleSearchId")

# def google_search(query):
#     """
#     Performs a Google Custom Search and returns the top 3 results.
#     """
#     url = "https://www.googleapis.com/customsearch/v1"
#     params = {
#         'q': query,
#         'key': API_KEY,
#         'cx': CSE_ID,
#         'num': 3  # Limit results to avoid too much data
#     }
#     response = requests.get(url, params=params)
    
#     if response.status_code != 200:
#         print(f"Google Search API Error: {response.status_code}, {response.text}")
#         return []
    
#     results = response.json().get("items", [])
#     return [item.get("snippet", "") for item in results]  # Extract snippets

# @app.route('/query', methods=['POST'])
# def query():
#     data = request.get_json()
#     user = data.get("user_name", "Unknown")
#     message = data.get("text", "")

#     if data.get("bot") or not message:
#         return jsonify({"status": "ignored"})

#     try:
#         # Step 1: Fetch data from Google Search API
#         search_results = google_search(message)
#         search_info = "\n".join(search_results) if search_results else "No relevant search results found."

#         # Step 2: Generate response with Google Search results
#         query_with_context = f"User query: {message}\n\nRelevant information from Google Search:\n{search_info}"
        
#         response = generate(
#             model='4o-mini',
#             system=(
#                 "You are a Personal Finance Assistant bot. Your role is to provide financial guidance, "
#                 "budgeting strategies, investment options, and savings tips based on user queries. "
#                 "Use external information when available. If financial data is lacking, rely on general financial principles."
#             ),
#             query=query_with_context,
#             temperature=0.1,
#             lastk=0,
#             session_id='FinanceBotSession'
#         )

#         response_text = response.get('response', '')
#         if not response_text:
#             print("Warning: No response text generated.")
#             response_text = "I'm unable to generate a response at the moment."

#         return jsonify({"text": response_text})

#     except Exception as e:
#         print(f"Error generating response: {e}")
#         return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

# @app.errorhandler(404)
# def page_not_found(e):
#     return "Not Found", 404

# if __name__ == "__main__":
#     app.run()


import os, requests
from flask import Flask, request, jsonify
from llmproxy import generate

app = Flask(__name__)

GOOGLE_API_KEY = os.environ.get("googleApiKey")
SEARCH_ENGINE_ID = os.environ.get("googleSearchId")

@app.route('/')
def hello_world():
    return jsonify({"text":'Hello from Koyeb - you reached the main page!'})

def google_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={CX}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("items", [])
        summaries = [item["snippet"] for item in results[:5]]
        return " ".join(summaries)
    return "No relevant results found."

@app.route('/query', methods=['POST'])
def main():
    data = request.get_json() 

    # Extract relevant information
    user = data.get("user_name", "Unknown")
    message = data.get("text", "")

    print(data)

    # Ignore bot messages
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    print(f"Message from {user} : {message}")

    search_summary = google_search(message)

    # Generate a response using LLMProxy
    response = generate(
        model='4o-mini',
        # system='answer my question and add keywords',
        system='Summarize the following information and answer the query:',
        # query= message,
        query=search_summary,
        temperature=0.0,
        lastk=0,
        session_id='GenericSession'
    )

    response_text = response['response']

    # Send response back
    print(response_text)

    return jsonify({"text": response_text})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()