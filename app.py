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
    return jsonify({"text": "Hello from Koyeb - you reached the main page!"})

def google_search(query):
    """
    Performs a Google Search focusing on datasets, academic sources, and cited references.
    """
    search_query = f"{query} filetype:pdf OR site:researchgate.net OR site:ncbi.nlm.nih.gov OR site:data.gov OR site:arxiv.org OR site:worldbank.org OR site:europa.eu"
    
    url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    response = requests.get(url)

    if response.status_code == 200:
        results = response.json().get("items", [])
        search_summaries = []

        for item in results[:5]:  # Top 5 results
            title = item.get("title", "No title available")
            snippet = item.get("snippet", "No description available")
            link = item.get("link", "#")

            search_summaries.append(f"**{title}**\n{snippet}\n[📖 Read more]({link})\n")

        return "\n".join(search_summaries)

    return "No relevant datasets or cited sources found."

@app.route('/query', methods=['POST'])
def main():
    """
    Handles user queries, retrieves research-focused search results, and summarizes findings.
    """
    data = request.get_json()  
    user = data.get("user_name", "Unknown")
    message = data.get("text", "")

    print(data)

    # Ignore bot messages or empty input
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    print(f"Message from {user}: {message}")

    search_summary = google_search(message)

    # Generate a research-oriented response
    response = generate(
        model="4o-mini",
        system=(
            "You are a Research Assistant Chatbot that specializes in retrieving and summarizing academic research, "
            "datasets, and verified sources. Your task is to extract key insights from search results and ensure "
            "that all responses are well-cited, factual, and reference datasets or scientific studies when possible."
        ),
        query=search_summary,
        temperature=0.0,
        lastk=0,
        session_id="ResearchSession"
    )

    response_text = response.get("response", "I'm unable to generate a research summary at the moment.")

    # Format the final response
    final_response = f"**📚 Research Summary:**\n{response_text}\n\n**🔗 Sources & Datasets:**\n{search_summary}"

    print(final_response)

    return jsonify({"text": final_response})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()