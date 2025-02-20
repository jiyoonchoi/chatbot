import os
from flask import Flask, request, jsonify
from llmproxy import generate
import requests

app = Flask(__name__)

API_KEY = os.environ.get("googleApiKey")
CSE_ID = os.environ.get("googleSearchId")

def google_search(query):
    # Performs a Google Custom Search focused on datasets and research papers.
    search_query = (
        f"{query} filetype:pdf OR site:researchgate.net OR site:ncbi.nlm.nih.gov OR site:data.gov "
        "OR site:arxiv.org OR site:worldbank.org OR site:europa.eu OR site:sciencedirect.com"
    )
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': search_query,
        'key': API_KEY,
        'cx': CSE_ID,
        'num': 3  # Limit to top 3 relevant results
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Google Search API Error: {response.status_code}, {response.text}")
        return []

    results = response.json().get("items", [])
    
    search_summaries = []
    for item in results:
        title = item.get("title", "No title available")
        snippet = item.get("snippet", "No description available")
        link = item.get("link", "#")
        search_summaries.append(f"**{title}**\n{snippet}\n[🔗 View Paper]({link})\n")

    return search_summaries

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user = data.get("user_name", "Unknown")
    message = data.get("text", "")

    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    try:
        # Step 1: Fetch research papers & datasets
        search_results = google_search(message)
        search_info = "\n".join(search_results) if search_results else "No relevant research papers found."

        # Step 2: Generate summary response
        query_with_context = f"User query: {message}\n\nResearch Findings:\n{search_info}"
        
        response = generate(
            model='4o-mini',
            system=(
                "You are a Research Assistant AI that specializes in retrieving and summarizing "
                "academic research, datasets, and scientific studies. Your goal is to provide well-cited, "
                "fact-based insights from reputable sources, ensuring that responses reference credible datasets "
                "or peer-reviewed papers whenever possible."
            ),
            query=query_with_context,
            temperature=0.0,
            lastk=0,
            session_id='ResearchAssistantSession'
        )

        response_text = response.get('response', "I'm unable to generate a research summary at the moment.")

        # Format the final response
        final_response = f"**📚 Research Summary:**\n{response_text}\n\n**🔗 Relevant Research Papers & Datasets:**\n{search_info}"

        return jsonify({"text": final_response})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()



# import os
# import requests
# from flask import Flask, request, jsonify
# from llmproxy import generate

# app = Flask(__name__)

# GOOGLE_API_KEY = os.environ.get("googleApiKey")
# SEARCH_ENGINE_ID = os.environ.get("googleSearchId")

# # Predefined keyword filters for different topics
# KEYWORD_FILTERS = {
#     "climate change": ["CO2 emissions", "global warming", "temperature rise", "scientific consensus"],
#     "finance": ["GDP", "inflation rate", "market trends", "economic indicators"],
#     "medicine": ["clinical trials", "medical research", "drug efficacy", "disease studies"],
#     "technology": ["AI advancements", "machine learning models", "cybersecurity", "quantum computing"],
# }

# def extract_relevant_keywords(query):
#     """
#     Matches the user query against predefined categories and extracts relevant keywords.
#     """
#     query_lower = query.lower()
#     matched_keywords = []

#     for category, keywords in KEYWORD_FILTERS.items():
#         if category in query_lower:
#             matched_keywords.extend(keywords)

#     return " OR ".join(set(matched_keywords)) if matched_keywords else ""


# def google_search(query):
#     """
#     Performs a Google Search focusing on datasets, academic sources, and cited references,
#     while filtering for relevant keywords.
#     """
#     keyword_filter = extract_relevant_keywords(query)
#     search_query = f"{query} {keyword_filter} filetype:pdf OR site:researchgate.net OR site:ncbi.nlm.nih.gov OR site:data.gov OR site:arxiv.org OR site:worldbank.org OR site:europa.eu"

#     url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
#     response = requests.get(url)

#     if response.status_code == 200:
#         results = response.json().get("items", [])
#         if not results:
#             return "No relevant datasets or cited sources found."

#         search_summaries = []
#         for item in results[:5]:  # Limit to Top 5 results
#             title = item.get("title", "No title available")
#             snippet = item.get("snippet", "No description available")
#             link = item.get("link", "#")

#             search_summaries.append(f"📌 **{title}**\n_{snippet}_\n[📖 Read More]({link})\n")

#         return "\n".join(search_summaries)

#     return "No relevant datasets or cited sources found."


# @app.route('/query', methods=['POST'])
# def main():
#     """
#     Handles user queries, retrieves research-focused search results, and summarizes findings.
#     """
#     data = request.get_json()
#     user = data.get("user_name", "Unknown")
#     message = data.get("text", "").strip()  # Strip to avoid empty spaces

#     print(data)

#     # Ignore bot messages or empty input, but provide a helpful response
#     if data.get("bot") or not message:
#         return jsonify({"text": "It looks like you didn't enter a topic. Try asking about a specific research area!"})

#     print(f"Message from {user}: {message}")

#     search_summary = google_search(message)

#     # Ensure there are valid results before summarizing
#     if "No relevant datasets" in search_summary:
#         return jsonify({"text": search_summary})

#     # Generate a research-oriented response
#     response = generate(
#         model="4o-mini",
#         system=(
#             "You are a Research Assistant Chatbot that specializes in retrieving and summarizing academic research, "
#             "datasets, and verified sources. Your task is to extract key insights from search results and ensure "
#             "that all responses are well-cited, factual, and reference datasets or scientific studies when possible."
#         ),
#         query=search_summary,
#         temperature=0.0,
#         lastk=0,
#         session_id="ResearchSession"
#     )

#     response_text = response.get("response", "I'm unable to generate a research summary at the moment.")

#     # Format the final response
#     final_response = f"📚 **Research Summary:**\n{response_text}\n\n🔗 **Sources & Datasets:**\n{search_summary}"

#     print(final_response)

#     return jsonify({"text": final_response})


# @app.errorhandler(404)
# def page_not_found(e):
#     return "Not Found", 404


# if __name__ == "__main__":
#     app.run()
