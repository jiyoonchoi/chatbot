import requests, os
from flask import Flask, request, jsonify
from llmproxy import generate

app = Flask(__name__)

# Function to perform a search using Google's Custom Search API
def google_search(query):
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('CSE_ID')
    websites = ['https://wagnerhigh.net/ourpages/auto/2013/2/13/48465391/Personal%20Finance%20for%20Dummies.pdf', 'https://www.nerdwallet.com/', 'https://www.reddit.com/r/personalfinance/']
    site_query = " OR ".join([f"site:{website}" for website in websites])
    search_query = f"{site_query} {query}"
    
    url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={api_key}&cx={cse_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()

        if 'items' in results:
            citations = []
            # Iterate through the results and gather snippets with sources
            for item in results['items']:
                snippet = item['snippet']
                link = item['link']
                citations.append(f"{snippet} (Source: {link})")
            return "\n\n".join(citations)  # Join all citations into a single response
        else:
            return "Sorry, I couldn't find relevant information."
    
    except requests.exceptions.RequestException as e:
        print(f"Error with Google Custom Search API: {e}")
        return "Sorry, there was an error with the search. Please try again later."

@app.route('/')
def hello_world():
    return jsonify({"text": 'Hello from Koyeb - you reached the main page!'})

@app.route('/query', methods=['POST'])
def query():
    # Extract data from the query
    data = request.get_json()
    user = data.get("user_name", "Unknown")
    message = data.get("text", "")

    # Ignore bot messages or empty text
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    try:
        # Perform Google search to gather financial advice related to the query
        search_response = google_search(message)

        # Generate the response from the model, incorporating the search response
        response = generate(
            model='4o-mini',
            system=(
                "You are a Personal Finance Assistant bot. Your role is to help individuals "
                "with financial matters such as tracking expenses, budgeting, setting savings goals, "
                "suggesting investment options, and answering questions related to taxes, loans, and more. "
                "Please do not greet users automatically. Answer based on the provided information or general financial principles."
            ),
            query=message + " " + search_response,  # Append the search response to the query
            temperature=0.1,
            lastk=0,
            session_id='GenericSession'
        )

        response_text = response.get('response', '')
        if not response_text:
            print("Warning: No response text generated.")
            response_text = f"Here's some information I found: {search_response}"

        return jsonify({"text": response_text})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()
