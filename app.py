import requests
from flask import Flask, request, jsonify
from llmproxy import generate

app = Flask(__name__)

# Function to perform a search using Google's Custom Search API
def google_search(query):
    api_key = 'AIzaSyDKNUeIRdGOIacjk--fNa2vcs00WHtqHIM'  # Your Google API key
    cse_id = '945654d55c45d4da4'  # Your Custom Search Engine ID
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP status codes
        results = response.json()

        if 'items' in results:
            # Extract the first result's snippet and link as relevant information
            first_result = results['items'][0]
            return first_result['snippet'], first_result['link']
        else:
            return "Sorry, I couldn't find relevant information.", ""
    
    except requests.exceptions.RequestException as e:
        print(f"Error with Google Custom Search API: {e}")
        return "Sorry, there was an error with the search. Please try again later.", ""

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
        search_snippet, search_link = google_search(message)

        # Generate the response from the model, incorporating the search snippet
        response = generate(
            model='4o-mini',
            system=(
                "You are a Personal Finance Assistant bot. Your role is to help individuals "
                "with financial matters such as tracking expenses, budgeting, setting savings goals, "
                "suggesting investment options, and answering questions related to taxes, loans, and more. "
                "Please do not greet users automatically. Answer based on the provided information or general financial principles."
            ),
            query=message + " " + search_snippet,  # Append the search snippet to the query
            temperature=0.1,
            lastk=0,
            session_id='GenericSession'
        )

        response_text = response.get('response', '')
        if not response_text:
            print("Warning: No response text generated.")
            response_text = f"Here's some information I found: {search_snippet} More details here: {search_link}"

        return jsonify({"text": response_text})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()
