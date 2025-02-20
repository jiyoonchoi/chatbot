from flask import Flask, request, jsonify
from llmproxy import generate

app = Flask(__name__)

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

    # Generate the response from the model
    try:
        response = generate(
            model='4o-mini',
            system=(
                "You are a Personal Finance Assistant bot. Your role is to help individuals "
                "with financial matters such as tracking expenses, budgeting, setting savings goals, "
                "suggesting investment options, and answering questions related to taxes, loans, and more. "
                "Please do not greet users automatically. Answer based on general financial principles."
            ),
            query=message,
            temperature=0.1,
            lastk=0,
            session_id='GenericSession',
            rag_usage=True,
            rag_threshold='0.3',
            rag_k=1
        )

        response_text = response.get('response', '')
        if not response_text:
            print("Warning: No response text generated.")
            response_text = "Sorry, I couldn't generate a helpful response at the moment."

        return jsonify({"text": response_text})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()
