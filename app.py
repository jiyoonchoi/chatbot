import os
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload

app = Flask(__name__)

# Set the file path for the PDF you want to provide directly
pdf_file_path = 'IfYouCan.pdf'
processed = False  # Flag to track whether the PDF has been processed

# Process the PDF when the server starts
@app.before_first_request
def process_pdf():
    global processed

    if os.path.exists(pdf_file_path):
        response = pdf_upload(
            path=pdf_file_path,
            session_id='GenericSession',
            strategy='smart'
        )
        processed = True 
        print("PDF processed successfully:", response)
    else:
        print("PDF file not found at:", pdf_file_path)

@app.route('/')
def hello_world():
    return jsonify({"text": 'Hello from Koyeb - you reached the main page!'})

@app.route('/query', methods=['POST'])
def query():
    # Ensure the PDF is processed before accepting queries
    if not processed:
        return jsonify({"error": "The system is still processing the PDF. Please try again later."}), 503

    data = request.get_json()

    # Extract relevant information from the request
    user = data.get("user_name", "Unknown")
    message = data.get("text", "")

    print(data)

    # Ignore bot messages or if no message is provided
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    print(f"Message from {user}: {message}")

    # Instruct the bot to respond as a Personal Finance Assistant and enable RAG
    response = generate(
        model='4o-mini',
        system=(
            "You are a Personal Finance Assistant bot. Your role is to help individuals "
            "with financial matters such as tracking expenses, budgeting, setting savings goals, "
            "suggesting investment options, and answering questions related to taxes, loans, and more. "
            "You use financial literacy books as your reference. "
            "Please provide answers based on the information from the uploaded PDF or general financial principles."
        ),
        query=message,
        temperature=0.0,
        lastk=0,
        session_id='GenericSession',
        rag_usage=True,         # Enable RAG
        rag_threshold='0.3',    # Set threshold for relevance of retrieved data
        rag_k=1                 # Retrieve top k most relevant passages
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
