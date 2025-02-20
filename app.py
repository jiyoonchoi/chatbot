import os
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
import time

app = Flask(__name__)

# Set the file path for the PDF directly
pdf_file_path = 'Personal Finance for Dummies.pdf'
processed = False
pdf_processing_complete = False

# Process the PDF when the server starts
@app.before_first_request
def process_pdf():
    global processed, pdf_processing_complete

    if os.path.exists(pdf_file_path):
        try:
            response = pdf_upload(
                path=pdf_file_path,
                session_id='GenericSession',
                strategy='smart'
            )
            processed = True
            print("PDF processing started:", response)

            # Wait for the PDF to be added to the context (this may require adjusting based on the actual system's behavior)
            while not pdf_processing_complete:
                # Simulate checking if the document is fully processed. Adjust based on actual processing behavior.
                time.sleep(5)  # wait for the processing to complete
                print("Waiting for PDF to be fully processed...")

            print("PDF processed successfully.")
        except Exception as e:
            print(f"Error processing PDF: {e}")
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
    
    if not pdf_processing_complete:
        return jsonify({"error": "The PDF is still being processed. Please try again later."}), 503

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
    try:
        response = generate(
            model='4o-mini',
            system=(
                "You are a Personal Finance Assistant bot. Your role is to help individuals "
                "with financial matters such as tracking expenses, budgeting, setting savings goals, "
                "suggesting investment options, and answering questions related to taxes, loans, and more. "
                "Please do not greet users automatically. Answer based on the information from the uploaded PDF or general financial principles."
            ),
            query=message,
            temperature=0.1,
            lastk=0,
            session_id='GenericSession',
            rag_usage=True,         
            rag_threshold='0.3',
            rag_k=3
        )

        response_text = response.get('response', '')
        if not response_text:
            print("Warning: No response text generated.")
            response_text = "Sorry, I couldn't generate a helpful response at the moment."

        print(f"Response from bot: {response_text}")
        return jsonify({"text": response_text})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()
