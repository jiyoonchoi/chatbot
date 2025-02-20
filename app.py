import os
import threading
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
import time

app = Flask(__name__)

# Set the file path for the PDF directly
pdf_file_path = 'Personal Finance for Dummies.pdf'
processed = False
pdf_processing_complete = False

# Function to process the PDF in a background thread
def process_pdf_async():
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

            # Simulate checking if the document is fully processed (adjust based on actual behavior)
            time.sleep(10)  # Adjust sleep time to reflect actual processing time
            pdf_processing_complete = True

            print("PDF processed successfully.")
        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print("PDF file not found at:", pdf_file_path)

# Process the PDF when the server starts in a background thread
@app.before_first_request
def before_first_request():
    # Start PDF processing asynchronously
    threading.Thread(target=process_pdf_async, daemon=True).start()

@app.route('/')
def hello_world():
    return jsonify({"text": 'Hello from Koyeb - you reached the main page!'})

@app.route('/query', methods=['POST'])
def query():
    global processed, pdf_processing_complete

    # Ensure the PDF is processed before accepting queries
    if not processed:
        return jsonify({"error": "The system is still processing the PDF. Please try again later."}), 503

    # Check if PDF processing is fully complete
    if not pdf_processing_complete:
        return jsonify({"error": "The PDF is still being processed. Please try again later."}), 503

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

        return jsonify({"text": response_text})

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500


@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()
