import os
import uuid
import time
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the directory of the current script and build an absolute path to the PDF.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "research_paper.pdf")

app = Flask(__name__)

# Global conversation history.
conversation_history = {}

def get_session_id(data):
    """
    Returns a consistent session ID based on the user_name (as in the TA code).
    """
    user = data.get("user_name", "unknown_user").strip().lower()
    return f"session_{user}"

def upload_pdf_if_needed(pdf_path, session_id):
    """
    Uploads the PDF using pdf_upload with the provided session_id.
    """
    response = pdf_upload(
        path=pdf_path,
        session_id=session_id,
        strategy="smart"
    )
    print(f"DEBUG: Upload response: {response}")
    return "Successfully uploaded" in response

def generate_summary_response(prompt, session_id):
    """
    Calls generate with the given prompt and session_id.
    """
    response = generate(
        model='4o-mini',
        system="You are a TA chatbot for CS-150: Generative AI for Social Impact.",
        query=prompt,
        temperature=0.0,
        lastk=0,
        session_id=session_id,
        rag_usage=True,
        rag_threshold=0.3,
        rag_k=1
    )
    return response.get('response', '').strip() if isinstance(response, dict) else response.strip()

def summarizing_agent(action_type, session_id):
    """
    Uploads the PDF (if needed) and then calls generate to produce a summary.
    """
    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return "Could not upload PDF."
    
    if action_type == "summarize_abstract":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, "
            "please provide a detailed summary focusing on the abstract. "
            "Include the main objectives and key points of the abstract."
        )
    elif action_type == "summarize_full":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, "
            "please provide a detailed summary of the entire paper, including the title, "
            "key findings, methodology, and conclusions."
        )
    else:
        return "Invalid summarization action."
    
    # Wait 10 seconds for the PDF to be fully processed.
    time.sleep(10)
    return generate_summary_response(prompt, session_id)

def answer_question(question, session_id):
    """
    Uploads the PDF (if needed) and then answers a specific question about the paper.
    """
    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return "Could not upload PDF."
    
    prompt = (
        f"Based solely on the research paper that was uploaded in this session, "
        f"answer the following question:\n\n{question}\n\n"
        "Provide the answer using only the content of the uploaded PDF."
    )
    time.sleep(10)
    return generate_summary_response(prompt, session_id)

def build_interactive_response(response_text, session_id):
    """
    Builds a response payload with interactive buttons for further options.
    """
    return {
        "text": response_text,
        "attachments": [
            {
                "title": "Would you like a summary?",
                "text": "Select an option:",
                "actions": [
                    {
                        "type": "button",
                        "text": "Summarize Abstract",
                        "msg": "summarize_abstract",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Summarize Full Paper",
                        "msg": "summarize_full",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }
        ]
    }

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json() or request.form  # Support JSON or form data
    print(f"DEBUG: Received request data: {data}")
    
    # Use user_name as in the TA example.
    user = data.get("user_name", "Unknown")
    message = data.get("text", "").strip()
    
    # Ignore bot messages or empty text.
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    print(f"Message from {user}: {message}")
    session_id = get_session_id(data)
    
    # If the user clicked a button, the message text will be one of our commands.
    if message == "summarize_abstract":
        summary_text = summarizing_agent("summarize_abstract", session_id)
        response = {
            "text": summary_text,
            "attachments": [
                {
                    "text": "You have selected: ✅ Summarize Abstract!",
                    "actions": [
                        {
                            "type": "button",
                            "text": "Thanks for the feedback 😃",
                            "msg": "post_acknowledge",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ]
        }
        return jsonify(response)
    
    elif message == "summarize_full":
        summary_text = summarizing_agent("summarize_full", session_id)
        response = {
            "text": summary_text,
            "attachments": [
                {
                    "text": "You have selected: ✅ Summarize Full Paper!",
                    "actions": [
                        {
                            "type": "button",
                            "text": "Thanks for the feedback 😃",
                            "msg": "post_acknowledge",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ]
        }
        return jsonify(response)
    else:
        # Otherwise, treat it as a research question.
        response_text = answer_question(message, session_id)
        response = {
            "text": response_text,
            "attachments": [
                {
                    "title": "User Options",
                    "text": "Would you like a summary?",
                    "actions": [
                        {
                            "type": "button",
                            "text": "Summarize Abstract",
                            "msg": "summarize_abstract",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "Summarize Full Paper",
                            "msg": "summarize_full",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ]
        }
        return jsonify(response)

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()