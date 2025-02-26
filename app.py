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
    Returns a consistent session ID based solely on the user_id.
    """
    user_id = data.get("user_id", "unknown_user").strip().lower()
    return f"session_{user_id}"

def upload_pdf_if_needed(pdf_path, session_id):
    """
    Uploads the PDF using pdf_upload with the provided session_id.
    Returns True if the upload response indicates success.
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

def classify_query(message):
    """
    Classifies the user message as 'greeting', 'research', or 'other'.
    """
    prompt = (
        "Determine if the following message is a greeting, a query about the research paper, or something else. "
        "Reply with one word: 'greeting', 'research', or 'other'.\n\n"
        f"Message: \"{message}\""
    )
    print(f"DEBUG: Classifying query: {message}")
    classification = generate(
        model='4o-mini',
        system="You are a query classifier.",
        query=prompt,
        temperature=0.0,
        lastk=0,
        session_id="classify_" + str(uuid.uuid4()),
        rag_usage=True,
        rag_threshold=0.3,
        rag_k=1
    )
    return classification.get('response', '').strip().lower() if isinstance(classification, dict) else classification.strip().lower()

# Button action handlers.
def handle_summarize_abstract(session_id):
    print(f"DEBUG: Summarizing abstract for session {session_id}")
    return summarizing_agent("summarize_abstract", session_id)

def handle_summarize_full(session_id):
    print(f"DEBUG: Summarizing full paper for session {session_id}")
    return summarizing_agent("summarize_full", session_id)

def build_interactive_response(text, session_id):
    """
    Builds a response payload with interactive buttons.
    The buttons use "msg_in_chat_window": true so that they are visible,
    but their "msg" value is just an internal action identifier.
    """
    return {
        "text": text,
        "session_id": session_id,
        "attachments": [
            {
                "callback_id": "summary_buttons",
                "actions": [
                    {
                        "type": "button",
                        "text": "Summarize Abstract",
                        "msg": "summarize_abstract",  # This is the internal action identifier.
                        "msg_in_chat_window": True
                    },
                    {
                        "type": "button",
                        "text": "Summarize Full Paper",
                        "msg": "summarize_full",  # This is the internal action identifier.
                        "msg_in_chat_window": True
                    }
                ]
            }
        ]
    }

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json() or request.form  # Support JSON and form-encoded requests
    print(f"DEBUG: Received request data: {data}")
    session_id = get_session_id(data)
    
    # Check if this is a button click.
    action = data.get("action") or data.get("msg")
    if action:
        print(f"DEBUG: Button clicked: {action}")
        # Process the button click without sending the static action message.
        if action == "summarize_abstract":
            summary_text = handle_summarize_abstract(session_id)
        elif action == "summarize_full":
            summary_text = handle_summarize_full(session_id)
        else:
            summary_text = "Unknown action."
        
        conversation_history.setdefault(session_id, []).append(("bot", summary_text))
        # Return the final summary as the message that gets sent in chat.
        return jsonify({"msg": summary_text, "session_id": session_id})
    
    # Process regular text messages.
    message = data.get("text", "").strip()
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    conversation_history.setdefault(session_id, []).append(("user", message))
    classification = classify_query(message)
    print(f"DEBUG: User message classified as: {classification}")
    
    if classification == "research":
        answer = answer_question(message, session_id)
        conversation_history.setdefault(session_id, []).append(("bot", answer))
        return jsonify({"msg": answer, "session_id": session_id})
    elif classification == "greeting":
        greeting_msg = "Hello! Please ask a question about the research paper or use the buttons below."
        conversation_history.setdefault(session_id, []).append(("bot", greeting_msg))
        return jsonify(build_interactive_response(greeting_msg, session_id))
    else:
        concise_summary = generate_summary_response(
            "Provide a 1-2 sentence summary of the research paper.", session_id
        )
        conversation_history.setdefault(session_id, []).append(("bot", concise_summary))
        summary_text = (
            f"Summary: {concise_summary}\n\n"
            "Would you like a detailed summary of the abstract or full paper?\n"
            "Or ask a specific question."
        )
        return jsonify(build_interactive_response(summary_text, session_id))

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()