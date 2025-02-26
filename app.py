import os
import uuid
import time
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PDF to be processed (the reading for this week)
PDF_PATH = "research_paper.pdf"

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

def generate_pdf_response(prompt, session_id):
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
    if isinstance(response, dict):
        return response.get('response', '').strip()
    else:
        return response.strip()

def summarizing_agent(action_type, session_id):
    """
    Uploads the PDF (if needed) and then calls generate to produce a summary.
    The prompt instructs the model to use the PDF uploaded under the given session.
    """
    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return "Could not upload PDF."
    
    if action_type == "summarize_abstract":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, please provide a detailed summary "
            "focusing on the abstract. Include the main objectives and key points of the abstract. Do not use any external context."
        )
    elif action_type == "summarize_full":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, please provide a detailed summary of the entire paper. "
            "Include the title, key findings, methodology, and conclusions. Do not use any external context."
        )
    else:
        return "Invalid summarization action."
    
    # Wait 10 seconds for the PDF to be fully processed
    time.sleep(10)
    return generate_pdf_response(prompt, session_id)

def answer_question(question, session_id):
    """
    Uploads the PDF (if needed) and then answers a specific question about the paper
    using the uploaded PDF as context.
    """
    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return "Could not upload PDF."
    
    prompt = (
        f"Based solely on the research paper that was uploaded in this session, answer the following question:\n\n{question}\n\n"
        "Provide the answer using only the content of the uploaded PDF."
    )
    # Wait 10 seconds to ensure the PDF is processed.
    time.sleep(10)
    return generate_pdf_response(prompt, session_id)

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
    if isinstance(classification, dict):
        return classification.get('response', '').strip().lower()
    else:
        return classification.strip().lower()

# Button action handlers.
def handle_summarize_abstract(session_id):
    return summarizing_agent("summarize_abstract", session_id)

def handle_summarize_full(session_id):
    return summarizing_agent("summarize_full", session_id)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    print(f"DEBUG: Received request data: {data}")
    session_id = get_session_id(data)
    
    # If an "action" key is present (i.e. a button press), handle it directly.
    action = data.get("action")
    if action:
        if action == "summarize_abstract":
            summary_text = handle_summarize_abstract(session_id)
            conversation_history.setdefault(session_id, []).append(("bot", summary_text))
            return jsonify({"text": summary_text, "session_id": session_id})
        elif action == "summarize_full":
            summary_text = handle_summarize_full(session_id)
            conversation_history.setdefault(session_id, []).append(("bot", summary_text))
            return jsonify({"text": summary_text, "session_id": session_id})
    
    # Process as a normal text message.
    message = data.get("text", "").strip()
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    conversation_history.setdefault(session_id, []).append(("user", message))
    classification = classify_query(message)
    print(f"DEBUG: User message classified as: {classification}")
    
    if classification == "research":
        answer = answer_question(message, session_id)
        conversation_history.setdefault(session_id, []).append(("bot", answer))
        return jsonify({"text": answer, "session_id": session_id})
    elif classification == "greeting":
        greeting_msg = "Hello! Please ask a question about the research paper, or use the buttons below for a detailed summary."
        conversation_history.setdefault(session_id, []).append(("bot", greeting_msg))
        return jsonify({"text": greeting_msg, "session_id": session_id})
    else:
        # For any other query, generate an interactive message with a concise summary and buttons.
        concise_prompt = (
            "Based solely on the research paper that was uploaded in this session, please provide a concise 1-2 sentence summary."
        )
        concise_summary = generate_pdf_response(concise_prompt, session_id)
        interactive_message = {
            "text": (
                f"Weekly Reading Summary: {concise_summary}\n\n"
                "Would you like a more detailed summary of the abstract or the full paper?\n"
                "Or ask a specific question about the paper."
            ),
            "attachments": [
                {
                    "actions": [
                        {
                            "type": "button",
                            "text": "Summarize Abstract",
                            "action": "summarize_abstract",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "Summarize Full Paper",
                            "action": "summarize_full",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ]
        }
        conversation_history.setdefault(session_id, []).append(("bot", concise_summary))
        return jsonify(interactive_message)

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()