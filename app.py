import os
import uuid
import io
import tempfile
from flask import Flask, request, jsonify
from llmproxy import generate
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin
import PyPDF2
from bs4 import BeautifulSoup

# Load environment variables (for local testing or production settings)
load_dotenv()

# PDF to be processed (the reading for this week)
PDF_PATH = "WebServer/ChatGPT Dissatisfaction Paper.pdf"

app = Flask(__name__)

# Global conversation history.
conversation_history = {}

# Global mapping for interactive command tokens.
action_tokens = {}

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        print(f"DEBUG: Extracted text from PDF at {pdf_path}")
    except Exception as e:
        print(f"DEBUG: Error reading PDF: {e}")
    return text

# LLM AGENT: Summarize the PDF reading.
def summarizing_agent(pdf_path, action_type):
    """
    Extracts text from the given PDF and performs a detailed summarization based
    on the action_type: either 'summarize_abstract' or 'summarize_full'.
    The prompts include the context for CS-150: Generative AI for Social Impact.
    Uses the entire PDF text for summarization.
    """
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text.strip():
        return "Could not extract text from the provided PDF."
    
    if action_type == "summarize_abstract":
        summary_prompt = (
            f"Please provide a detailed summary focusing on the abstract of the reading for this week of CS-150: Generative AI for Social Impact based on the following paper text:\n\n{pdf_text}"
        )
    elif action_type == "summarize_full":
        summary_prompt = (
            f"Please provide a detailed summary of the full reading for this week of CS-150: Generative AI for Social Impact based on the following paper text, including key findings, methodology, and conclusions:\n\n{pdf_text}"
        )
    else:
        return "Invalid summarization action."
    
    summary_response = generate(
        model='4o-mini',
        system="You are a TA chatbot for the Tufts course called CS-150: Generative AI for Social Impact. Your task is to be an expert on the reading for this week.",
        query=summary_prompt,
        temperature=0.0,
        lastk=0,
        session_id="summarize_" + str(uuid.uuid4())
    )
    
    if isinstance(summary_response, dict):
        summary_text = summary_response.get('response', '').strip()
    else:
        summary_text = summary_response.strip()
    
    return summary_text

# LLM AGENT: Classify the query.
def classify_query(pdf_path, message):
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text.strip():
        return "Could not extract text from the provided PDF."
    
    prompt = (
        f"Determine if the following message is a greeting, a query about this week's research paper (\n\n{pdf_text}), or something else. "
        f"Reply with just one word: 'greeting' if it's a simple greeting, 'research' if it's asking "
        f"for research-related information, or 'other' if it is unrelated to research. "
        f"Message: \"{message}\""
    )
    print(f"DEBUG: Classifying query: {message}")
    classification = generate(
        model='4o-mini',
        system="You are a query classifier. Classify the user message as 'greeting', 'research', or 'other'.",
        query=prompt,
        temperature=0.0,
        lastk=0,
        session_id="classify_" + str(uuid.uuid4())
    )
    if isinstance(classification, dict):
        classification_text = classification.get('response', '').strip().lower()
    else:
        classification_text = classification.strip().lower()
    print(f"DEBUG: Classification result: {classification_text}")
    return classification_text

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    print(f"DEBUG: Received request data: {data}")

    message = data.get("text", "")
    user_id = data.get("user_id", "unknown_user")
    session_id = data.get("session_id", f"session_{user_id}_{str(uuid.uuid4())}")
    
    # Ignore messages from the bot itself.
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    # Check if the message is a direct execution command from an interactive button.
    if message.startswith("/execute_action"):
        parts = message.split(maxsplit=1)
        if len(parts) < 2:
            error_msg = "No action token provided."
            print(f"DEBUG: {error_msg}")
            return jsonify({"error": error_msg}), 400
        token = parts[1].strip()
        if token not in action_tokens:
            error_msg = "Invalid or expired action token."
            print(f"DEBUG: {error_msg}")
            return jsonify({"error": error_msg}), 400
        action_type, pdf_path = action_tokens.pop(token)
        summary_text = summarizing_agent(pdf_path, action_type)
        return jsonify({"text": summary_text})
    
    # Initialize conversation history with an intro message if needed.
    if session_id not in conversation_history:
        intro_message = (
            "Hello! I am your friendly TA for CS-150: Generative AI for Social Impact. "
            "Let's review this week's reading."
        )
        conversation_history[session_id] = [("bot", intro_message)]
    else:
        intro_message = None
    conversation_history[session_id].append(("user", message))

    classification = classify_query(PDF_PATH, message)

    if classification == "research":
        # Use the provided PDF path if available, otherwise fall back to the global PDF_PATH.
        pdf_path = data.get("pdf_path", PDF_PATH)
        if not os.path.exists(pdf_path):
            error_msg = f"PDF not found at {pdf_path}"
            print(f"DEBUG: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Extract text from the PDF and generate a concise summary specifically for the weekly reading.
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            error_msg = "Could not extract text from the provided PDF."
            print(f"DEBUG: {error_msg}")
            return jsonify({"error": error_msg}), 400

        summary_prompt = (
            f"Please provide a concise 1-2 sentence summary of the reading for this week of CS-150: Generative AI for Social Impact based on the following paper text:\n\n{pdf_text}"
        )
        concise_summary_response = generate(
            model='4o-mini',
            system="You are a TA chatbot for the Tufts course called CS-150: Generative AI for Social Impact. Your task is to be an expert on the reading for this week.",
            query=summary_prompt,
            temperature=0.0,
            lastk=0,
            session_id="pdf_concise_" + str(uuid.uuid4())
        )
        if isinstance(concise_summary_response, dict):
            concise_summary = concise_summary_response.get('response', '').strip()
        else:
            concise_summary = concise_summary_response.strip()

        # Generate opaque tokens for the two interactive actions.
        abstract_token = str(uuid.uuid4())
        full_token = str(uuid.uuid4())
        action_tokens[abstract_token] = ("summarize_abstract", pdf_path)
        action_tokens[full_token] = ("summarize_full", pdf_path)

        # Build interactive Rocket.Chat message with buttons that use the opaque tokens.
        interactive_message = {
            "text": (
                f"Weekly Reading Summary: {concise_summary}\n\n"
                "Would you like a more detailed summary of the abstract or a detailed summary of the full paper?"
            ),
            "attachments": [
                {
                    "actions": [
                        {
                            "type": "button",
                            "text": "Summarize Abstract",
                            "msg": f"/execute_action {abstract_token}",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "Summarize Full Paper",
                            "msg": f"/execute_action {full_token}",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ]
        }
        conversation_history[session_id].append(("bot", concise_summary))
        return jsonify(interactive_message)

    elif classification == "greeting":
        query_with_context = "\n".join(text for _, text in conversation_history[session_id])
        general_response = generate(
            model='4o-mini',
            system="You are a friendly TA chatbot for CS-150: Generative AI for Social Impact. Please prompt the user with the weekly reading summary.",
            query=query_with_context,
            temperature=0.5,
            lastk=0,
            session_id=session_id
        )
        if isinstance(general_response, dict):
            response_text = general_response.get('response', "").strip()
        else:
            response_text = general_response.strip()
        bot_reply = response_text
    elif classification == "other":
        bot_reply = (
            "I'm the TA for CS-150: Generative AI for Social Impact. I can help you with this week's reading. "
            "Please ask me about the reading or request a summary."
        )
    else:
        query_with_context = "\n".join(text for _, text in conversation_history[session_id])
        general_response = generate(
            model='4o-mini',
            system="You are a friendly TA chatbot for CS-150: Generative AI for Social Impact. Please assist the student with questions about the weekly reading.",
            query=query_with_context,
            temperature=0.5,
            lastk=0,
            session_id=session_id
        )
        if isinstance(general_response, dict):
            response_text = general_response.get('response', "").strip()
        else:
            response_text = general_response.strip()
        bot_reply = response_text

    conversation_history[session_id].append(("bot", bot_reply))
    return jsonify({"text": bot_reply, "session_id": session_id})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()