import os
import uuid
import io
import tempfile
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Load environment variables (for local testing or production settings)
load_dotenv()

# PDF to be processed (the reading for this week)
PDF_PATH = "ChatGPT Dissatisfaction Paper.pdf"

app = Flask(__name__)

# Global conversation history.
conversation_history = {}

def get_session_id(data):
    """
    Standardizes and returns a session ID.
    If a session_id is provided, it trims and lowercases it.
    Otherwise, generates one based on the user_id.
    """
    user_id = data.get("user_id", "unknown_user").strip().lower()
    if "session_id" in data and data["session_id"].strip():
        return data["session_id"].strip().lower()
    else:
        return f"session_{user_id}"

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using the pdf_upload function from llmproxy.
    Uses the 'smart' strategy to process the PDF.
    """
    try:
        response = pdf_upload(
            path=pdf_path,
            session_id="pdf_upload_" + str(uuid.uuid4()),
            strategy="smart"
        )
        # If pdf_upload returns a dictionary, extract the text from the 'text' key.
        # Otherwise, assume the response is already the extracted text.
        if isinstance(response, dict):
            pdf_text = response.get("text", "")
        else:
            pdf_text = response

        if pdf_text:
            print(f"DEBUG: Extracted text from PDF using pdf_upload at {pdf_path}")
        else:
            print(f"DEBUG: No text returned from pdf_upload for {pdf_path}")
        return pdf_text
    except Exception as e:
        print(f"DEBUG: Error during pdf_upload: {e}")
        return ""

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
        system="You are a TA chatbot for the Tufts course CS-150: Generative AI for Social Impact. Your task is to be an expert on this week’s reading.",
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
def classify_query(message):
    """
    Classifies the user message as 'greeting', 'research', or 'other'.
    This function no longer includes PDF text extraction.
    """
    prompt = (
        "Determine if the following message is a greeting, a query about this week's research paper "
        "(reading for CS-150: Generative AI for Social Impact), or something else. "
        "Reply with just one word: 'greeting', 'research', or 'other'.\n\n"
        f"Message: \"{message}\""
    )
    print(f"DEBUG: Classifying query: {message}")
    classification = generate(
        model='4o-mini',
        system="You are a query classifier for CS-150. Classify the user message as 'greeting', 'research', or 'other'.",
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

# Button action handlers.
def handle_summarize_abstract():
    return summarizing_agent(PDF_PATH, "summarize_abstract")

def handle_summarize_full():
    return summarizing_agent(PDF_PATH, "summarize_full")

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    print(f"DEBUG: Received request data: {data}")

    message = data.get("text", "").strip()
    session_id = get_session_id(data)
    
    # Ignore messages from the bot itself.
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    # Initialize conversation history with an intro message if needed.
    if session_id not in conversation_history:
        intro_message = (
            "Hello! I am your friendly TA for CS-150: Generative AI for Social Impact. "
            "Let's review this week's reading."
        )
        conversation_history[session_id] = [("bot", intro_message)]
    conversation_history[session_id].append(("user", message))
    
    # If the user clicked one of the interactive buttons, handle those actions.
    if message.lower() == "summarize abstract":
        summary_text = handle_summarize_abstract()
        conversation_history[session_id].append(("bot", summary_text))
        return jsonify({"text": summary_text, "session_id": session_id})
    elif message.lower() == "summarize full paper":
        summary_text = handle_summarize_full()
        conversation_history[session_id].append(("bot", summary_text))
        return jsonify({"text": summary_text, "session_id": session_id})
    
    # Call classify_query to log the classification result.
    classification = classify_query(message)
    print(f"DEBUG: User message classified as: {classification}")

    # For any other query, always generate an interactive message that prompts the user
    # to ask a question about the paper or request a summary.
    if not os.path.exists(PDF_PATH):
        error_msg = f"PDF not found at {PDF_PATH}"
        print(f"DEBUG: {error_msg}")
        return jsonify({"error": error_msg}), 400

    pdf_text = extract_text_from_pdf(PDF_PATH)
    if not pdf_text.strip():
        error_msg = "Could not extract text from the provided PDF."
        print(f"DEBUG: {error_msg}")
        return jsonify({"error": error_msg}), 400

    summary_prompt = (
        f"Please provide a concise 1-2 sentence summary of the reading for this week of CS-150: Generative AI for Social Impact based on the following paper text:\n\n{pdf_text}"
    )
    concise_summary_response = generate(
        model='4o-mini',
        system="You are a TA chatbot for CS-150: Generative AI for Social Impact. Provide a concise summary of this week's reading.",
        query=summary_prompt,
        temperature=0.0,
        lastk=0,
        session_id="pdf_concise_" + str(uuid.uuid4())
    )
    if isinstance(concise_summary_response, dict):
        concise_summary = concise_summary_response.get('response', '').strip()
    else:
        concise_summary = concise_summary_response.strip()

    interactive_message = {
        "text": (
            f"Weekly Reading Summary: {concise_summary}\n\n"
            "Would you like a more detailed summary of the abstract or the full paper?\n"
            "Please ask a question about the paper or use the buttons below."
        ),
        "attachments": [
            {
                "actions": [
                    {
                        "type": "button",
                        "text": "Summarize Abstract",
                        "msg": "Summarize Abstract",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Summarize Full Paper",
                        "msg": "Summarize Full Paper",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }
        ]
    }
    conversation_history[session_id].append(("bot", concise_summary))
    return jsonify(interactive_message)

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()