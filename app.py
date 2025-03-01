import os
import uuid
import time
import requests
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the directory of the current script and build an absolute path to the PDF.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(os.getcwd(), 'static', 'pdfs', 'twips_paper.pdf')

app = Flask(__name__)

# Global conversation history.
conversation_history = {}

# Rocket.Chat Bot Credentials & URL
ROCKET_CHAT_URL = "https://chat.genaiconnect.net"
BOT_USER_ID = os.getenv("botUserId")
BOT_AUTH_TOKEN = os.getenv("botToken")
TA_USERNAME = os.getenv("taUserName")
MSG_ENDPOINT = os.getenv("msgEndPoint")

# TODO: NOT WORKING
def send_typing_indicator(room_id):
    """
    Sends a typing indicator to Rocket.Chat.
    """
    headers = {
        "X-Auth-Token": BOT_AUTH_TOKEN,
        "X-User-Id": BOT_USER_ID,
        "Content-type": "application/json",
    }
    payload = {"roomId": room_id}
    url = f"{ROCKET_CHAT_URL}/api/v1/chat.sendTyping"
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"DEBUG: Typing indicator response: {response.json()}")
    except Exception as e:
        print(f"DEBUG: Error sending typing indicator: {e}")

def get_session_id(data):
    """
    Returns a consistent session ID based solely on the user_name.
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

def generate_response(prompt, session_id):
    """
    Calls generate with the given prompt and session_id.
    currently used to summrize abstract/full paper and answer questions 
    """
    response = generate(
        model='4o-mini',
        system="You are a TA chatbot for CS-150: Generative AI for Social Impact.",
        query=prompt,
        temperature=0.0,
        lastk=5,
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
    
    # Send typing indicator before processing.
    time.sleep(10)  # Simulate processing delay (PDF processing)
    return generate_response(prompt, session_id)

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
    return generate_response(prompt, session_id)

def send_direct_message_to_TA(question, session_id):
    """
    Sends a direct message to the TA with the student's question.
    """
    ta_username = TA_USERNAME  
    msg_url = MSG_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": BOT_AUTH_TOKEN,
        "X-User-Id": BOT_USER_ID,
    }
    
    student = session_id[len("session_"):]
    message_text = f"Student '{student}' asks: {question}"
    
    payload = {
        "channel": f"@{ta_username}",
        "text": message_text
    }
    
    try:
        response = requests.post(msg_url, json=payload, headers=headers)
        print("DEBUG: Direct message sent:", response.json())
    except Exception as e:
        print("DEBUG: Error sending direct message to TA:", e)


def classify_query(message):
    """
    Classifies the user message as 'greeting', 'research', or 'other'.
    """
    prompt = (
        "Determine if the following message is a greeting, a query about the research paper, or something else. "
        "Reply with one word: 'greeting' or 'not greeting'.\n\n"
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
            }, 
            {
                "title": "Ask a TA",
                "text": "Do you have a question for your TA?", 
                "actions": [
                    {
                        "type": "button",
                        "text": "Ask a TA",
                        "msg": "ask_TA",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }
        ]
    }

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json() or request.form  # Support JSON and form data
    print(f"DEBUG: Received request data: {data}")
    
    # Use user_name as in the TA example.
    user = data.get("user_name", "Unknown")
    message = data.get("text", "").strip()
    
    # Extract room id if provided (for typing indicator)
    room_id = data.get("roomId") or data.get("rid")
    print(f"DEBUG: User: {user}, Message: {message}, Room ID: {room_id}")
    
    # Ignore bot messages or empty text.
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    # Trigger the typing indicator if room id and credentials are available.
    if room_id:
        send_typing_indicator(room_id)
    
    print(f"Message from {user}: {message}")
    session_id = get_session_id(data)

    if session_id not in conversation_history:
        conversation_history[session_id] = {"messages": [], "awaiting_ta_question": False}

    # Check if we are awaiting a question for TA
    if conversation_history[session_id].get("awaiting_ta_question", False): # **********
        # Process this message as the TA question.
        ta_question = message  # The message is the question.
        send_direct_message_to_TA(ta_question, user)  
        confirmation = "Your TA question has been forwarded. They will get back to you soon."
        # Reset the waiting flag.
        conversation_history[session_id]["awaiting_ta_question"] = False # **********
        return jsonify({"text": confirmation, "session_id": session_id})
    
    # If the message is exactly "summarize_abstract" or "summarize_full", handle the summarization button clicks.
    if message == "summarize_abstract":
        summary_text = summarizing_agent("summarize_abstract", session_id)
        return jsonify({"text": summary_text, "session_id": session_id})
    
    elif message == "summarize_full":
        summary_text = summarizing_agent("summarize_full", session_id)
        return jsonify({"text": summary_text, "session_id": session_id})
    
    # send a direct message to the TA.
    elif message == "ask_TA":
        prompt = "Please type your question for your TA."
        conversation_history[session_id]["awaiting_ta_question"] = True # **********
        return jsonify({"text": prompt, "session_id": session_id})
    
    else:
        # For general messages, classify the query.
        # conversation_history.setdefault(session_id, {"messages": [], "awaiting_ta_question": False})
        conversation_history[session_id]["messages"].append(("user", message))
       
        classification = classify_query(message)
        print(f"DEBUG: User message classified as: {classification}")
        
        if classification == "not greeting":
            answer = answer_question(message, session_id)
            # conversation_history.setdefault(session_id, []).append(("bot", answer))

            # conversation_history.setdefault(session_id, {"messages": [], "awaiting_ta_question": False})
            conversation_history[session_id]["messages"].append(("bot", answer))
            return jsonify({"text": answer, "session_id": session_id})
        
        
        elif classification == "greeting":
            greeting_msg = "Hello! Please ask a question about the research paper, or use the buttons below for a detailed summary."
           
            # conversation_history.setdefault(session_id, {"messages": [], "awaiting_ta_question": False})
            conversation_history[session_id]["messages"].append(("bot", greeting_msg))
            return jsonify(build_interactive_response(greeting_msg, session_id))
        # else:
        #     concise_summary = generate_summary_response(
        #         "Provide a 1-2 sentence summary of the research paper.", session_id
        #     )
        #     conversation_history.setdefault(session_id, []).append(("bot", concise_summary))
        #     summary_text = (
        #         f"Summary: {concise_summary}\n\n"
        #         "Would you like a detailed summary of the abstract or full paper?\n"
        #         "Or ask a specific question."
        #     )
        #     return jsonify(build_interactive_response(summary_text, session_id))
        return jsonify({"text": "Sorry, I didn't understand that.", "session_id": session_id})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()