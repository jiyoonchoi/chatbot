import os
import time
import re
import requests
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
from dotenv import load_dotenv

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, 'twips_paper.pdf')

ROCKET_CHAT_URL = "https://chat.genaiconnect.net"
BOT_USER_ID = os.getenv("botUserId")
BOT_AUTH_TOKEN = os.getenv("botToken")
TA_USERNAME = os.getenv("taUserName")
MSG_ENDPOINT = os.getenv("msgEndPoint")

summary_cache = {}
processed_pdf = {}
pdf_ready = {}
conversation_history = {}
ta_msg_to_student_session = {}

app = Flask(__name__)

# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------
def get_session_id(data):
    user = data.get("user_name", "unknown_user").strip().lower()
    return f"session_{user}_twips_research"

def upload_pdf_if_needed(pdf_path, session_id):
    if processed_pdf.get(session_id):
        return True
    if not os.path.exists(pdf_path):
        return False
    try:
        response = pdf_upload(path=pdf_path, session_id=session_id, strategy="smart")
        if "Successfully uploaded" in response:
            processed_pdf[session_id] = True
            return True
        return False
    except Exception:
        return False

def wait_for_pdf_ready(session_id, max_attempts=15, delay=2):
    print(f"DEBUG: Waiting for PDF\n")
    if pdf_ready.get(session_id):
        return True
    for _ in range(max_attempts):
        response = generate_response("", "What is the title of the uploaded paper?", session_id)
        if "twips" in response.lower():
            pdf_ready[session_id] = True
            return True
        time.sleep(delay)
    return False

def ensure_pdf_processed(session_id):
    print(f"DEBUG: PDF processed\n")
    return upload_pdf_if_needed(PDF_PATH, session_id) and wait_for_pdf_ready(session_id)

def generate_response(system, prompt, session_id):
    if not system:
        system = ("You are a TA chatbot for CS-150. Answer only based on the uploaded paper. "
                  "Keep answers short, encourage users to check sections, and avoid creating your own questions.")
    response = generate(model='4o-mini', system=system, query=prompt, session_id=session_id, temperature=0.0,
                        lastk=5, rag_usage=True, rag_threshold=0.1, rag_k=5)

    if isinstance(response, dict):
        return response.get("response", "").strip()
    return response.strip()

def generate_paper_response(system, prompt, session_id):
    if not system:
        system = ("You are a TA chatbot for CS-150. Answer only based on the uploaded paper. "
                  "Keep answers short, encourage users to check sections, and avoid creating your own questions.")
    response = generate(model='4o-mini', system=system, query=prompt, session_id=session_id, temperature=0.0,
                        lastk=5, rag_usage=True, rag_threshold=0.01, rag_k=10)

    if isinstance(response, dict):
        return response.get("response", "").strip()
    return response.strip()

# def classify_message(message, session_id):
#     prompt = (
#         f"Given this message: \"{message}\", classify it in 3 ways:\n\n"
#         f"1. Topic: One of ['greeting', 'content_about_paper', 'class_logistics', 'off_topic']\n"
#         f"2. Difficulty: 'factual' or 'conceptual'\n"
#         f"3. Specificity: 'asking_for_details' or 'confirming_understanding'\n\n"
#         f"Respond in JSON:\n"
#         f"{{\"topic\": ..., \"difficulty\": ..., \"specificity\": ...}}"
#     )
#     try:
#         response = generate_response("", prompt, session_id)
#         import json
#         return json.loads(response)
#     except Exception as e:
#         print("DEBUG: Error in classify_message:", e)
#         # Fallback to default
#         return {
#             "topic": "content_about_paper",
#             "difficulty": "conceptual",
#             "specificity": "asking_for_details"
#         }


def classify_query(message, session_id):
    prompt = (f"Classify the following user question into exactly one of:\n\n"
              "- 'greeting' (if it's just hello/hi/hey)\n"
              "- 'content_about_paper' (if it asks anything about the uploaded research paper, e.g., methods, results, ideas, implications)\n"
              "- 'class_logistics' (if it asks about class logistics: deadlines, project presentations, grading, TA office hours, etc.)\n"
              "- 'off_topic' (if it talks about unrelated things like food, movies, hobbies, etc.)\n\n"
              "Return only the label itself.\n\n"
              f"User Message: \"{message}\"")
    
    classification = generate_response("", prompt, session_id).lower().strip()
    print(f"DEBUG: Classification: ", classification)
    
    if "greeting" in classification:
        return "greeting"
    if "content_about_paper" in classification:
        return "content_about_paper"
    if "class_logistics" in classification:
        return "class_logistics"
    if "off_topic" in classification:
        return "off_topic"
    return "content_about_paper"  # safe fallback


def classify_difficulty(question, session_id):
    prompt = (f"Classify the following question as 'factual' or 'conceptual'. "
              f"Factual = lookup info; Conceptual = requires explanation.\n\nQuestion: \"{question}\"")
    difficulty = generate_response("", prompt, session_id).lower()
    return "factual" if "factual" in difficulty else "conceptual"

def classify_specificity(question: str, session_id: str) -> str:
    """
    Use the LLM to classify a question as 'general' or 'specific'.
    """
    prompt = (
        "Classify the following question based on its intent:\n\n"
        "- 'asking_for_details' → if the user is trying to understand a topic, section, or process in the paper that they likely don't know yet. "
        "These questions are broad, open-ended, or exploratory.\n"
        "- 'confirming_understanding' → if the user is checking whether something they believe or suspect is correct based on the paper. "
        "These questions are often yes/no, comparative, or reflect partial understanding.\n\n"
        f"Question: \"{question}\"\n\n"
        "Respond with only one word: 'asking_for_details' or 'confirming_understanding'."
    )
    response = generate_response("", prompt, session_id)
    print(f"DEBUG: Specificity classifed as: ", response)
    return response.strip().lower()


def generate_followup(session_id, override_last_bot=None):
    if override_last_bot:
        last_bot_message = override_last_bot
    else:
        history = conversation_history.get(session_id, {}).get("messages", [])
        last_bot_message = next(
            (msg for speaker, msg in reversed(history) if speaker == "bot"),
            None
        )
    # graceful fallback
    if not last_bot_message:
        return ("I need a little context first — ask me something about "
                "the paper, then press **Get a Follow-up Question**! 😊")

    print(f"DEBUG: creating followup question \n")
    prompt = (
        f"You are acting as a TA chatbot helping a student think critically about a research paper.\n\n"
        f"Based on the last response you gave:\n\n"
        f"\"{last_bot_message}\"\n\n"
        f"Generate **one** thoughtful follow-up question that meets these goals:\n"
        f"- Can be either **open-ended** (invites reflection) or **specific** (asks for a particular detail).\n"
        f"- Should **encourage deeper thinking** about the topic.\n"
        f"- Should **feel natural**, like a real conversation.\n"
        f"- Should **stay focused** on the context of the uploaded paper (not general unrelated ideas).\n"
        f"- Keep it **short**, clear, and engaging (1-2 sentences at most).\n"
        f"- Do NOT include any extra commentary or introductions — return only the question itself.\n\n"
        f"Write the best follow-up you can!"
    )

    followup = generate_response("", prompt, session_id)
    return followup.strip()

def show_buttons(text, session_id, summary_button=False, followup_button=False):
    attachments = []
    if summary_button:
        attachments.append({
            "actions": [{
                "type": "button",
                "text": "📄 Quick Summary",
                "msg": "summarize",
                "msg_in_chat_window": True,
                "msg_processing_type": "sendMessage"
            }]
        })
    if followup_button:
        # embed the last bot message after a special prefix
        encoded = text.replace("\n", "\\n").replace('"', '\\"')
        attachments.append({
            "actions": [{
                "type": "button",
                "text": "🎲 Get a Follow-up Question",
                "msg": f"__FOLLOWUP__ | {encoded}",
                "msg_in_chat_window": True,
                "msg_processing_type": "sendMessage"
            }]
        })
    attachments.append({
        "actions": [{
            "type": "button",
            "text": "👩‍🏫 Ask a TA",
            "msg": "ask_TA",
            "msg_in_chat_window": True,
            "msg_processing_type": "sendMessage"
        }]
    })
    return {"text": text, "session_id": session_id, "attachments": attachments}

def build_TA_button():
    return {
        "text": "Select a TA to ask your question:",
        "attachments": [
            {
                "title": "Choose a TA",
                "actions": [
                    {
                        "type": "button",
                        "text": "Ask TA Aya",
                        "msg": "ask_TA_Aya",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Ask TA Jiyoon",
                        "msg": "ask_TA_Jiyoon",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Ask TA Amanda",
                        "msg": "ask_TA_Amanda",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }
        ]
    }

# -----------------------------------------------------------------------------
# TA Messaging Function (send message to TA)
# -----------------------------------------------------------------------------
def send_direct_message_to_TA(question, session_id, ta_username):
    """
    Send a direct message to the TA using Rocket.Chat.
    """
    msg_url = MSG_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": BOT_AUTH_TOKEN,
        "X-User-Id": BOT_USER_ID,
    }
    message_text = f"Student '{session_id}' asks: {question}"
    payload = {
        "channel": f"@{ta_username}",
        "text": message_text,
        "attachments": [
            {
            "actions": [
                {
                "type": "button",
                "text": "Respond to Student",
                "msg": "respond",
                "msg_in_chat_window": True,
                "msg_processing_type": "sendMessage"
                }
            ]
        }]
    }
    try:
        response = requests.post(msg_url, json=payload, headers=headers)
        resp_data = response.json()
        print("DEBUG: Direct message sent:", resp_data)
        # Extract the unique message _id returned from Rocket.Chat:
        if resp_data.get("success") and "message" in resp_data:
            message_id = resp_data["message"].get("_id")
            if message_id:
                # Save the mapping from message id to student session.
                ta_msg_to_student_session[message_id] = session_id
                print(f"DEBUG: Mapped message id {message_id} to session {session_id}")
        # print("DEBUG: Direct message sent:", response.json())
    except Exception as e:
        print("DEBUG: Error sending direct message to TA:", e)

# -----------------------------------------------------------------------------
# TA-student Messaging Function (forward question to student)
# -----------------------------------------------------------------------------
def extract_user(session_id):
    prefix = "session_"
    if session_id.startswith(prefix):
        return session_id[len(prefix):]
    else:
        return session_id
def extract_first_token(session_id):
    # First, remove the "session_" prefix.
    user_part = extract_user(session_id)
    # Then, split and get the first token.
    return user_part.split('_')[0]

def build_refinement_buttons(q_flow):
    base = q_flow.get("suggested_question") or q_flow["raw_question"]
    return {
      "attachments": [{
        "actions": [
          {"type":"button","text":"✅ Approve","msg":"approve","msg_in_chat_window":True,"msg_processing_type":"sendMessage"},
          {"type":"button","text":"✏️ Modify","msg":"modify","msg_in_chat_window":True,"msg_processing_type":"sendMessage"},
          {
            "type":"button",
            "text":"📝 Manual Edit",
            "msg": f"Editing: {base}",
            "msg_in_chat_window": True,
            "msg_processing_type": "respondWithMessage"
          },
          {"type":"button","text":"❌ Cancel","msg":"cancel","msg_in_chat_window":True,"msg_processing_type":"sendMessage"}
        ]
      }]
    }

def build_manual_edit_buttons(prefill_text):
    return {
      "attachments": [{
        "actions": [
          {
            "type": "button",
            "text": "✏️ Edit…",
            "msg": prefill_text,
            "msg_in_chat_window": True,
            "msg_processing_type": "respondWithMessage"
          },
          {
            "type": "button",
            "text": "📤 Send",
            "msg": prefill_text,
            "msg_in_chat_window": True,
            "msg_processing_type": "sendMessage"
          }
        ]
      }]
    }

def forward_message_to_student(ta_response, session_id, student_session_id):
    msg_url = MSG_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": BOT_AUTH_TOKEN,
        "X-User-Id": BOT_USER_ID,
    }
    
    ta = "Aya" if session_id == "session_aya.ismail_twips_research" else "Jiyoon" 
    message_text = (

    f"Your TA {ta} says: '{ta_response} 💬'\n\n"

    "If you want to continue this conversation, please message your TA "
    f"{ta} directly in Rocket.Chat or send a private Piazza post here:\n"
    "https://piazza.com/class/m5wtfh955vwb8/create\n\n"
    )
    
    student = extract_first_token(student_session_id)
    print(f"DEBUG****: ta session id: {session_id}")
    print(f"DEBUG: Forwarding message to student {student}: {message_text}")
    
    payload = {
        "channel": f"@{student}",
        "text": message_text
    }
    
    try:
        response = requests.post(msg_url, json=payload, headers=headers)
        print("DEBUG: TA Response forwarded to student:", response.json())
    except Exception as e:
        print("DEBUG: Error sending TA response to student:", e)
        

def generate_suggested_question(session_id, student_question, feedback=None):
    """
    Generate a rephrased and clearer version of the student's question.
    """
    print(f"DEBUG: session_id inside generate_suggested_question: {session_id}")
    ta_name = conversation_history[session_id]["question_flow"]["ta"]
    if feedback:
        prompt = (
            f"""Original question: "{student_question}"\n"""
            f"""Feedback: "{feedback}"\n\n"""
            f"""Based on session-id **{session_id}** (which encodes the student's name) """
            f"""generate a refined, concise version of the question.\n"""
            f"""Address the TA directly, e.g. 'Hi {ta_name}, …'.\n"""
            f"""• Mention the TwIPS paper if relevant.\n"""
            f"""• Keep it polite and no longer than necessary.\n"""
        )
    else:
        prompt = (
            f"""Based on session-id **{session_id}** generate a clearer version """
            f"""of the student's question.\n"""
            f"""Address the TA: 'Hi {ta_name}, …'.\n"""
            f"""Avoid adding irrelevant detail.\n\n"""
            f"""Student question: "{student_question}"\n"""
            f"""Suggested improved question:"""
        )

    response = generate(
            model='4o-mini',
            system=(
                "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
                "Rephrase or refine the student's question to be clearer and more comprehensive, "
                "incorporating any provided feedback and referring to the paper where relevant."
            ),
            query=prompt,
            temperature=0.0,
            lastk=5,
            session_id="suggestion_session",
            rag_usage=False,
            rag_threshold=0.3,
            rag_k=0
    )

    if isinstance(response, dict):
         result = response.get('response', '').strip()
    else:
         result = response.strip()
         
    # Optionally extract a quoted sentence if present
    match = re.search(r'"(.*?)"', result)
    suggested_question_clean = match.group(1) if match else result
    print(f"DEBUG: Suggested question: {result}")
    print("END OF SUGGESTED QUESTION")
    return result, suggested_question_clean

# ------------------------------------------------------------------------
# Flask Route
# ------------------------------------------------------------------------
@app.route('/query', methods=['POST'])
def query():
    print("DEBUG: Handling query...")
    data = request.get_json() or request.form
    user = data.get("user_name", "Unknown")
    message = data.get("text", "").strip()
    session_id = data.get("session_id") or get_session_id(data)

    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    if session_id not in conversation_history:
        conversation_history[session_id] = {"messages": []}
        summary_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)

    conversation_history[session_id]["messages"].append(("user", message))

    # Ensure the PDF is processed and loaded
    if not ensure_pdf_processed(session_id):
        return jsonify({"text": "PDF not processed yet. Please try again shortly."})

    # Feed the message directly to the LLM using paper-aware generation
    response = generate(
        model='4o-mini',
        system=("You are a TA chatbot for CS-150. Answer questions strictly based on the uploaded paper."
                " Encourage the student to reference specific sections. Do not invent answers outside the paper."),
        query=message,
        session_id=session_id,
        temperature=0.0,
        lastk=5,
        rag_usage=True,
        rag_threshold=0.01,
        rag_k=10
    )

    result = response.get("response", "").strip() if isinstance(response, dict) else response.strip()

    conversation_history[session_id]["messages"].append(("bot", result))
    return jsonify({"text": result, "session_id": session_id})

# ------------------------------------------------------------------------
# Server Start
# ------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()