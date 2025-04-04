import os
import uuid
import time
import re
import requests
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
from dotenv import load_dotenv
import threading

# -----------------------------------------------------------------------------
# Configuration and Global Variables
# -----------------------------------------------------------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, 'twips_paper.pdf')
print("DEBUG: PDF path set to:", PDF_PATH)

# Rocket.Chat credentials and endpoints
ROCKET_CHAT_URL = "https://chat.genaiconnect.net"
BOT_USER_ID = os.getenv("botUserId")
BOT_AUTH_TOKEN = os.getenv("botToken")
TA_USERNAME = os.getenv("taUserName")
MSG_ENDPOINT = os.getenv("msgEndPoint")

# Global caches (session-specific)
summary_abstract_cache = {}
summary_full_cache = {}
processed_pdf = {}
pdf_ready = {}
conversation_history = {}

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_session_id(data):
    """Generate a session id based on the user name."""
    user = data.get("user_name", "unknown_user").strip().lower()
    return f"session_{user}"

# -----------------------------------------------------------------------------
# PDF Handling Functions
# -----------------------------------------------------------------------------
def upload_pdf_if_needed(pdf_path, session_id):
    """Upload the PDF if it hasn't already been processed for the session."""
    print(f"DEBUG: upload_pdf_if_needed() called for session {session_id}")
    if not os.path.exists(pdf_path):
        print(f"DEBUG: PDF file not found at {pdf_path}")
        return False
    print("DEBUG: PDF file found, proceeding with upload...")
    if processed_pdf.get(session_id):
        print(f"DEBUG: PDF already processed for session {session_id}.")
        return True
    print("DEBUG: Uploading PDF now...")
    try:
        response = pdf_upload(
            path=pdf_path,
            session_id=session_id,
            strategy="smart"
        )
        print(f"DEBUG: Upload response: {response}")
        if "Successfully uploaded" in response:
            processed_pdf[session_id] = True
            return True
        else:
            print("DEBUG: Upload failed, unexpected response:", response)
            return False
    except Exception as e:
        print(f"DEBUG: Exception in pdf_upload(): {e}")
        return False

def wait_for_pdf_readiness(session_id, max_attempts=10, delay=5):
    """
    Poll until the PDF is indexed and its content is integrated.
    We test by asking for the title and checking that a valid title is returned.
    """
    if pdf_ready.get(session_id):
        print(f"DEBUG: PDF already marked as ready for session {session_id}")
        return True
    for attempt in range(max_attempts):
        print(f"DEBUG: Checking PDF readiness (Attempt {attempt+1}/{max_attempts}) for session {session_id}")
        test_prompt = (
            "Based solely on the research paper that was uploaded in this session, "
            "what is the title of the paper?"
        )
        test_response = generate_response(test_prompt, session_id)
        print(f"DEBUG: Test response (Attempt {attempt+1}): {test_response}")
        if ("unable to access" not in test_response.lower() and 
            "i don't have the capability" not in test_response.lower()):
            pdf_ready[session_id] = True
            print(f"DEBUG: PDF readiness confirmed on attempt {attempt+1} for session {session_id}")
            return True
        print(f"DEBUG: PDF not ready on attempt {attempt+1} for session {session_id}")
        time.sleep(delay)
    print(f"DEBUG: PDF failed to be indexed within timeout for session {session_id}")
    return False

def ensure_pdf_processed(session_id):
    """
    Ensure the PDF has been uploaded and processed for this session.
    Uses cached flags to avoid re-analyzing the PDF on every query.
    """
    if processed_pdf.get(session_id) and pdf_ready.get(session_id):
        print(f"DEBUG: PDF already processed and ready for session {session_id}")
        return True
    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return False
    return wait_for_pdf_readiness(session_id)

# -----------------------------------------------------------------------------
# LLM Generation Functions
# -----------------------------------------------------------------------------
def generate_response(prompt, session_id):
    """
    Generate a response based on the prompt using a fixed system prompt.
    """
    system_prompt = (
        "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
        "As a TA, you challenge students to practice critical thinking and question "
        "assumptions about the research paper. Your knowledge is based solely on the "
        "research paper that was uploaded in this session."
    )

    print(f"DEBUG: Sending prompt for session {session_id}: {prompt}")
    response = generate(
        model='4o-mini',
        system=system_prompt,
        query=prompt,
        temperature=0.0,
        lastk=5,
        session_id=session_id,
        rag_usage=True,
        rag_threshold=0.3,
        rag_k=5
    )
    if isinstance(response, dict):
        result = response.get('response', '').strip()
        rag_context = response.get('rag_context', None)
        print(f"DEBUG: Received response for session {session_id}: {result}")
        if rag_context:
            print(f"DEBUG: RAG context for session {session_id}: {rag_context}")
    else:
        result = response.strip()
        print(f"DEBUG: Received response for session {session_id}: {result}")
    return result

def generate_greeting_response(prompt, session_id):
    """
    Generate a greeting response without any follow-up question.
    This function uses a system prompt that omits any instruction to ask follow-up questions.
    """
    system_prompt = (
        "As a TA chatbot for CS-150: Generative AI for Social Impact, "
        "your job is to help the student think critically about the research paper. "
        "Provide a concise answer based solely on the research paper that was uploaded in this session. "
        "Do not include any follow-up questions in your response."
    )
    
    print(f"DEBUG: Sending greeting prompt for session {session_id}: {prompt}")
    response = generate(
        model='4o-mini',
        system=system_prompt,
        query=prompt,
        temperature=0.0,
        lastk=5,
        session_id=session_id,
        rag_usage=True,
        rag_threshold=0.3,
        rag_k=5
    )
    if isinstance(response, dict):
        result = response.get('response', '').strip()
    else:
        result = response.strip()
    print(f"DEBUG: Received greeting response for session {session_id}: {result}")
    return result

def generate_follow_up(session_id):
    """
    Generate a follow-up question based on the conversation history.
    This function analyzes previous messages and uses an LLM call to generate
    a follow-up question that encourages the student to reflect further.
    If no follow-up is appropriate, it returns an empty string.
    """
    # Build conversation context from history
    messages = conversation_history.get(session_id, {}).get("messages", [])
    if not messages:
        return ""
    
    # Create a textual summary of the conversation so far.
    context = "\n".join([f"{speaker}: {msg}" for speaker, msg in messages])
    
    followup_prompt = (
        "Based on the conversation so far, generate a follow-up question that "
        "encourages the student to think more critically about the research paper. "
        "If it doesn't make sense to ask a follow-up question, return an empty string.\n\n"
        "Conversation:\n" + context
    )
    
    system_prompt = (
        "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
        "Your role is to help students practice critical thinking by asking follow-up questions about the research paper."
    )
    
    print(f"DEBUG: Sending follow-up prompt for session {session_id}: {followup_prompt}")
    response = generate(
        model='4o-mini',
        system=system_prompt,
        query=followup_prompt,
        temperature=0.0,
        lastk=5,
        session_id=f"{session_id}_followup",
        rag_usage=True,
        rag_threshold=0.3,
        rag_k=5
    )
    
    if isinstance(response, dict):
        result = response.get("response", "").strip()
    else:
        result = response.strip()
    
    print(f"DEBUG: Follow-up question for session {session_id}: {result}")
    return result

def classify_query(message, session_id):
    """
    Classify the incoming message into one of three categories:
      - greeting
      - content_answerable
      - human_TA_query (not included rn)
      
    A query is considered 'content_answerable' if it can be answered solely based on the content of the uploaded research paper.
    A query is 'human_TA_query' if it includes topics that require human judgment (e.g. ambiguous deadlines, scheduling, or info not covered in the paper).
    """
    prompt = (
        # "Classify the following query into one of these categories: 'greeting', 'content_answerable', or 'human_TA_query'. "
        "Classify the following query into one of these categories: 'greeting', 'not greeting'. "
        "If the query is a greeting, answer with 'greeting'. "
        "If the query is not a greeting, answer with 'not greeting'. "
        # "If the query is a greeting, answer with 'greeting'. "
        # "If the query can be answered solely based on the uploaded research paper, answer with 'content_answerable'. "
        # "If the query involves deadlines, scheduling, or requires additional human judgment beyond the paper's content, answer with 'human_TA_query'.\n\n"
        f"Query: \"{message}\""
    )
    classification = generate_response(prompt, session_id)
    classification = classification.lower().strip()
    print(f"DEBUG: Query classified as: {classification}")
    # If the classification doesn't match our expected labels, default to content_answerable.
    # if classification in ["greeting", "content_answerable", "human_ta_query"]:
    if classification in ["greeting"]:
        return classification
    # return "content_answerable"

def generate_suggested_question(student_question):
    prompt = (
        f"""Your job is to suggest a clearer and more comprehensive version of
        a student question that they want to ask for their CS 150 TA about the TWIPs paper:\n\n"""
        f"Student question: \"{student_question}\"\n\n"
        "Suggested improved question:"
    )

    suggested_question = generate_response(prompt, "suggestion_session")

    suggested_question_full = suggested_question.strip()

    # Extract only the first sentence for sending to the TA
    suggested_question_clean = re.search(r'"(.*?)"', suggested_question_full)

    if suggested_question_clean:
        suggested_question_clean = suggested_question_clean.group(1)  # Extracted text inside the first quotation marks
    else:
        suggested_question_clean = suggested_question_full  # Fallback if no quotes are found

    print(f"DEBUG: Suggested question: {suggested_question}")
    print("END OF SUGGESTED QUESTION")
    return suggested_question_full, suggested_question_clean


# def should_ask_ta(query, session_id):
#     """
#     Use an LLM to determine whether additional clarification is needed by the TA.
#     The prompt now explicitly instructs the LLM to flag queries about deadlines,
#     submission dates, or scheduling as requiring human TA intervention.
#     """
#     prompt = (
#         f"Analyze the following query and decide if it requires human TA intervention for additional clarification, "
#         f"or if it can be answered using only the information contained in the uploaded PDF in this session.\n\n"
#         f"Query: \"{query}\"\n\n"
#         "If the query mentions deadlines, submission dates, or scheduling, reply with 'ask TA'. "
#         "If the query is answerable using only the PDF content, reply with 'answerable'."
#     )
#     response = generate_response(prompt, session_id)
#     print(f"DEBUG: should_ask_ta response for session {session_id}: {response}")
#     return "ask ta" in response.lower()

# -----------------------------------------------------------------------------
# Response Building Functions
# -----------------------------------------------------------------------------
def build_interactive_response(response_text, session_id):
    """Build interactive response payload with summary and TA options."""
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

def build_menu_response():
    """Return the full interactive menu."""
    return {
        "text": "Select an option:",
        "attachments": [
            {
                "title": "Select an option:",
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
                "title": "Do you have a question for your TA?",
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

def show_menu(response_text, session_id):
    return {
        "text": response_text,
        "session_id": session_id,
        "attachments": [
            {
                "actions": [
                    {
                        "type": "button",
                        "text": "Menu",
                        "msg": "menu",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }
        ]
    }

def add_menu_button(response_payload):
    """Add a Menu button to the existing response payload."""
    response_payload["attachments"] = [
        {
            "actions": [
                {
                    "type": "button",
                    "text": "Menu",
                    "msg": "menu",
                    "msg_in_chat_window": True,
                    "msg_processing_type": "sendMessage"
                }
            ]
        }
    ]
    return response_payload

def build_TA_button():
    """Build TA selection buttons for asking a TA."""
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
                    }
                ]
            }
        ]
    }

# -----------------------------------------------------------------------------
# TA Messaging Function
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
        "text": message_text
    }
    try:
        response = requests.post(msg_url, json=payload, headers=headers)
        print("DEBUG: Direct message sent:", response.json())
    except Exception as e:
        print("DEBUG: Error sending direct message to TA:", e)

# -----------------------------------------------------------------------------
# Summarization and Question Answering Agents
# -----------------------------------------------------------------------------
def generate_intro_summary(session_id):
    """Generate an introductory summary from the uploaded PDF."""
    if not ensure_pdf_processed(session_id):
        return "PDF processing is not complete. Please try again shortly."
    prompt = (
        "Based solely on the research paper that was uploaded in this session, "
        "please provide a one sentence summary of what the paper is about. "
        "The summary should continue the following sentence with a brief summary "
        "about the uploaded research paper: "
        "'I'm here to assist you with understanding this week's reading, which is about...'"
    )
    return generate_response(prompt, session_id)

def summarizing_agent(action_type, session_id):
    """
    Agent to summarize the abstract or the full paper based on the action type.
    """
    cache = summary_abstract_cache if action_type == "summarize_abstract" else summary_full_cache
    if session_id in cache:
        return cache[session_id]
    if not ensure_pdf_processed(session_id):
        return "PDF processing is not complete. Please try again shortly."
    
    if action_type == "summarize_abstract":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, "
            "please provide a detailed summary focusing on the abstract. "
            "Include the main objectives and key points of the abstract."
        )
    elif action_type == "summarize_full":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, please provide a comprehensive and well-organized summary of the entire paper. "
            "Your summary should include the following sections with clear bullet points:\n\n"
            "1. **Title & Publication Details:** List the paper's title, authors, publication venue, and year.\n\n"
            "2. **Abstract & Problem Statement:** Summarize the abstract, highlighting the key challenges and the motivation behind the study.\n\n"
            "3. **Methodology:** Describe the research methods, experimental setup, and techniques used in the paper.\n\n"
            "4. **Key Findings & Results:** Outline the major results, findings, and any evaluations or experiments conducted.\n\n"
            "5. **Conclusions & Future Work:** Summarize the conclusions, implications of the study, and suggestions for future research.\n\n"
            "Please present your summary using clear headings and bullet points or numbered lists where appropriate."
        )
    else:
        return "Invalid summarization action."
    
    print(f"DEBUG: creating first cache")
    summary = generate_response(prompt, session_id)
    cache[session_id] = summary
    return summary

def answer_question(question, session_id):
    """Answer a student question based solely on the uploaded PDF content."""
    if not ensure_pdf_processed(session_id):
        return "PDF processing is not complete. Please try again shortly."
    prompt = (
        f"Based solely on the research paper that was uploaded in this session, "
        f"answer the following question:\n\n{question}\n\n"
        "Provide the answer using only the content of the uploaded PDF."
    )
    return generate_response(prompt, session_id)

# -----------------------------------------------------------------------------
# Flask Route: Query Handling
# -----------------------------------------------------------------------------
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json() or request.form
    print(f"DEBUG: Received request data: {data}")
    user = data.get("user_name", "Unknown")
    message = data.get("text", "").strip()
    
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    print(f"Message from {user}: {message}")
    session_id = get_session_id(data)
    
    # Initialize conversation state if new session
    if session_id not in conversation_history:
        conversation_history[session_id] = {
            "messages": [],
            "awaiting_ta_question": False
        }
        # Clear caches for this session.
        summary_abstract_cache.pop(session_id, None)
        summary_full_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)
    
    # Command to clear conversation and caches
    if message == "clear_history":
        conversation_history.pop(session_id, None)
        summary_abstract_cache.pop(session_id, None)
        summary_full_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)
        return jsonify(add_menu_button({
            "text": "Your conversation history and caches have been cleared.",
            "session_id": session_id
        }))
    
    # # -------------
    # # Process TA Confirmation Response
    # # -------------
    # if conversation_history[session_id].get("awaiting_ta_confirmation"):
    #     if message.lower() in ["yes", "y"]:
    #         conversation_history[session_id].pop("awaiting_ta_confirmation", None)
    #         ta_button_response = build_TA_button()
    #         ta_button_response["session_id"] = session_id
    #         return jsonify(ta_button_response)
    #     else:
    #         conversation_history[session_id].pop("awaiting_ta_confirmation", None)
    #         menu_response = build_menu_response()
    #         menu_response["session_id"] = session_id
    #         return jsonify({
    #             "text": "Please feel free to ask a question about the research paper, or explore the menu below for more actions.",
    #             "session_id": session_id,
    #             **menu_response
    #         })
    
    # # Handling TA question flow when the student is typing their TA question.
    # if (conversation_history[session_id].get("awaiting_ta_question")
    #     and message not in ["use_suggested_question", "keep_own", "confirm_send", "cancel_send"]):
        
    #     conversation_history[session_id]["student_question"] = message
    #     suggested_full, suggested_clean = generate_suggested_question(message)
    #     conversation_history[session_id]["suggested_question"] = suggested_full
    #     conversation_history[session_id]["final_question"] = suggested_clean
        
    #     return jsonify({
    #         "text": f"Here is a suggested question based on what you wrote:\n\n**{suggested_full}**\n\nWould you like to send this suggested question or keep your own?",
    #         "attachments": [
    #             {
    #                 "title": "Select an option:",
    #                 "actions": [
    #                     {
    #                         "type": "button",
    #                         "text": "Use Suggested Question",
    #                         "msg": "use_suggested_question",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     },
    #                     {
    #                         "type": "button",
    #                         "text": "Keep My Own",
    #                         "msg": "keep_own",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     }
    #                 ]
    #             }
    #         ],
    #         "session_id": session_id
    #     })
    
    # if message == "use_suggested_question":
    #     final_question = conversation_history[session_id].get("final_question", "No question available.")
    #     return jsonify({
    #         "text": f"You selected the suggested question:\n\n**{final_question}**\n\nDo you want to send this to the TA?",
    #         "attachments": [
    #             {
    #                 "title": "Confirm sending:",
    #                 "actions": [
    #                     {
    #                         "type": "button",
    #                         "text": "Confirm Send",
    #                         "msg": "confirm_send",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     },
    #                     {
    #                         "type": "button",
    #                         "text": "Cancel",
    #                         "msg": "cancel_send",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     }
    #                 ]
    #             }
    #         ],
    #         "session_id": session_id
    #     })
    
    # if message == "keep_own":
    #     final_question = conversation_history[session_id].get("student_question", "No question available.")
    #     conversation_history[session_id]["final_question"] = final_question
    #     return jsonify({
    #         "text": f"Do you want to send your question:\n\n**{final_question}**\n\nto the TA?",
    #         "attachments": [
    #             {
    #                 "title": "Confirm sending:",
    #                 "actions": [
    #                     {
    #                         "type": "button",
    #                         "text": "Confirm Send",
    #                         "msg": "confirm_send",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     },
    #                     {
    #                         "type": "button",
    #                         "text": "Cancel",
    #                         "msg": "cancel_send",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     }
    #                 ]
    #             }
    #         ],
    #         "session_id": session_id
    #     })
    
    # if message == "confirm_send":
    #     ta_name = conversation_history[session_id].get("awaiting_ta_question")
    #     if not ta_name:
    #         return jsonify({"text": "No TA selected.", "session_id": session_id})
    #     ta_username = "aya.ismail" if ta_name == "Aya" else "jiyoon.choi"
    #     final_question = conversation_history[session_id].get("final_question")
    #     if not final_question:
    #         return jsonify({"text": "No final question available. Please provide a question for your TA.", "session_id": session_id})
    #     send_direct_message_to_TA(final_question, user, ta_username)
    #     confirmation = f"Your question has been sent to TA {ta_name}!"
    #     conversation_history[session_id]["awaiting_ta_question"] = False
    #     conversation_history[session_id].pop("student_question", None)
    #     conversation_history[session_id].pop("suggested_question", None)
    #     conversation_history[session_id].pop("final_question", None)
    #     return jsonify(add_menu_button({
    #         "text": confirmation,
    #         "session_id": session_id
    #     }))
    
    # if message == "cancel_send":
    #     conversation_history[session_id]["awaiting_ta_question"] = False
    #     conversation_history[session_id].pop("student_question", None)
    #     conversation_history[session_id].pop("suggested_question", None)
    #     conversation_history[session_id].pop("final_question", None)
    #     return jsonify(add_menu_button({
    #         "text": "Your question was not sent. Let me know if you need anything else.",
    #         "session_id": session_id
    #     }))
#     if (conversation_history[session_id].get("awaiting_ta_question") 
#     and message not in ["use_suggested_question", "keep_own", "confirm_send", "cancel_send"]):
        
#         ta_name = conversation_history[session_id]["awaiting_ta_question"]
#         ta_username = "aya.ismail" if ta_name == "Aya" else "jiyoon.choi"
        
#         suggested_question_full, suggested_question_clean = generate_suggested_question(message)
#         print(f"DEBUG: Suggested question full: {suggested_question_full}")
#         print("END OF SUGGESTED QUESTION")
#         print(f"DEBUG: Suggested question clean: {suggested_question_clean}")
#         print("END OF SUGGESTED QUESTION")
#         conversation_history[session_id]["student_question"] = message
#         conversation_history[session_id]["suggested_question"] = suggested_question_full  # Full version for student
#         conversation_history[session_id]["final_question"] = suggested_question_clean  # Clean version for TA

#         return jsonify({
#         "text": f"Here is a suggested question based on what you wrote:\n\n"
#                 f"**{suggested_question_full}**\n\n"
#                 "Would you like to send this suggested question or keep your own?",
#         "attachments": [
#             {
#                 "title": "Select an option:",
#                 "actions": [
#                     {
#                         "type": "button",
#                         "text": "Use Suggested Question",
#                         "msg": "use_suggested_question",
#                         "msg_in_chat_window": True,
#                         "msg_processing_type": "sendMessage"
#                     },
#                     {
#                         "type": "button",
#                         "text": "Keep My Own",
#                         "msg": "keep_own",
#                         "msg_in_chat_window": True,
#                         "msg_processing_type": "sendMessage"
#                     }
#                 ]
#             }
#         ],
#         "session_id": session_id
#     })

#     # If the student chooses to use the suggested question
#     if message == "use_suggested_question":
       
#         final_question = conversation_history[session_id].get("final_question", "No question available.")
        
#         return jsonify({
#             "text": f"You selected the suggested question:\n\n**{final_question}**\n\n"
#                     "Do you want to send this to TA?",
#             "attachments": [
#                 {
#                     "title": "Confirm sending:",
#                     "actions": [
#                         {
#                             "type": "button",
#                             "text": "Confirm Send",
#                             "msg": "confirm_send",
#                             "msg_in_chat_window": True,
#                             "msg_processing_type": "sendMessage"
#                         },
#                         {
#                             "type": "button",
#                             "text": "Cancel",
#                             "msg": "cancel_send",
#                             "msg_in_chat_window": True,
#                             "msg_processing_type": "sendMessage"
#                         }
#                     ]
#                 }
#             ],
#             "session_id": session_id
#         })

# # If the student wants to write their own question again
#     if message == "keep_own":
#         final_question = conversation_history[session_id].get("student_question", "No question available.")
#         conversation_history[session_id]["final_question"] = final_question
#         return jsonify ({"text": f"Do you want to send your question :\n\n**{final_question}**\n\n"
#                 "to the TA?",
#         "attachments": [
#             {
#                 "title": "Confirm sending:",
#                 "actions": [
#                     {
#                         "type": "button",
#                         "text": "Confirm Send",
#                         "msg": "confirm_send",
#                         "msg_in_chat_window": True,
#                         "msg_processing_type": "sendMessage"
#                     },
#                     {
#                         "type": "button",
#                         "text": "Cancel",
#                         "msg": "cancel_send",
#                         "msg_in_chat_window": True,
#                         "msg_processing_type": "sendMessage"
#                     }
#                 ]
#             }
#         ],
#         "session_id": session_id
#     })

#     # If the student confirms sending the question
#     if message == "confirm_send":
#         ta_name = conversation_history[session_id]["awaiting_ta_question"]
#         ta_username = "aya.ismail" if ta_name == "Aya" else "jiyoon.choi"
#         final_question = conversation_history[session_id]["final_question"]

#         send_direct_message_to_TA(final_question, user, ta_username)

#         confirmation = f"Your question has been sent to TA {ta_name}!"
       
#         conversation_history[session_id]["awaiting_ta_question"] = False 
#         conversation_history[session_id].pop("student_question", None)
#         conversation_history[session_id].pop("suggested_question", None)
#         conversation_history[session_id].pop("final_question", None)
        
#         return jsonify(add_menu_button({"text": confirmation, "session_id": session_id}))


#     # If the student cancels sending
#     if message == "cancel_send":
#         conversation_history[session_id]["awaiting_ta_question"] = False
#         return jsonify(add_menu_button({"text": "Your question was not sent. Let me know if you need anything else.", "session_id": session_id}))
    
#     if message == "menu":
#         menu_response = build_menu_response()
#         menu_response["session_id"] = session_id
#         return jsonify(menu_response)
    
    if message == "ask_TA": 

        conversation_history[session_id]["awaiting_ta_question"] = False
        conversation_history[session_id].pop("student_question", None)
        conversation_history[session_id].pop("suggested_question", None)
        conversation_history[session_id].pop("final_question", None)

        ta_button_response = build_TA_button()
        ta_button_response["session_id"] = session_id
        return jsonify(ta_button_response)
        
#     if message in ["ask_TA_Aya", "ask_TA_Jiyoon"]:
#         ta_selected = "Aya" if message == "ask_TA_Aya" else "Jiyoon"
#         conversation_history[session_id]["awaiting_ta_question"] = ta_selected
#         return jsonify({
#             "text": f"Please type your question for TA {ta_selected}.",
#             "session_id": session_id
#         })

  # ----------------------------
    # TA Question Workflow
    # ----------------------------
    # If the student clicks "ask_TA_Aya" or "ask_TA_Jiyoon", initialize the question_flow
    if message in ["ask_TA_Aya", "ask_TA_Jiyoon"]:
        ta_selected = "Aya" if message == "ask_TA_Aya" else "Jiyoon"
        # Initialize question_flow state
        conversation_history[session_id]["question_flow"] = {
            "ta": ta_selected,
            "state": "awaiting_question",  # waiting for the student to type the question
            "raw_question": "",
            "suggested_question": ""
        }
        return jsonify({
            "text": f"Please type your question for TA {ta_selected}.",
            "session_id": session_id
        })
    
    # Check if we are in the middle of a TA question workflow
    if conversation_history[session_id].get("question_flow"):
        q_flow = conversation_history[session_id]["question_flow"]
        state = q_flow.get("state", "")
        
        # State 1: Awaiting the initial question from the student.
        if state == "awaiting_question":
            # Save the raw question
            q_flow["raw_question"] = message
            # Ask whether to refine or send the question
            q_flow["state"] = "awaiting_decision"
            return jsonify({
                "text": f"You typed: \"{message}\".\nWould you like to **refine** your question or **send** it as is?",
                "attachments": [
                    {
                        "actions": [
                            {
                                "type": "button",
                                "text": "Refine",
                                "msg": "refine",
                                "msg_in_chat_window": True,
                                "msg_processing_type": "sendMessage"
                            },
                            {
                                "type": "button",
                                "text": "Send",
                                "msg": "send",
                                "msg_in_chat_window": True,
                                "msg_processing_type": "sendMessage"
                            }
                        ]
                    }
                ],
                "session_id": session_id
            })
        
        # State 2: Awaiting decision from student on whether to refine or send
        if state == "awaiting_decision":
            if message.lower() == "send":
                q_flow["state"] = "awaiting_final_confirmation"
                return jsonify({
                    "text": f"Are you sure you want to send the following question to TA {q_flow['ta']}?\n\n\"{q_flow['raw_question']}\"",
                    "attachments": [
                        {
                            "actions": [
                                {
                                    "type": "button",
                                    "text": "Confirm",
                                    "msg": "confirm",
                                    "msg_in_chat_window": True,
                                    "msg_processing_type": "sendMessage"
                                },
                                {
                                    "type": "button",
                                    "text": "Cancel",
                                    "msg": "cancel",
                                    "msg_in_chat_window": True,
                                    "msg_processing_type": "sendMessage"
                                }
                            ]
                        }
                    ],
                    "session_id": session_id
                })
            elif message.lower() == "refine":
                # Generate an initial suggested version using the LLM
                suggested = generate_suggested_question(q_flow["raw_question"])[0]
                q_flow["suggested_question"] = suggested
                q_flow["state"] = "awaiting_refinement_decision"
                return jsonify({
                    "text": f"Here is a suggested version of your question:\n\n\"{suggested}\"\n\nDo you **approve** this version or would you like to **modify** it?",
                    "attachments": [
                        {
                            "actions": [
                                {
                                    "type": "button",
                                    "text": "Approve",
                                    "msg": "approve",
                                    "msg_in_chat_window": True,
                                    "msg_processing_type": "sendMessage"
                                },
                                {
                                    "type": "button",
                                    "text": "Modify",
                                    "msg": "modify",
                                    "msg_in_chat_window": True,
                                    "msg_processing_type": "sendMessage"
                                }, 
                                 {
                                    "type": "button",
                                    "text": "Manually Edit",
                                    "msg": "edit",
                                    "msg_in_chat_window": True,
                                    "msg_processing_type": "sendMessage"
                                }
                            ]
                        }
                    ],
                    "session_id": session_id
                })
            else:
                # Unrecognized response in this state
                return jsonify({
                    "text": "Please choose either **refine** or **send**.",
                    "session_id": session_id
                })
        
        # State 3: Awaiting refinement decision on the suggested version.
        if state == "awaiting_refinement_decision":
            if message.lower() == "approve":
                q_flow["state"] = "awaiting_final_confirmation"
                return jsonify({
                    "text": f"Do you want to send the following question to TA {q_flow['ta']}?\n\n\"{q_flow['suggested_question']}\"",
                    "attachments": [
                        {
                            "actions": [
                                {
                                    "type": "button",
                                    "text": "Confirm",
                                    "msg": "confirm",
                                    "msg_in_chat_window": True,
                                    "msg_processing_type": "sendMessage"
                                },
                                {
                                    "type": "button",
                                    "text": "Cancel",
                                    "msg": "cancel",
                                    "msg_in_chat_window": True,
                                    "msg_processing_type": "sendMessage"
                                }
                            ]
                        }
                    ],
                    "session_id": session_id
                })
            elif message.lower() == "modify":
                q_flow["state"] = "awaiting_feedback"
                return jsonify({
                    "text": "Please type your modifications or feedback for refining your question.",
                    "session_id": session_id
                })
            elif message.lower() == "edit":
                q_flow["state"] = "awaiting_feedback"
                return jsonify({
                    "text": "Please type your edits manually.",
                    "session_id": session_id
                })
            else:
                return jsonify({
                    "text": "Please choose either **approve** or **modify**.",
                    "session_id": session_id
                })
        
        # State 4: Awaiting student feedback to modify the suggested question.
        if state == "awaiting_feedback":
            # Combine the student’s feedback with the original raw question (or the previous suggested version)
            feedback = message
            # You can decide how to combine the original and feedback; here we simply pass both to the LLM.
            prompt = f"Original question: \"{q_flow['raw_question']}\"\nFeedback: \"{feedback}\"\nGenerate a refined version of the question."
            new_suggested = generate_response(prompt, session_id)
            q_flow["suggested_question"] = new_suggested
            q_flow["state"] = "awaiting_refinement_decision"
            return jsonify({
                "text": f"Here is an updated suggested version of your question:\n\n\"{new_suggested}\"\n\nDo you **approve** this version or would you like to **modify** it further?",
                "attachments": [
                    {
                        "actions": [
                            {
                                "type": "button",
                                "text": "Approve",
                                "msg": "approve",
                                "msg_in_chat_window": True,
                                "msg_processing_type": "sendMessage"
                            },
                            {
                                "type": "button",
                                "text": "Modify",
                                "msg": "modify",
                                "msg_in_chat_window": True,
                                "msg_processing_type": "sendMessage"
                            }
                        ]
                    }
                ],
                "session_id": session_id
            })
        # State for handling manual edit input:
        if state == "awaiting_manual_edit":
            # Directly store the manually edited question as the final question.
            q_flow["suggested_question"] = message
            q_flow["state"] = "awaiting_final_confirmation"
            return jsonify({
                "text": f"Your manually edited question is:\n\n\"{message}\"\n\nDo you want to send this question to TA {q_flow['ta']}?",
                "attachments": [
                    {
                        "actions": [
                            {"type": "button", "text": "Confirm", "msg": "confirm", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                            {"type": "button", "text": "Cancel", "msg": "cancel", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}
                        ]
                    }
                ],
                "session_id": session_id
            })
        # State 5: Awaiting final confirmation to send the question.
        if state == "awaiting_final_confirmation":
            if message.lower() == "confirm":
                # Send the question to the TA
                ta_username = "aya.ismail" if q_flow["ta"] == "Aya" else "jiyoon.choi"
                # Choose which version to send (if a suggestion exists, send that; otherwise, send the raw question)
                final_question = q_flow.get("suggested_question") or q_flow.get("raw_question")
                send_direct_message_to_TA(final_question, user, ta_username)
                # Clear the question_flow state
                conversation_history[session_id]["question_flow"] = None
                return jsonify(add_menu_button({
                    "text": f"Your question has been sent to TA {q_flow['ta']}!",
                    "session_id": session_id
                }))
            elif message.lower() == "cancel":
                # Cancel the process and clear question_flow
                conversation_history[session_id]["question_flow"] = None
                return jsonify(add_menu_button({
                    "text": "Your TA question process has been canceled. Let me know if you need anything else.",
                    "session_id": session_id
                }))
            else:
                return jsonify({
                    "text": "Please choose **confirm** to send or **cancel** to abort.",
                    "session_id": session_id
                })

    # ----------------------------
    # End of TA Question Workflow
    # ----------------------------
    
    if message in ["summarize_abstract", "summarize_full"]:
        summary = summarizing_agent(message, session_id)
        return jsonify(add_menu_button({"text": summary, "session_id": session_id}))
    
    # Process general chatbot queries
    conversation_history[session_id]["messages"].append(("user", message))
    classification = classify_query(message, session_id)
    print(f"DEBUG: User query classified as: {classification}")

    if classification == "greeting":
        def prepopulate_summaries(session_id):
            summarizing_agent("summarize_abstract", session_id)
            summarizing_agent("summarize_full", session_id)
            
        intro_summary = generate_greeting_response(
            "Based solely on the research paper that was uploaded in this session, please provide a one sentence summary of what the paper is about.",
            session_id
        )
        greeting_msg = (
            "Hello! I am the TA chatbot for CS-150: Generative AI for Social Impact. "
            + intro_summary +
            " Please feel free to ask a question about the research paper, or explore the menu below for more actions."
        )
        conversation_history[session_id]["messages"].append(("bot", greeting_msg))
        
        threading.Thread(target=prepopulate_summaries, args=(session_id,)).start()

        interactive_payload = show_menu(greeting_msg, session_id)
        return jsonify(interactive_payload)

    else:
        # Generate the primary answer
        answer = answer_question(message, session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))
        
        # Optionally generate a follow-up question based on conversation history
        follow_up = generate_follow_up(session_id)
        if follow_up:
            conversation_history[session_id]["messages"].append(("bot", follow_up))
            answer += f"\n\n{follow_up}"
        
        payload = {"text": answer, "session_id": session_id}
        return jsonify(add_menu_button(payload))

    return jsonify(add_menu_button({
        "text": "Sorry, I didn't understand that.",
        "session_id": session_id
    }))

# -----------------------------------------------------------------------------
# Error Handling and App Runner
# -----------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()