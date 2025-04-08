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
    # system_prompt = (
    #     "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
    #     "As a TA, you challenge students to practice critical thinking and question "
    #     "assumptions about the research paper. Your knowledge is based solely on the "
    #     "research paper that was uploaded in this session."
    # )
    system_prompt = (
        "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
        "Your role is to guide students in developing their own understanding of the research paper. "
        "Rather than giving direct answers, encourage students to think critically. "
        "Do NOT directly answer questions with specific numbers, names, or results. "
        "Instead, guide students toward where they can find the information in the paper (e.g., introduction, methods section, results, discussion). "
        "Do not summarize the entire answer; instead, promote thoughtful engagement with the content. "
        "Encourage them to reflect on why that information is relevant and how it connects to the paper's broader goals."
        "Your responses should be grounded solely in the research paper uploaded for this session. "
        "Please keep answers concise unless otherwise specified."
    
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
    # system_prompt = (
    #     "As a TA chatbot for CS-150: Generative AI for Social Impact, "
    #     "your job is to help the student think critically about the research paper. "
    #     "Provide a concise answer based solely on the research paper that was uploaded in this session. "
    #     "Do not include any follow-up questions in your response."
    # )
    system_prompt = (
    "As a TA chatbot for CS-150: Generative AI for Social Impact, your goal is to support students in understanding the research paper through critical thinking. "
    "Provide a brief, general overview to orient them, but avoid summarizing too much. "
    "Encourage them to explore specific sections of the paper for more detail, and avoid giving full answers. "
    "Do not include follow-up questions."
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
    Dynamically generate a follow-up question that:
      - references the conversation context
      - invites the user to specify what they want to explore further
    """
    messages = conversation_history.get(session_id, {}).get("messages", [])
    if not messages:
        return ""

    # Build a textual summary of the conversation so far.
    context = "\n".join([f"{speaker}: {msg}" for speaker, msg in messages])

    prompt = (
        "Based on the conversation so far, craft a single follow-up prompt that invites the user "
        "to elaborate or clarify what they want to explore further in the paper. If the conversation "
        "does not logically require a follow-up, return an empty string.\n\n"
        "Conversation:\n" + context
    )
    # system_prompt = (
    #     "You are a TA chatbot that generates follow-up questions based on context. "
    #     "Encourage the student to clarify or broaden their exploration of the research paper, "
    #     "mentioning aspects like methodology, findings, or next steps if relevant."
    # )
    system_prompt = (
        "You are a TA chatbot that encourages students to think more deeply about the research paper. "
        "Craft a follow-up question that helps the student examine aspects such as the paper’s assumptions, evidence, methodology, or implications. "
        "Avoid answering for them—instead, prompt them to explore further."
    )

    response = generate(
        model='4o-mini',
        system=system_prompt,
        query=prompt,
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

    # Optionally, you could decide to return "" if the LLM suggests a question not relevant or empty.
    return result

def classify_query(message, session_id):
    """
    Classify the user's message into categories. Normally, we have:
      - greeting
      - content_answerable
      - human_TA_query

    However, if the bot is currently awaiting a follow-up response,
    we expand our categories to also include:
      - followup_decline
      - followup_continue
      - new_topic

    We do all of this in a single classification call, using
    context from the conversation_history to help the LLM decide.
    """

    # Get the conversation so far
    messages = conversation_history.get(session_id, {}).get("messages", [])
    conversation_text = "\n".join([f"{speaker}: {msg}" for speaker, msg in messages])

    # Check if we're expecting a follow-up response
    if conversation_history[session_id].get("awaiting_followup_response"):
        # If so, we instruct the LLM to classify among an expanded set of labels.
        prompt = f"""
                    We are in the middle of a conversation about a research paper, and I (the TA chatbot)
                    recently asked a follow-up question. The conversation so far is:
                    ---
                    {conversation_text}
                    ---
                    The user just said: "{message}"

                    Please classify the user's message into EXACTLY ONE of these categories:

                    1) "followup_decline": The user wants to skip or end this follow-up topic.
                    2) "followup_continue": The user is still on the same follow-up topic, clarifying or asking more.
                    3) "new_topic": The user is shifting to a completely different topic.
                    4) "greeting": The user is just greeting ("hello", "hi", "hey", etc.).
                    5) "content_answerable": The user's question can be answered by the PDF content.
                    6) "human_ta_query": The user is asking about deadlines, scheduling, or other info not in the PDF.

                    Return only the label. No extra text.
                """

        classification_response = generate_response(prompt, session_id).lower().strip()

        # Match to known labels or default to content_answerable if unrecognized
        if "followup_decline" in classification_response:
            return "followup_decline"
        elif "followup_continue" in classification_response:
            return "followup_continue"
        elif "new_topic" in classification_response:
            return "new_topic"
        elif "greeting" in classification_response:
            return "greeting"
        elif "content_answerable" in classification_response:
            return "content_answerable"
        elif "human_ta_query" in classification_response:
            return "human_ta_query"
        else:
            return "content_answerable"

    # ----------------------------------------------------------------------
    # If we're NOT awaiting a follow-up, we do your original classification
    # ----------------------------------------------------------------------
    prompt = (
        "Classify the following query into one of these categories: 'greeting', "
        "'content_answerable', or 'human_TA_query'. "
        "If the query is a greeting, answer with 'greeting'. "
        "If the query can be answered solely based on the uploaded research paper, "
        "answer with 'content_answerable'. "
        "If the query involves deadlines, scheduling, or requires additional human judgment, "
        "answer with 'human_TA_query'.\n\n"
        f"Query: \"{message}\""
    )
    classification = generate_response(prompt, session_id).lower().strip()

    if classification in ["greeting", "content_answerable", "human_ta_query"]:
        return classification

    # Default if unrecognized
    return "content_answerable"
  
def generate_suggested_question(student_question):
    """
    Generate a rephrased and clearer version of the student's question.
    """
    prompt = (
        f"Based on the following student question, generate a clearer and more concise version that does not reference any PDF content:\n\n"
        f"Student question: \"{student_question}\"\n\n"
        "Suggested improved question:"
    )
    response = generate(
         model='4o-mini',
         system=(
             "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
             "Rephrase the student's question to be clearer and more concise without referring to any external context."
         ),
         query=prompt,
         temperature=0.0,
         lastk=5,
         session_id="suggestion_session",
         rag_usage=False,  # RAG disabled
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
# def build_interactive_response(response_text, session_id):
#     """Build interactive response payload with summary and TA options."""
#     return {
#         "text": response_text,
#         "attachments": [
#             {
#                 "title": "Would you like a summary?",
#                 "text": "Select an option:",
#                 "actions": [
#                     {
#                         "type": "button",
#                         "text": "Quick Summary",
#                         "msg": "quick_summary",
#                         "msg_in_chat_window": True,
#                         "msg_processing_type": "sendMessage"
#                     }
#                 ]
#             },
#             {
#                 "title": "Ask a TA",
#                 "text": "Do you have a question for your TA?",
#                 "actions": [
#                     {
#                         "type": "button",
#                         "text": "Ask a TA",
#                         "msg": "ask_TA",
#                         "msg_in_chat_window": True,
#                         "msg_processing_type": "sendMessage"
#                     }
#                 ]
#             }
#         ]
#     }

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
                        "text": "Quick Summary",
                        "msg": "quick_summary",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
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
    # cache = summary_abstract_cache if action_type == "quick_summary" else summary_full_cache
    cache = summary_abstract_cache
    if session_id in cache:
        return cache[session_id]
    if not ensure_pdf_processed(session_id):
        return "PDF processing is not complete. Please try again shortly."
    
    if action_type == "quick_summary":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, "
            "please provide a detailed summary focusing on the abstract. "
            "Include the main objectives and key points of the abstract."
        )
    # elif action_type == "summarize_full":
    #     prompt = (
    #         "Based solely on the research paper that was uploaded in this session, please provide a comprehensive and well-organized summary of the entire paper. "
    #         "Your summary should include the following sections with clear bullet points:\n\n"
    #         "1. **Title & Publication Details:** List the paper's title, authors, publication venue, and year.\n\n"
    #         "2. **Abstract & Problem Statement:** Summarize the abstract, highlighting the key challenges and the motivation behind the study.\n\n"
    #         "3. **Methodology:** Describe the research methods, experimental setup, and techniques used in the paper.\n\n"
    #         "4. **Key Findings & Results:** Outline the major results, findings, and any evaluations or experiments conducted.\n\n"
    #         "5. **Conclusions & Future Work:** Summarize the conclusions, implications of the study, and suggestions for future research.\n\n"
    #         "Please present your summary using clear headings and bullet points or numbered lists where appropriate."
    #     )
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
        f"answer the following question:\n\n{question}\n, do **not** provide a direct answer. "
        "Instead, guide the student to the section where they can find this answer (such as the methods or results), "
        "and encourage them to think about why this information matters."
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
            "awaiting_ta_question": False,
            "awaiting_ta_confirmation": False  # flag for TA confirmation
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
    
    # -------------
    # Process TA Confirmation Response
    # -------------
    if conversation_history[session_id].get("awaiting_ta_confirmation"):
        if message.lower() in ["yes", "y"]:
            conversation_history[session_id].pop("awaiting_ta_confirmation", None)
            ta_button_response = build_TA_button()
            ta_button_response["session_id"] = session_id
            return jsonify(ta_button_response)
        else:
            conversation_history[session_id].pop("awaiting_ta_confirmation", None)
            menu_response = build_menu_response()
            menu_response["session_id"] = session_id
            return jsonify({
                "text": "Please feel free to ask a question about the research paper, or explore the menu below for more actions.",
                "session_id": session_id,
                **menu_response
            })
    
    # Handling TA question flow when the student is typing their TA question.
    if (conversation_history[session_id].get("awaiting_ta_question")
        and message not in ["use_suggested_question", "keep_own", "confirm_send", "cancel_send"]):
        
        conversation_history[session_id]["student_question"] = message
        suggested_full, suggested_clean = generate_suggested_question(message)
        conversation_history[session_id]["suggested_question"] = suggested_full
        conversation_history[session_id]["final_question"] = suggested_clean
        
        return jsonify({
            "text": f"Here is a suggested question based on what you wrote:\n\n**{suggested_full}**\n\nWould you like to send this suggested question or keep your own?",
            "attachments": [
                {
                    "title": "Select an option:",
                    "actions": [
                        {
                            "type": "button",
                            "text": "Use Suggested Question",
                            "msg": "use_suggested_question",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "Keep My Own",
                            "msg": "keep_own",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ],
            "session_id": session_id
        })
    
    if message == "use_suggested_question":
        final_question = conversation_history[session_id].get("final_question", "No question available.")
        return jsonify({
            "text": f"You selected the suggested question:\n\n**{final_question}**\n\nDo you want to send this to the TA?",
            "attachments": [
                {
                    "title": "Confirm sending:",
                    "actions": [
                        {
                            "type": "button",
                            "text": "Confirm Send",
                            "msg": "confirm_send",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "Cancel",
                            "msg": "cancel_send",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ],
            "session_id": session_id
        })
    
    if message == "keep_own":
        final_question = conversation_history[session_id].get("student_question", "No question available.")
        conversation_history[session_id]["final_question"] = final_question
        return jsonify({
            "text": f"Do you want to send your question:\n\n**{final_question}**\n\nto the TA?",
            "attachments": [
                {
                    "title": "Confirm sending:",
                    "actions": [
                        {
                            "type": "button",
                            "text": "Confirm Send",
                            "msg": "confirm_send",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "Cancel",
                            "msg": "cancel_send",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ],
            "session_id": session_id
        })
    
    if message == "confirm_send":
        ta_name = conversation_history[session_id].get("awaiting_ta_question")
        if not ta_name:
            return jsonify({"text": "No TA selected.", "session_id": session_id})
        ta_username = "aya.ismail" if ta_name == "Aya" else "jiyoon.choi"
        final_question = conversation_history[session_id].get("final_question")
        if not final_question:
            return jsonify({"text": "No final question available. Please provide a question for your TA.", "session_id": session_id})
        send_direct_message_to_TA(final_question, user, ta_username)
        confirmation = f"Your question has been sent to TA {ta_name}!"
        conversation_history[session_id]["awaiting_ta_question"] = False
        conversation_history[session_id].pop("student_question", None)
        conversation_history[session_id].pop("suggested_question", None)
        conversation_history[session_id].pop("final_question", None)
        return jsonify(add_menu_button({
            "text": confirmation,
            "session_id": session_id
        }))
    
    if message == "cancel_send":
        conversation_history[session_id]["awaiting_ta_question"] = False
        conversation_history[session_id].pop("student_question", None)
        conversation_history[session_id].pop("suggested_question", None)
        conversation_history[session_id].pop("final_question", None)
        return jsonify(add_menu_button({
            "text": "Your question was not sent. Let me know if you need anything else.",
            "session_id": session_id
        }))
    
    if message == "menu":
        menu_response = build_menu_response()
        menu_response["session_id"] = session_id
        return jsonify(menu_response)
    
    if message == "ask_TA":
        # When the student clicks "Ask TA" from the clarification prompt,
        # set a flag so that a follow-up "yes/no" response is expected.
        conversation_history[session_id]["awaiting_ta_confirmation"] = True
        return jsonify({
            "text": "Would you like to ask your TA for further clarification? Please reply with 'yes' or 'no'.",
            "session_id": session_id,
            "attachments": [
                {
                    "title": "Confirm TA assistance:",
                    "actions": [
                        {
                            "type": "button",
                            "text": "Yes",
                            "msg": "yes",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "No",
                            "msg": "no",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ]
        })
    
    if message in ["ask_TA_Aya", "ask_TA_Jiyoon", "ask_TA_Amanda"]:
        ta_mapping = {
            "ask_TA_Aya": "Aya",
            "ask_TA_Jiyoon": "Jiyoon",
            "ask_TA_Amanda": "Amanda"
        }
        ta_selected = ta_mapping.get(message)
        conversation_history[session_id]["awaiting_ta_question"] = ta_selected
        return jsonify({
            "text": f"Please type your question for TA {ta_selected}.",
            "session_id": session_id
        })
    
    if message in ["quick_summary"]:
        summary = summarizing_agent(message, session_id)
        return jsonify(add_menu_button({"text": summary, "session_id": session_id}))
    
    # Process general chatbot queries
    conversation_history[session_id]["messages"].append(("user", message))
    classification = classify_query(message, session_id)
    print(f"DEBUG: User query classified as: {classification}")

    # -------------------------------------------
    # 1) Handle the new follow-up categories first
    # -------------------------------------------
    if classification == "followup_decline":
        # The user is declining further discussion of the follow-up topic
        conversation_history[session_id]["awaiting_followup_response"] = False
        decline_text = "No problem! Let me know if you have any other questions."
        conversation_history[session_id]["messages"].append(("bot", decline_text))
        return jsonify(add_menu_button({
            "text": decline_text,
            "session_id": session_id
        }))

    elif classification == "followup_continue":
        # The user wants to keep talking about the same topic.
        # Instead of just prompting "Could you tell me more specifically what you'd like to clarify?"
        # we can do something more productive.
        
        # 1) Provide an actual content-based answer, using answer_question
        # (assuming the user is repeating their interest in "features and functionalities")
        answer = answer_question(message, session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))
        
        # 2) Optionally generate another follow-up or ask if they'd like more details
        # but let's keep it simple so we don't loop forever
        continue_text = "Let me know if you'd like more details or if you have a different question."
        conversation_history[session_id]["messages"].append(("bot", continue_text))
        
        # 3) IMPORTANT: Mark that we've handled the follow-up, so we don't loop
        conversation_history[session_id]["awaiting_followup_response"] = False
        
        return jsonify(add_menu_button({
            "text": f"{answer}\n\n{continue_text}",
            "session_id": session_id
        }))

    elif classification == "new_topic":
        # The user is pivoting away from the old follow-up
        conversation_history[session_id]["awaiting_followup_response"] = False
        # We now proceed to check if the new message also requires an answer from the PDF
        # For simplicity, let's let the normal flow continue below
        pass

    # ---------------------------------------
    # 2) Handle the usual categories (if not a follow-up)
    # ---------------------------------------
    if classification == "greeting":
        def prepopulate_summaries(session_id):
            summarizing_agent("quick_summary", session_id)
            # summarizing_agent("summarize_full", session_id)
            
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

    # elif classification == "human_ta_query":
    #     # Immediately trigger TA flow: prompt the student to confirm if they'd like human assistance.
    #     answer = "It appears that this question might require additional clarification. Would you like to ask your TA for more details?"
    #     conversation_history[session_id]["awaiting_ta_confirmation"] = True
    #     payload = {
    #         "text": answer,
    #         "session_id": session_id,
    #         "attachments": [
    #             {
    #                 "title": "Ask your TA for clarification:",
    #                 "actions": [
    #                     {
    #                         "type": "button",
    #                         "text": "Yes",
    #                         "msg": "yes",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     },
    #                     {
    #                         "type": "button",
    #                         "text": "No",
    #                         "msg": "no",
    #                         "msg_in_chat_window": True,
    #                         "msg_processing_type": "sendMessage"
    #                     }
    #                 ]
    #             }
    #         ]
    #     }
    #     return jsonify(payload)

    # elif classification == "content_answerable":
    else:
        # Generate the primary answer
        answer = answer_question(message, session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))
        
        # Dynamically generate an open-ended follow-up based on the conversation.
        universal_followup = generate_follow_up(session_id)
        if universal_followup:
            conversation_history[session_id]["awaiting_followup_response"] = True
            conversation_history[session_id]["messages"].append(("bot", universal_followup))
            answer_with_prompt = f"{answer}\n\n{universal_followup}"
        else:
            answer_with_prompt = answer
    
        payload = {
            "text": answer_with_prompt,
            "session_id": session_id
        }
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