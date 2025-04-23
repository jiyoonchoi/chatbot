
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
TA_USER_LIST = ["aya.ismail", "jiyoon.choi"]

# Global caches (session-specific)
summary_abstract_cache = {}
processed_pdf = {}
pdf_ready = {}
conversation_history = {}
ta_msg_to_student_session = {}

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_session_id(data):
    """Generate a session id based on the user name."""
    user = data.get("user_name", "unknown_user").strip().lower()
    return f"session_{user}_twips_research"

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

def wait_for_pdf_readiness(session_id, max_attempts=20, delay=2):
    if pdf_ready.get(session_id):
        print(f"DEBUG: PDF already marked as ready for session {session_id}")
        return True
    for attempt in range(max_attempts):
        print(f"DEBUG: Checking PDF readiness (Attempt {attempt+1}/{max_attempts}) for session {session_id}")
        test_prompt = (
            "Based solely on the research paper that was uploaded in this session, "
            "what is the title of the paper?"
        )
        test_response = generate_response("", test_prompt, session_id)
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
def generate_response(system, prompt, session_id):
    """
    Generate a response based on the prompt using a fixed system prompt.
    """

    if not system:
        system = (
            "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
            "Your role is to guide students in developing their own understanding of the research paper. "
            "Rather than giving direct answers, encourage students to think critically. "
            "If the question is requesting for a summary of the paper, please provide a summary of the paper. Otherwise,"
            "Do NOT directly answer questions with specific numbers, names, or results. "
            "Instead, guide students toward where they can find the information in the paper (e.g., introduction, methods section, results, discussion). "
            "Do not summarize the entire answer; instead, promote thoughtful engagement with the content. "
            "Encourage them to reflect on why that information is relevant and how it connects to the paper's broader goals."
            "Your responses should be grounded solely in the research paper uploaded for this session. "
            "Please keep answers concise unless otherwise specified."
            "Bold with surrounding '**' to any follow up questions so they are easily visible to the user."
        )

    print(f"DEBUG: Sending prompt for session {session_id}: {prompt}")
    response = generate(
        model='4o-mini',
        system=system,
        query=prompt,
        temperature=0.0,
        lastk=5,
        session_id=session_id,
        rag_usage=True,
        rag_threshold=0.1,
        rag_k=5
    )
    if isinstance(response, dict):
        result = response.get('response', '').strip()
        rag_context = response.get('rag_context', None)
        print(f"DEBUG: Received response for session {session_id}: {result}")
        if rag_context:
            # print(f"DEBUG: RAG context for session {session_id}: {rag_context}")
            pass
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
        rag_threshold=0.1,
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

        classification_response = generate_response("", prompt, session_id).lower().strip()

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
    # If we're NOT awaiting a follow-up, we do original classification
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
    classification = generate_response("", prompt, session_id).lower().strip()

    if classification in ["greeting", "content_answerable", "human_ta_query"]:
        return classification

    # Default if unrecognized
    return "content_answerable"

def classify_difficulty_of_question(question, session_id):
    """
    Determine whether this 'content_answerable' question is purely factual
    (like a simple numeric or name lookup) or conceptual (requires
    reasoning, interpretation, or explanation).
    Return either 'factual' or 'conceptual'.
    """
    prompt = f"""
    The user asked: "{question}"
    Categorize the question strictly as either 'factual' or 'conceptual' based on the research paper context:
    - 'factual' means it requests a direct fact (a number, a name, a specific piece of data) with little to no interpretation.
    - 'conceptual' means it requires interpretation, explanation, or analysis beyond a single data point or name.
    Return only 'factual' or 'conceptual'.
    """
    classification_response = generate_response("", prompt, session_id).lower().strip()
    if "factual" in classification_response:
        return "factual"
    else:
        return "conceptual"
  
def generate_suggested_question(session_id, student_question, feedback=None):
    """
    Generate a rephrased and clearer version of the student's question.
    """
    print(f"DEBUG: session_id inside generate_suggested_question: {session_id}")
    ta_name = conversation_history[session_id]["question_flow"]["ta"]
    if feedback:
        prompt = (
            f"""Original question: \"{student_question}\"\n"""
            f"""Feedback: \"{feedback}\"\n"""
            f"""Based on the following session id {session_id} which has the name of the student and the paper 
            \n\n they are reading, generate a refined and more comprehensive version of the question that incorporates the feedback, "
            "including a reference to the paper and an intro with the name of the student (something like "Hi {ta_name}! This is [first name of student]). 
            Add any relevant context that the TA might need to understand the question.
            Add the name of the TA based on the selected TA {ta_name}.\n\n
            Do not add details that might seem not relevant or that make the question too long\n\n"""
        )
    else:
        prompt = (
            f"""Based on the following student question and the following session id {session_id} which has the name of the student and the paper 
            \n\n they are reading, generate a refined and more comprehensive version of the student question\n\n"""
            f"""Remember that the TA does not have context to the conversation, so be sure to include a reference to the paper and an intro with the name of the student (something like "Hi {ta_name}! This is [first name of student]).
            Add any relevant context that the TA might need to understand the question\n\n
            Do not add details that might seem not relevant or that make the question too long\n\n
            Add the name of the TA based on the selected TA {ta_name}.\n\n"""
            f""""Student question: \"{student_question}\"\n\n"""
            "Suggested improved question:"
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

def build_menu_response():
    """Return the full interactive menu."""
    return {
        "text": "Please feel free to ask a question about the research paper, or explore the menu below for more actions:",
        "attachments": [
            {
                "title": "Summarize:",
                "actions": [
                    {
                        "type": "button",
                        "text": "Quick Summary",
                        "msg": "summarize",
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

def forward_message_to_student(ta_response, session_id, student_session_id):
    # Build a payload to send to the student.
    
    msg_url = MSG_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": BOT_AUTH_TOKEN,
        "X-User-Id": BOT_USER_ID,
    }
    
    ta = "Aya" if session_id == "session_aya.ismail_twips_research" else "Jiyoon" 
    message_text = (
    f"Your TA {ta} says: '{ta_response}'\n\n"
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
        "'This week's paper discusses...'"
    )
    return generate_response("", prompt, session_id)

def summarizing_agent(action_type, session_id):
    """
    Agent to summarize the abstract or the full paper based on the action type.
    """
    cache = summary_abstract_cache
    if session_id in cache:
        return cache[session_id]
    if not ensure_pdf_processed(session_id):
        return "PDF processing is not complete. Please try again shortly."
    
    if action_type == "summarize":
        prompt = (
            "Based solely on the research paper that was uploaded in this session, "
            "please provide a detailed quick summary that is 3-4 sentences long, "
            "including the main objectives, key points, and conclusions of the paper. "
        )
    else:
        return "Invalid summarization action."
    
    print(f"DEBUG: creating first cache")
    summary = generate_response("", prompt, session_id)
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
        "and encourage them to think about why this information matters when applicable."
        "Limit the length of the response and keep language concise."
        "Bold key words in the response."
        "Provide the answer using only the content of the uploaded PDF."
    )
    return generate_response("", prompt, session_id)

def answer_factual_question(question, session_id):
    """
    Provide a direct factual answer from the PDF if possible,
    with minimal prompting to the student to look deeper.
    Specifically references 'TwIPS' so we don't mix up authors from references.
    """
    if not ensure_pdf_processed(session_id):
        return "PDF processing is not complete. Please try again shortly."

    # We explicitly mention the exact paper title so the LLM is much more likely
    # to pull from the correct chunk, rather than from references to other works.
    system_prompt = (
        "You are a TA chatbot. If the user is asking a purely factual question "
        "about the TwIPS paper (a large language model powered texting application "
        "for autistic users), then provide the exact fact from the TwIPS paper. "
        "Ignore any references or citations to other papers. "
        "If the TwIPS paper doesn't state the fact clearly, say so."
    )

    prompt = (
        "Based solely on the **TwIPS** paper titled: "
        "\"TwIPS: A Large Language Model Powered Texting Application to Simplify Conversational Nuances "
        "for Autistic Users,\" ignoring any other references or cited works:\n\n"
        f"Question: {question}\n\n"
        "If the paper does not clearly state it, say so."
        "Bold with surrounding '**' to any follow up questions so they are easily visible to the user."
    )

    return generate_response(system_prompt, prompt, session_id)

def answer_conceptual_question(question, session_id):
    """
    Provide an answer that references where in the paper
    the user might find the info, but encourages deeper reflection.
    """
    if not ensure_pdf_processed(session_id):
        return "PDF processing is not complete. Please try again shortly."
    
    prompt = (
        f"Based solely on the research paper that was uploaded in this session, "
        f"answer the following question:\n\n"
        f"{question}\n\n"
        "Do NOT provide a direct numeric or word-for-word result. "
        "Guide the student to the specific sections where they can find the information, "
        "and prompt them to think about its significance or implications."
    )
    return generate_response("", prompt, session_id)


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
            "question_flow": None,
            "awaiting_ta_response": False
        }
        # Clear caches for this session.
        summary_abstract_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)
    
    # Command to clear conversation and caches
    if message == "clear_history":
        conversation_history.pop(session_id, None)
        summary_abstract_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)
        return jsonify(add_menu_button({
            "text": "Your conversation history and caches have been cleared.",
            "session_id": session_id
        }))
    
    if message == "menu":
        menu_response = build_menu_response()
        menu_response["session_id"] = session_id
        return jsonify(menu_response)
    
     # ----------------------------
    # TA Question Workflow
    # ----------------------------

    if message == "ask_TA": 

        conversation_history[session_id]["awaiting_ta_question"] = False
        conversation_history[session_id].pop("student_question", None)
        conversation_history[session_id].pop("suggested_question", None)
        conversation_history[session_id].pop("final_question", None)

        ta_button_response = build_TA_button()
        ta_button_response["session_id"] = session_id
        return jsonify(ta_button_response)
    
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
        # If the user types the safeguard exit keyword "exit", cancel the TA flow.
        if message.lower() == "exit":
            conversation_history[session_id]["question_flow"] = None
            return jsonify(add_menu_button({
                "text": "Exiting TA query mode. How can I help you with the research paper?",
                "session_id": session_id
            }))
        
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
                                {"type": "button", "text": "Confirm", "msg": "confirm", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                                {"type": "button", "text": "Cancel", "msg": "cancel", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}
                            ]
                        }
                    ],
                    "session_id": session_id
                })
            elif message.lower() == "refine":
                # Default refine using LLM feedback
                suggested = generate_suggested_question(session_id, q_flow["raw_question"])[0]
                q_flow["suggested_question"] = suggested
                q_flow["state"] = "awaiting_refinement_decision"
                return jsonify({
                    "text": f"Here is a suggested version of your question:\n\n\"{suggested}\"\n\nDo you **approve** this version, want to **modify**, or do a **Manual Edit**?",
                    "attachments": [
                        {
                            "actions": [
                                {"type": "button", "text": "Approve", "msg": "approve", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                                {"type": "button", "text": "Modify", "msg": "modify", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                                {"type": "button", "text": "Manual Edit", "msg": "manual_edit", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}
                            ]
                        }
                    ],
                    "session_id": session_id
                })
            else:
                return jsonify({
                    "text": "Please choose either **refine** or **send**.",
                    "session_id": session_id
                })

        # Handling the decision in the refinement phase:
        if state == "awaiting_refinement_decision":
            print(f"DEBUGGING****: {session_id} - {message}")
            if message.lower() == "approve":
                q_flow["state"] = "awaiting_final_confirmation"
                return jsonify({
                    "text": f"Do you want to send the following question to TA {q_flow['ta']}?\n\n\"{q_flow['suggested_question']}\"",
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
            
            elif message.lower() == "modify":
                q_flow["state"] = "awaiting_feedback"
                return jsonify({
                    "text": "Please type your feedback for refining your question.",
                    "session_id": session_id
                })
            elif message.lower() == "manual_edit":
                q_flow["state"] = "awaiting_manual_edit"
                return jsonify({
                    "text": "Please type your manually edited question.",
                    "session_id": session_id
                })
            else:
                return jsonify({
                    "text": "Please choose **approve**, **Modify**, or **Manual Edit**.",
                    "session_id": session_id
                })

        # State 3: Awaiting feedback for LLM refinement
        if state == "awaiting_feedback":
            feedback = message
            # Combine the raw question and feedback to generate a refined version.
            base_question = q_flow.get("suggested_question", q_flow["raw_question"])
            new_suggested, new_suggested_clean = generate_suggested_question(session_id, base_question, feedback)
            q_flow["suggested_question"] = new_suggested_clean
            q_flow["state"] = "awaiting_refinement_decision"
            return jsonify({
                "text": f"Here is an updated suggested version of your question:\n\n\"{new_suggested_clean}\"\n\nDo you **approve**, want to **Modify**, or do a **Manual Edit**?",
                "attachments": [
                    {
                        "actions": [
                            {"type": "button", "text": "Approve", "msg": "approve", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                            {"type": "button", "text": "Modify", "msg": "modify", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                            {"type": "button", "text": "Manual Edit", "msg": "manual_edit", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}
                        ]
                    }
                ],
                "session_id": session_id
            })

        # State 4: Handling manual edit input
        if state == "awaiting_manual_edit":
            # Directly store the manually edited question as the suggested/final version.
            q_flow["suggested_question"] = message
            q_flow["state"] = "awaiting_refinement_decision"
            return jsonify({
                "text": f"Your manually edited question is:\n\n\"{message}\"\n\nDo you **approve**, want to **Modify**, or do another **Manual Edit**?",
                "attachments": [
                    {
                        "actions": [
                            {"type": "button", "text": "Approve", "msg": "approve", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                            {"type": "button", "text": "Modify", "msg": "modify", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"},
                            {"type": "button", "text": "Manual Edit", "msg": "manual_edit", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}
                        ]
                    }
                ],
                "session_id": session_id
            })

        # State 5: Awaiting final confirmation to send the question.
        if state == "awaiting_final_confirmation":
            if message.lower() == "confirm":
                ta_username = "aya.ismail" if q_flow["ta"] == "Aya" else "jiyoon.choi"
                final_question = q_flow.get("suggested_question") or q_flow.get("raw_question")
                send_direct_message_to_TA(final_question, user, ta_username)
                conversation_history[session_id]["question_flow"] = None
                return jsonify(add_menu_button({
                    "text": f"Your question has been sent to TA {q_flow['ta']}!",
                    "session_id": session_id
                }))
            elif message.lower() == "cancel":
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

    
    
    # Look up the student session ID using the mapping.

    if ta_msg_to_student_session:
        msg_id = next(reversed(ta_msg_to_student_session))
        student_username = ta_msg_to_student_session[msg_id]
        student_session_id = f"session_{student_username}_twips_research"

        if not student_session_id:
            return jsonify({"error": "No student session mapped for this message ID."}), 400
        
        if conversation_history[student_session_id].get("awaiting_ta_response"):
        # Assume this message is the TA's typed answer.
            conversation_history[student_session_id]["awaiting_ta_response"] = False
            conversation_history[student_session_id]["messages"].append(("TA", message))
            print(f"DEBUG: Received TA reply for session {student_session_id}: {message}")
            forward_message_to_student(message, session_id, student_session_id)
            response = f"Your response has been forwarded to student {student_username}."
            return jsonify({"text": response, "session_id": session_id})
    else:
        msg_id = None
   
    if message == "respond":
            # Process TA response prompt. For example, set flag and prompt for typed response.
            print(data.get("text"))
            conversation_history[student_session_id]["awaiting_ta_response"] = True
            print(f"DEBUG: Session {student_session_id} is now awaiting TA response from {user}")

            return jsonify({"text": "Please type your response to the student.", "session_id": student_session_id})
    
    
    # ----------------------------
    # End of TA Question Workflow
    # ----------------------------
    
    if conversation_history[session_id].get("awaiting_ta_confirmation"):
        if message.lower() in ["yes", "y"]:
            conversation_history[session_id].pop("awaiting_ta_confirmation", None)
            # Then start the TA question flow: show TA selection (or directly send if a default TA is used)
            ta_button_response = build_TA_button()  # reuse your TA button builder
            ta_button_response["session_id"] = session_id
            return jsonify(ta_button_response)
        else:
            conversation_history[session_id].pop("awaiting_ta_confirmation", None)
            # Fall back to answering the question using the document context
            answer = answer_question(message, session_id)
            conversation_history[session_id]["messages"].append(("bot", answer))
            # Optionally add a follow-up prompt here
            return jsonify(add_menu_button({"text": answer, "session_id": session_id}))

    if message in ["summarize"]:
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
        
        print(f"DEBUG: Greeting for {session_id}; processed={processed_pdf.get(session_id)}, ready={pdf_ready.get(session_id)}")
        ensure_pdf_processed(session_id)
        print(f"DEBUG: Greeting for {session_id}; processed={processed_pdf.get(session_id)}, ready={pdf_ready.get(session_id)}")
        intro_summary = generate_intro_summary(session_id)

       
        greeting_msg = (
        "**Hello! 👋 I am the TA chatbot for CS-150: Generative AI for Social Impact. 🤖**\n\n"
        "I'm here to help you *critically analyze ONLY this week's* research paper. "
        "I'll guide you to the key sections and ask thought-provoking questions—but I won't just hand you the answers. 🤫🧠\n\n"
        "You have two buttons to choose from:\n"
        "- **Quick Summary** - Get a concise 3-4 sentence overview of the paper's main objectives and findings.\n"
        "- **Ask TA** - Send your question to a human TA if you'd like extra help.\n\n"
        f"**{intro_summary}**\n\n"
        "If there's a question I can't fully answer, I'll prompt you to forward it to your TA. "
        "Please ask a question about the paper now or click one of the buttons below!")
        
        # Save and return the greeting without any follow-up questions, i.e. no food for thought.
        conversation_history[session_id]["messages"].append(("bot", greeting_msg))
        interactive_payload = show_menu(greeting_msg, session_id)
        return jsonify(interactive_payload)
    
    if classification == "content_answerable":
        difficulty = classify_difficulty_of_question(message, session_id)
        if difficulty == "factual":
            classification = "content_factual"
        else:
            classification = "content_conceptual"

    if classification == "content_factual":
        answer = answer_factual_question(message, session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))

        # Optionally generate a follow-up question (like you do for normal answers)
        universal_followup = generate_follow_up(session_id)
        if universal_followup:
            conversation_history[session_id]["awaiting_followup_response"] = True
            conversation_history[session_id]["messages"].append(("bot", universal_followup))
            answer_with_prompt = f"{answer}\n\n{universal_followup}"
        else:
            answer_with_prompt = answer

        return jsonify(add_menu_button({
            "text": answer_with_prompt,
            "session_id": session_id
        }))

    elif classification == "content_conceptual":
        answer = answer_conceptual_question(message, session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))

        universal_followup = generate_follow_up(session_id)
        if universal_followup:
            conversation_history[session_id]["awaiting_followup_response"] = True
            conversation_history[session_id]["messages"].append(("bot", universal_followup))
            answer_with_prompt = f"{answer}\n\n{universal_followup}"
        else:
            answer_with_prompt = answer

        return jsonify(add_menu_button({
            "text": answer_with_prompt,
            "session_id": session_id
        }))

    elif classification == "human_ta_query": 
        conversation_history[session_id]["awaiting_ta_confirmation"] = True
        payload = {
            "text": "It looks like your question may require human TA intervention. Would you like to ask your TA for more clarification?",
            "attachments": [
                {
                    "actions": [
                        {
                            "type": "button",
                            "text": "Yes, Ask TA",
                            "msg": "yes",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ], 
            session_id: session_id
        }
        return jsonify(payload)

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

# -----------------------------------------------------------------------------
# Error Handling and App Runner
# -----------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()