import os
import uuid
import time
import requests
import re
from flask import Flask, request, jsonify
from llmproxy import generate, pdf_upload
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use BASE_DIR to reliably set the PDF path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, 'twips_paper.pdf')
print("DEBUG: PDF path set to:", PDF_PATH)

app = Flask(__name__)

# Global caches (session-specific)
summary_cache = {}
processed_pdf = {}
pdf_ready = {}
conversation_history = {}

# Rocket.Chat Bot Credentials & URL
ROCKET_CHAT_URL = "https://chat.genaiconnect.net"
BOT_USER_ID = os.getenv("botUserId")
BOT_AUTH_TOKEN = os.getenv("botToken")
TA_USERNAME = os.getenv("taUserName")
MSG_ENDPOINT = os.getenv("msgEndPoint")

# def send_typing_indicator(room_id):
#     headers = {
#         "X-Auth-Token": BOT_AUTH_TOKEN,
#         "X-User-Id": BOT_USER_ID,
#         "Content-type": "application/json",
#     }
#     payload = {"roomId": room_id}
#     url = f"{ROCKET_CHAT_URL}/api/v1/chat.sendTyping"
#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         print(f"DEBUG: Typing indicator response: {response.json()}")
#     except Exception as e:
#         print(f"DEBUG: Error sending typing indicator: {e}")

def get_session_id(data):
    user = data.get("user_name", "unknown_user").strip().lower()
    return f"session_{user}"


def upload_pdf_if_needed(pdf_path, session_id):
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
    Polls until the PDF is indexed and its content is integrated.
    We test by asking for the title and checking that a valid title is returned.
    """
    if pdf_ready.get(session_id):
        print(f"DEBUG: PDF already marked as ready for session {session_id}")
        return True

    for attempt in range(max_attempts):
        print(f"DEBUG: Checking PDF readiness (Attempt {attempt+1}/{max_attempts}) for session {session_id}")
        test_prompt = ("Based solely on the research paper that was uploaded in this session, "
                       "what is the title of the paper?")
        test_response = generate_response(test_prompt, session_id)
        print(f"DEBUG: Test response (Attempt {attempt+1}): {test_response}")
        # Check if the fallback phrases are absent
        if ("unable to access" not in test_response.lower() and 
            "i don't have the capability" not in test_response.lower()):
            pdf_ready[session_id] = True
            print(f"DEBUG: PDF readiness confirmed on attempt {attempt+1} for session {session_id}")
            return True
        print(f"DEBUG: PDF not ready on attempt {attempt+1} for session {session_id}")
        time.sleep(delay)
    print(f"DEBUG: PDF failed to be indexed within timeout for session {session_id}")
    return False
    

def generate_response(prompt, session_id):
    personality = conversation_history.get(session_id, {}).get("personality", "default")
    
    # Choose the system prompt based on the personality.
    if personality == "critical":
        system_prompt = (
            "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
            "As a TA, you challenge students to think deeply and question assumptions. "
            "Provide thorough analysis and constructive criticism in your responses."
            "After answering the student's query, ask a follow-up question that prompts them to reflect further or explore the topic more deeply."
        )
    elif personality == "empathetic":
        system_prompt = (
            "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
            "As a TA, you are caring and supportive, offering kind explanations and understanding of complex topics."
            "After answering the student's query, ask a follow-up question that prompts them to reflect further or explore the topic more deeply."
        )
    elif personality == "straightforward":
        system_prompt = (
            "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
            "As a TA, you provide clear and concise answers without unnecessary details."
            "After answering the student's query, ask a follow-up question that prompts them to reflect further or explore the topic more deeply."
        )
    else:
        # Default personality
        system_prompt = (
            "You are a TA chatbot for CS-150: Generative AI for Social Impact. "
            "As a TA, you want to encourage students to think critically and learn more by providing insightful answers. "
            "After answering the student's query, ask a follow-up question that prompts them to reflect further or explore the topic more deeply."
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

def generate_intro_summary(session_id):
    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return "Could not upload PDF."
    if not wait_for_pdf_readiness(session_id):
        return "PDF processing is not complete. Please try again shortly."
    prompt = ("Based solely on the research paper that was uploaded in this session, "
              "please provide a one sentence summary of what the paper is about.")
    return generate_response(prompt, session_id)

def summarizing_agent(action_type, session_id):
    if session_id not in summary_cache:
        summary_cache[session_id] = {}

    if summary_cache[session_id].get(action_type):
        return summary_cache[session_id][action_type]

    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return "Could not upload PDF."

    if not wait_for_pdf_readiness(session_id):
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
            "Your summary should include the following sections:\n\n"
            "1. **Title & Publication Details:** List the paper's title, authors, publication venue, and year.\n\n"
            "2. **Abstract & Problem Statement:** Summarize the abstract, highlighting the key challenges and the motivation behind the study.\n\n"
            "3. **Methodology:** Describe the research methods, experimental setup, and techniques used in the paper.\n\n"
            "4. **Key Findings & Results:** Outline the major results, findings, and any evaluations or experiments conducted.\n\n"
            "5. **Conclusions & Future Work:** Summarize the conclusions, implications of the study, and suggestions for future research.\n\n"
            "Please present your summary using clear headings and bullet points or numbered lists where appropriate."
        )
    else:
        return "Invalid summarization action."

    summary = generate_response(prompt, session_id)
    summary_cache[session_id][action_type] = summary
    return summary

def answer_question(question, session_id):
    if not upload_pdf_if_needed(PDF_PATH, session_id):
        return "Could not upload PDF."
    
    if not wait_for_pdf_readiness(session_id):
        return "PDF processing is not complete. Please try again shortly."

    prompt = (
        f"Based solely on the research paper that was uploaded in this session, "
        f"answer the following question:\n\n{question}\n\n"
        "Provide the answer using only the content of the uploaded PDF."
    )
    return generate_response(prompt, session_id)

def send_direct_message_to_TA(question, session_id, ta_username):
     
    msg_url = MSG_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": BOT_AUTH_TOKEN,
        "X-User-Id": BOT_USER_ID,
    }
    
    student = session_id
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
    prompt = (
        "Determine if the following message is a greeting or a query about the research paper. "
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
            },
            {
                "title": "Manage Conversation",
                "text": "Choose an action:",
                "actions": [
                    {
                        "type": "button",
                        "text": "Clear Conversation History",
                        "msg": "clear_history",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }
        ]
    }

def add_menu_button(response_payload):
    # Replace any attachments with just the Menu button
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

def build_greeting_response(response_text, session_id):
    return {
        "text": response_text,
        "session_id": session_id,
        "attachments": [
            {
                "actions": [
                    {
                        "type": "button",
                        "text": "Choose Personality",
                        "msg": "choose_personality",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
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

def build_personality_response():
    return {
        "text": "Select a personality:",
        "attachments": [
            {
                "title": "Select an option:",
                "actions": [
                    {
                        "type": "button",
                        "text": "Default",
                        "msg": "personality_default",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Straightforward",
                        "msg": "personality_straightforward",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Critical",
                        "msg": "personality_critical",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Empathetic",
                        "msg": "personality_empathetic",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Other",
                        "msg": "personality_other",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                    ]}]
    }

def add_personality_button(response_payload):
    # Replace any attachments with just the Personality button
    response_payload["attachments"] = [
        {
            "actions": [
                {
                    "type": "button",
                    "text": "Choose Personality",
                    "msg": "choose_personality",
                    "msg_in_chat_window": True,
                    "msg_processing_type": "sendMessage"
                }
            ]
        }
    ]
    return response_payload

  
# def send_typing_indicator(room_id):
#     headers = {
#         "X-Auth-Token": BOT_AUTH_TOKEN,
#         "X-User-Id": BOT_USER_ID,
#         "Content-type": "application/json",
#     }
#     payload = {"channel_id": room_id}
#     url = f"{ROCKET_CHAT_URL}/api/v1/chat.sendTyping"
#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         print(f"DEBUG: Typing indicator response: {response.json()}")
#     except Exception as e:
#         print(f"DEBUG: Error sending typing indicator: {e}")

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
                    }
                ]
            }
        ]
    }    

def generate_suggested_question(student_question):
    prompt = (
        f"Based on the following student question, generate a clearer and more refined version:\n\n"
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

@app.route('/query', methods=['POST'])

def query():

    # Debugging: Print request headers and raw data
    print("DEBUG Headers:", request.headers)
    print("DEBUG Raw Data:", request.data)  # This helps debug if JSON is malformed

    data = request.get_json() or request.form
    print(f"DEBUG: Received request data: {data}")
    
    user = data.get("user_name", "Unknown")
    message = data.get("text", "").strip()
    
    # room_id = data.get("channel_id")
    
    # if data.get("text") == "debug_data":
    #     # This sends the entire request payload back to the chat.
    #     return jsonify({"text": f"DEBUG: Received data: {data}"})
    
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    
    print(f"Message from {user}: {message}")
    session_id = get_session_id(data)
    
    # if room_id:
    #     print(f"\nDEBUG: loading indicator shown\n")
    #     send_typing_indicator(room_id)
    
    # Initialize conversation if not present.
    if session_id not in conversation_history:
        conversation_history[session_id] = {"messages": [], "awaiting_ta_question": False}
    
    # Clear history command: clear conversation and caches.
    if message == "clear_history":
        conversation_history.pop(session_id, None)
        summary_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)
        return jsonify(add_menu_button({
            "text": "Your conversation history and caches have been cleared.",
            "session_id": session_id
        }))

    # If awaiting a TA question, forward it.
    if (conversation_history[session_id].get("awaiting_ta_question") 
    and message not in ["use_suggested_question", "keep_own", "confirm_send", "cancel_send"]):
        
        ta_name = conversation_history[session_id]["awaiting_ta_question"]
        ta_username = "aya.ismail" if ta_name == "Aya" else "jiyoon.choi"
        
        suggested_question_full, suggested_question_clean = generate_suggested_question(message)
        print(f"DEBUG: Suggested question full: {suggested_question_full}")
        print("END OF SUGGESTED QUESTION")
        print(f"DEBUG: Suggested question clean: {suggested_question_clean}")
        print("END OF SUGGESTED QUESTION")
        conversation_history[session_id]["student_question"] = message
        # conversation_history[session_id]["suggested_question"] = suggested_question
        conversation_history[session_id]["suggested_question"] = suggested_question_full  # Full version for student
        conversation_history[session_id]["final_question"] = suggested_question_clean  # Clean version for TA

        return jsonify({
        "text": f"Here is a suggested question based on what you wrote:\n\n"
                f"**{suggested_question_full}**\n\n"
                "Would you like to send this suggested question or keep your own?",
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

    # If the student chooses to use the suggested question
    if message == "use_suggested_question":
        # final_question = conversation_history[session_id]["suggested_question"]
        final_question = conversation_history[session_id].get("final_question", "No question available.")
        # conversation_history[session_id]["final_question"] = final_question

        return jsonify({
            "text": f"You selected the suggested question:\n\n**{final_question}**\n\n"
                    "Do you want to send this to TA?",
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

# If the student wants to write their own question again
    if message == "keep_own":
        final_question = conversation_history[session_id].get("student_question", "No question available.")
        conversation_history[session_id]["final_question"] = final_question
        return jsonify ({"text": f"Do you want to send your question :\n\n**{final_question}**\n\n"
                "to the TA?",
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

    # If the student confirms sending the question
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
        
        return jsonify(add_menu_button({"text": confirmation, "session_id": session_id}))


    # If the student cancels sending
    if message == "cancel_send":
        conversation_history[session_id].pop("awaiting_ta_question", None)
        conversation_history[session_id].pop("student_question", None)
        conversation_history[session_id].pop("suggested_question", None)
        conversation_history[session_id].pop("final_question", None)
        return jsonify(add_menu_button({
            "text": "Your question was not sent. Let me know if you need anything else.",
            "session_id": session_id
        }))

        # send_direct_message_to_TA(message, user, ta_username)
        # confirmation = f"Your TA question has been forwarded to TA {ta_name}. They will get back to you soon."
        # conversation_history[session_id]["awaiting_ta_question"] = False 
        # payload = {"text": confirmation, "session_id": session_id}
        # return jsonify(add_menu_button(payload))
    
    # If the user clicked the Menu button, return the full interactive menu.
    if message == "menu":
        menu_response = build_menu_response()
        menu_response["session_id"] = session_id
        return jsonify(menu_response)
    
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
        conversation_history[session_id]["awaiting_ta_question"] = ta_selected
        return jsonify({
            "text": f"Please type your question for TA {ta_selected}.",
            "session_id": session_id
        })


    # if the user selects a personality, 
    if message == "choose_personality":
        return jsonify(build_personality_response())
    
    # Handle messages from personality option buttons.
    if message.startswith("personality_"):
        personality_value = message.replace("personality_", "")
        conversation_history[session_id]["personality"] = personality_value
        confirmation = f"Personality set to: {personality_value.capitalize()}. You may now continue your conversation."
        payload = {"text": confirmation, "session_id": session_id}
        return jsonify(add_menu_button(payload))
    
    # Otherwise, process the query.
    conversation_history[session_id]["messages"].append(("user", message))
    classification = classify_query(message)
    print(f"DEBUG: User message classified as: {classification}")
    
    if classification == "not greeting":
        answer = answer_question(message, session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))
        payload = {"text": answer, "session_id": session_id}
        return jsonify(add_menu_button(payload))
    elif classification == "greeting":
        # Generate the one-sentence summary for a greeting.
        intro_summary = generate_intro_summary(session_id)
        greeting_msg = (f"Hello! Here is a one sentence summary of the paper: {intro_summary}\n"
                        "Please ask a question about the research paper, or use the buttons below for a detailed summary.\n")
        conversation_history[session_id]["messages"].append(("bot", greeting_msg))
        # interactive_payload = build_interactive_response(greeting_msg, session_id)
        # interactive_payload["session_id"] = session_id
        # return jsonify(add_menu_button(interactive_payload))
        interactive_payload = build_greeting_response(greeting_msg, session_id)
        return jsonify(interactive_payload)
        

    
    return jsonify(add_menu_button({"text": "Sorry, I didn't understand that.", "session_id": session_id}))


@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()