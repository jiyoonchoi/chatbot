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

def classify_query(message, session_id):
    prompt = (f"Classify the following user question into exactly one of:\n\n"
              "- 'greeting' (if it's just hello/hi/hey)\n"
              "- 'content_about_paper' (if it asks anything about the uploaded research paper, e.g., methods, results, ideas, implications)\n"
              "- 'class_logistics' (if it asks about class logistics: deadlines, project presentations, grading, TA office hours, etc.)\n"
              "- 'off_topic' (if it talks about unrelated things like food, movies, hobbies, etc.)\n\n"
              "Return only the label itself.\n\n"
              f"User Message: \"{message}\"")
    
    classification = generate_response("", prompt, session_id).lower().strip()
    
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
                "the paper, then press **Generate Follow-up**! 😊")

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
                "text": "🎲 Generate Follow-up",
                "msg": f"__FOLLOWUP__|{encoded}",
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
            "msg": base,
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


def send_direct_message_to_TA(question, session_id, ta_username):
    """
    Send a direct message to the TA using Rocket.Chat API.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Auth-Token": BOT_AUTH_TOKEN,
        "X-User-Id": BOT_USER_ID,
    }
    message_text = f"Student '{session_id}' asks:\n\n{question}"
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
            }
        ]
    }
    try:
        response = requests.post(MSG_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        print(f"✅ Successfully sent question to TA {ta_username}")
    except Exception as e:
        print(f"🚨 Failed to send message to TA {ta_username}: {e}")

# ------------------------------------------------------------------------
# Flask Route
# ------------------------------------------------------------------------
@app.route('/query', methods=['POST'])
def query():
    print("DEBUG: Handling query...")
    data       = request.get_json() or request.form
    user       = data.get("user_name", "Unknown")
    message    = data.get("text", "").strip()
    session_id = data.get("session_id") or get_session_id(data)

    # 0) ignore empty or bot messages
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    # 1) Human‐TA “Respond” button
    if message.lower() == "respond":
        print("DEBUG: Respond button clicked")
        # Try to pull the message _id from the payload:
        msg_id = data.get("message", {}).get("_id")
        # Look up the student session from that id:
        student_sess = ta_msg_to_student_session.get(msg_id)

        # Fallback: if Rocket.Chat didn't include exactly that id,
        # use the last mapping we have
        if not student_sess and ta_msg_to_student_session:
            _, student_sess = next(reversed(ta_msg_to_student_session.items()))
            print(f"DEBUG: Fallback mapping to student session {student_sess}")

        if student_sess:
            print(f"DEBUG: Responding to student session {student_sess}")
            # Ensure we have a history entry
            conversation_history.setdefault(student_sess, {"messages": []})
            # Set the flag so that next message goes to them
            conversation_history[student_sess]["awaiting_ta_response"] = True
            return jsonify({
                "text":       "Please type your response to the student.",
                "session_id": student_sess
            })

    # 2) TA is now typing their answer
    if conversation_history.get(session_id, {}).get("awaiting_ta_response"):
        print(f"DEBUG: Forwarding TA's reply in session {session_id}")
        conversation_history[session_id]["awaiting_ta_response"] = False
        forward_message_to_student(message, user, session_id)
        return jsonify({
            "text":       "✅ Your response has been forwarded to the student.",
            "session_id": session_id
        })

    # 3) Skip a follow-up
    if message.lower() == "skip_followup":
        conversation_history[session_id]["awaiting_followup_response"] = False
        conversation_history[session_id].pop("last_followup_question", None)
        text = "No worries! Let's continue whenever you're ready. 📚\nPlease ask another question!"
        conversation_history[session_id]["messages"].append(("bot", text))
        return jsonify(show_buttons(text, session_id))

    # 4) Initialize per-session state
    if session_id not in conversation_history:
        conversation_history[session_id] = {"messages": []}
        summary_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)

    # 5) Handle Yes/No for “awaiting_ta_confirmation”
    if conversation_history[session_id].get("awaiting_ta_confirmation") \
       and not conversation_history[session_id].get("question_flow"):

        # If they clicked a TA button, just fall through to ask_TA logic
        if message in ["ask_TA_Aya", "ask_TA_Jiyoon", "ask_TA_Amanda"]:
            conversation_history[session_id].pop("awaiting_ta_confirmation", None)

        else:
            choice = (data.get("value") or "").lower()
            if message.lower() in ["yes","y"] or choice in ["yes","y"] or message == "ask_TA":
                conversation_history[session_id].pop("awaiting_ta_confirmation", None)
                return jsonify({
                    "text": "👩‍🏫 Please select which TA you would like to ask:",
                    "attachments": [{
                        "actions": [
                            {"type":"button","text":"Ask TA Aya","msg":"ask_TA_Aya","msg_in_chat_window":True,"msg_processing_type":"sendMessage"},
                            {"type":"button","text":"Ask TA Jiyoon","msg":"ask_TA_Jiyoon","msg_in_chat_window":True,"msg_processing_type":"sendMessage"},
                            {"type":"button","text":"Ask TA Amanda","msg":"ask_TA_Amanda","msg_in_chat_window":True,"msg_processing_type":"sendMessage"}
                        ]
                    }],
                    "session_id": session_id
                })
            elif message.lower() in ["no","n"] or choice in ["no","n"]:
                conversation_history[session_id].pop("awaiting_ta_confirmation", None)
                text = "✅ No problem! Let's keep exploring the paper."
                return jsonify(show_buttons(text, session_id))
            else:
                return jsonify(show_buttons("❓ Please click Yes or No.", session_id))

    # 6) Admin commands
    if message.lower() == "clear_history":
        conversation_history.pop(session_id, None)
        summary_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)
        return jsonify(show_buttons("✅ History and caches cleared.", session_id))

    if message.lower() == "summarize":
        if not ensure_pdf_processed(session_id):
            return jsonify(show_buttons("PDF not processed yet. Please try again shortly.", session_id))
        summary_cache[session_id] = generate_response("", "Summarize the uploaded paper in 3-4 sentences.", session_id)
        return jsonify(show_buttons(summary_cache[session_id], session_id))

    # 7) Follow-up question workflow
    if message.startswith("__FOLLOWUP__|") or message.lower() == "generate_followup":
        override = None
        if message.startswith("__FOLLOWUP__|"):
            _, raw = message.split("|", 1)
            override = raw.replace("\\n", "\n").replace('\\"', '"')
        followup = generate_followup(session_id, override_last_bot=override)
        conversation_history[session_id]["awaiting_followup_response"] = True
        conversation_history[session_id]["last_followup_question"] = followup
        conversation_history[session_id]["messages"].append(("bot", followup))
        return jsonify({
            "text":       f"🧐 Follow-up:\n\n{followup}\n\nPlease reply with your thoughts!",
            "session_id": session_id,
            "attachments": [{
                "actions":[{"type":"button","text":"❌ Skip","msg":"skip_followup","msg_in_chat_window":True,"msg_processing_type":"sendMessage"}]
            }]
        })

    cmds = {"summarize", "generate_followup", "skip_followup", "clear_history"}
    
    # 8) Grading a student’s follow-up response
    if conversation_history[session_id].get("awaiting_followup_response") and message.lower() not in cmds:
        last_q = conversation_history[session_id].get("last_followup_question", "")
        grading_prompt = (
            f"Original follow-up question:\n\n\"{last_q}\"\n\n"
            f"Student's response:\n\n\"{message}\"\n\n"
            "…(your grading-and-feedback instructions here)…"
        )
        feedback = generate_response("", grading_prompt, session_id)
        conversation_history[session_id]["messages"].append(("bot", feedback))
        conversation_history[session_id].pop("awaiting_followup_response", None)
        conversation_history[session_id].pop("last_followup_question", None)
        return jsonify(show_buttons(feedback, session_id, followup_button=True))

    # ----------------------------
    # TA Question Workflow
    # ----------------------------
    if message == "ask_TA":
        # reset any previous TA‐question state
        conversation_history[session_id]["awaiting_ta_question"] = False
        conversation_history[session_id].pop("student_question",   None)
        conversation_history[session_id].pop("suggested_question", None)
        conversation_history[session_id].pop("final_question",     None)

        ta_button_response = build_TA_button()
        ta_button_response["session_id"] = session_id
        return jsonify(ta_button_response)

    # student clicked on one of the TA buttons
    if message in ["ask_TA_Aya", "ask_TA_Jiyoon", "ask_TA_Amanda"]:
        ta_selected = {
            "ask_TA_Aya":    "Aya",
            "ask_TA_Jiyoon": "Jiyoon",
            "ask_TA_Amanda":"Amanda"
        }[message]

        # start the question_flow
        conversation_history[session_id]["question_flow"] = {
            "ta": ta_selected,
            "state": "awaiting_question",
            "raw_question": "",
            "suggested_question": ""
        }
        return jsonify({
            "text":       f"Please type your question for TA {ta_selected}.",
            "session_id": session_id
        })

    # if we're in the middle of authoring a TA‐question
    q_flow = conversation_history[session_id].get("question_flow")
    if q_flow:
        state = q_flow.get("state", "")

        # safeguard exit
        if message.lower() == "exit":
            conversation_history[session_id]["question_flow"] = None
            return jsonify(show_buttons(
                "Exiting TA query mode. How can I help you with the research paper?",
                session_id
            ))

        # ─── State 1: student types their question ───
        if state == "awaiting_question":
            q_flow["raw_question"] = message
            q_flow["state"] = "awaiting_decision"
            return jsonify({
                "text": f'You typed: "{message}".\nWould you like to **refine**, **send**, or **cancel**?',
                "attachments": [{
                    "actions": [
                        {"type":"button","text":"✏️ Refine", "msg":"refine", "msg_in_chat_window":True, "msg_processing_type":"sendMessage"},
                        {"type":"button","text":"✅ Send",   "msg":"send",   "msg_in_chat_window":True, "msg_processing_type":"sendMessage"},
                        {"type":"button","text":"❌ Cancel", "msg":"cancel", "msg_in_chat_window":True, "msg_processing_type":"sendMessage"}
                    ]
                }],
                "session_id": session_id
            })

        # ─── State 2: refine / send / cancel ───
        if state == "awaiting_decision":
            cmd = message.lower()
            if cmd == "send":
                ta_username = "aya.ismail" if q_flow["ta"] == "Aya" else "jiyoon.choi"
                final_q = q_flow.get("suggested_question") or q_flow["raw_question"]
                send_direct_message_to_TA(final_q, user, ta_username)
                conversation_history[session_id]["question_flow"] = None
                return jsonify(show_buttons(
                    f"Your question has been sent to TA {q_flow['ta']}!",
                    session_id
                ))
            elif cmd == "cancel":
                conversation_history[session_id]["question_flow"] = None
                return jsonify(show_buttons(
                    "Your TA question process has been canceled. Let me know if you need anything else.",
                    session_id
                ))
            elif cmd == "refine":
                suggested, clean = generate_suggested_question(session_id, q_flow["raw_question"])
                q_flow["suggested_question"] = clean
                q_flow["state"] = "awaiting_refinement_decision"
                return jsonify({
                    "text": (
                        f"Here is a suggestion:\n\n\"{clean}\"\n\n"
                        "Do you **approve**, **modify**, **manual edit**, or **cancel**?"
                    ),
                    "session_id": session_id,
                    **build_refinement_buttons(q_flow)
                })
            else:
                return jsonify({
                    "text": "Please choose **refine**, **send**, or **cancel**.",
                    "session_id": session_id
                })

        # ─── State 3: awaiting feedback on suggestion ───
        if state == "awaiting_feedback":
            feedback = message
            base = q_flow.get("suggested_question") or q_flow["raw_question"]
            _, clean = generate_suggested_question(session_id, base, feedback)
            q_flow["suggested_question"] = clean
            q_flow["state"] = "awaiting_refinement_decision"
            return jsonify({
                "text": (
                    f"Updated suggestion:\n\n\"{clean}\"\n\n"
                    "Approve, modify, manual edit, or cancel?"
                ),
                "session_id": session_id,
                **build_refinement_buttons(q_flow)
            })

        # ─── State 4: manual edit ───
        if state == "awaiting_manual_edit":
            edited = message.strip()
            q_flow["suggested_question"] = edited
            q_flow["state"] = "awaiting_refinement_decision"
            return jsonify({
                "text": f"Here's your manually edited question:\n\n\"{edited}\"\n\nWhat next?",
                "session_id": session_id,
                **build_refinement_buttons(q_flow)
            })
    # ─────────────────── end of TA Workflow ───────────────────

    # 11) Finally, normal conversation
    conversation_history[session_id]["messages"].append(("user", message))
    classification = classify_query(message, session_id)
    print(f"DEBUG: Classified as {classification}")

    # 11a) greeting
    if classification == "greeting":
        ensure_pdf_processed(session_id)
        intro = generate_response("", "Give a one-line overview: 'This week's paper discusses...'", session_id)
        greeting = (
            "**Hello! 👋 I am the TA chatbot for CS-150…**\n\n"
            f"**{intro}**\n\n"
            "📄 Quick Summary | 👩‍🏫 Ask a TA"
        )
        conversation_history[session_id]["messages"].append(("bot", greeting))
        return jsonify(show_buttons(greeting, session_id, summary_button=True))

    # 11b) paper content
    if classification == "content_about_paper":
        difficulty = classify_difficulty(message, session_id)
        ensure_pdf_processed(session_id)
        if difficulty == "factual":
            answer = generate_response("", f"Answer factually: {message}", session_id)
        else:
            answer = generate_response("", f"Answer conceptually in 1-2 sentences, then suggest where to look: {message}", session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))
        return jsonify(show_buttons(answer, session_id, followup_button=True))

    # 11c) class logistics
    if classification == "class_logistics":
        tip = generate_response("", f"Student asked: \"{message}\". Give a 1-2 sentence friendly tip.", session_id)
        conversation_history[session_id]["messages"].append(("bot", tip))
        # then offer TA
        conversation_history[session_id]["awaiting_ta_confirmation"] = True
        return jsonify({
            "text":       f"{tip}\n\nWould you like to ask your TA for more clarification? 🧐",
            "attachments": [{
                "actions": [
                    {"type":"button","text":"✅ Yes, Ask TA","msg":"ask_TA","value":"yes","msg_in_chat_window":True,"msg_processing_type":"sendMessage"},
                    {"type":"button","text":"❌ No","msg":"ask_TA","value":"no","msg_in_chat_window":True,"msg_processing_type":"sendMessage"}
                ]
            }],
            "session_id": session_id
        })

    # 11d) off-topic
    if classification == "off_topic":
        text = "🚫 That seems off-topic! Let's focus on the paper or class logistics."
        conversation_history[session_id]["messages"].append(("bot", text))
        return jsonify(show_buttons(text, session_id))

    # 11e) fallback
    fallback = "❓ I didn't quite catch that. Try asking about the paper!"
    conversation_history[session_id]["messages"].append(("bot", fallback))
    return jsonify(show_buttons(fallback, session_id))


# ------------------------------------------------------------------------
# Server Start
# ------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()