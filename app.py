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

def generate_followup(session_id):
    history = conversation_history.get(session_id, {}).get("messages", [])
    last_bot_message = next((msg for speaker, msg in reversed(history) if speaker == "bot"), None)
    if not last_bot_message:
        return None

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
        attachments.append({
            "actions": [{
                "type": "button",
                "text": "🎲 Generate Follow-up",
                "msg": "generate_followup",
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
# ------------------------------------------------------------------------
# Flask Route
# ------------------------------------------------------------------------
@app.route('/query', methods=['POST'])
def query():
    print("DEBUG: Handling query...")
    data = request.get_json() or request.form
    message = data.get("text", "").strip()

    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    session_id = get_session_id(data)

    if message.lower() == "skip_followup":
        conversation_history[session_id]["awaiting_followup_response"] = False
        conversation_history[session_id].pop("last_followup_question", None)
        text = "No worries! Let's continue whenever you're ready. 📚"
        conversation_history[session_id]["messages"].append(("bot", text))
        return jsonify(show_buttons(text, session_id))

    if session_id not in conversation_history:
        conversation_history[session_id] = {"messages": []}
        summary_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)

    if conversation_history[session_id].get("awaiting_ta_confirmation"):
        if message.lower() in ["yes", "y"]:
            conversation_history[session_id]["awaiting_ta_confirmation"] = False
            conversation_history[session_id]["question_flow"] = {
                "state": "awaiting_question",
                "ta": None,  # you can fill this if you allow choosing a TA
                "raw_question": "",
                "suggested_question": ""
            }
            return jsonify({
                "text": "✅ Great! Please type your question for the TA.",
                "session_id": session_id
            })

        elif message.lower() in ["no", "n"]:
            conversation_history[session_id]["awaiting_ta_confirmation"] = False
            text = "✅ No problem! Let's keep exploring the paper."
            conversation_history[session_id]["messages"].append(("bot", text))
            return jsonify(show_buttons(text, session_id))
        else:
            return jsonify(show_buttons("❓ Please click Yes or No.", session_id))

    if conversation_history[session_id].get("awaiting_followup_response"):
        conversation_history[session_id]["awaiting_followup_response"] = False
        last_followup = conversation_history[session_id].pop("last_followup_question", "")

        grading_prompt = (
            f"Original follow-up question:\n\n"
            f"\"{last_followup}\"\n\n"
            f"Student's response:\n\n"
            f"\"{message}\"\n\n"
            "Consider the following 2 cases and keep response concise, encouraging, and related to the uploaded paper:\n"
            "Case 1: If the original follow-up question prompts a concrete answer, evaluate the student's response:\n"
            "- If correct or mostly correct, confirm warmly and optionally elaborate briefly.\n"
            "- If partially correct, point out missing parts politely.\n"
            "- If wrong, gently correct them and guide them where to look in the paper.\n"
            "Case 2: If the original follow-up question is vague or open-ended, evaluate the student's response:\n"
            "- If the student provides a concrete answer, confirm warmly and optionally elaborate briefly.\n"
            "- If the student provides a vague or open-ended answer, gently correct them and guide them where to look in the paper.\n\n"
        )

        feedback = generate_response("", grading_prompt, session_id)
        conversation_history[session_id]["messages"].append(("bot", feedback))

        return jsonify(show_buttons(feedback, session_id, followup_button=True))

    # Special admin commands
    if message.lower() == "clear_history":
        conversation_history.pop(session_id, None)
        summary_cache.pop(session_id, None)
        processed_pdf.pop(session_id, None)
        pdf_ready.pop(session_id, None)
        return jsonify(show_buttons("✅ History and caches cleared.", session_id))

    if message.lower() == "summarize":
        if not ensure_pdf_processed(session_id):
            return jsonify(show_buttons("PDF not processed yet. Please try again shortly.", session_id))
        summary = generate_response("", "Summarize the uploaded paper in 3-4 sentences.", session_id)
        summary_cache[session_id] = summary
        return jsonify(show_buttons(summary, session_id))

    # follow up
    if message.lower() == "generate_followup":
        followup = generate_followup(session_id)
        if followup:
            conversation_history[session_id]["awaiting_followup_response"] = True
            conversation_history[session_id]["last_followup_question"] = followup
            conversation_history[session_id]["messages"].append(("bot", followup))
            return jsonify({
                "text": f"🧐 Follow-up:\n\n{followup}\n\nPlease reply with your thoughts!",
                "session_id": session_id,
                "attachments": [{"actions": [{"type": "button", "text": "❌ Skip", "msg": "skip_followup", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}]}]
            })
        
    if conversation_history[session_id].get("question_flow"):
        q_flow = conversation_history[session_id]["question_flow"]
        if q_flow.get("state") == "awaiting_question":
            q_flow["raw_question"] = message
            q_flow["state"] = "ready_to_send"
            return jsonify({
                "text": f"📩 You wrote: \"{message}\".\n\nDo you want to send it as-is, refine it, or cancel?",
                "session_id": session_id,
                "attachments": [
                    {"actions": [{"text": "✅ Send", "msg": "send_ta", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}]},
                    {"actions": [{"text": "✏️ Refine", "msg": "refine_ta", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}]},
                    {"actions": [{"text": "❌ Cancel", "msg": "cancel_ta", "msg_in_chat_window": True, "msg_processing_type": "sendMessage"}]}
                ]
            })

    # Process normal message
    conversation_history[session_id]["messages"].append(("user", message))
    classification = classify_query(message, session_id)
    print(f"DEBUG: Classified as {classification}")

    if classification == "greeting":
        ensure_pdf_processed(session_id)
        intro = generate_response("", "Give a one-line overview: 'This week's paper discusses...'", session_id)

        greeting_msg = (
            "**Hello! 👋 I am the TA chatbot for CS-150: Generative AI for Social Impact. 🤖**\n\n"
            "I'm here to help you *critically analyze ONLY this week's* research paper, which I *encourage you to read* before interacting with me. "
            "I'll guide you to the key sections and ask thought-provoking questions—but I won't just hand you the answers. 🤫\n\n"
            "You have two buttons to choose from:\n"
            "- 📄 **Quick Summary** - Get a concise 3-4 sentence overview of the paper's main objectives and findings.\n"
            "- 🧑‍🏫 **Ask TA** - Send your question to a human TA if you'd like extra help.\n\n"
            f"**{intro}**\n\n"
            "If there's a question I can't fully answer, I'll prompt you to forward it to your TA. "
            "Please ask a question about the paper now or click one of the buttons below!"
        )

        conversation_history[session_id]["messages"].append(("bot", greeting_msg))
        return jsonify(show_buttons(greeting_msg, session_id, summary_button=True))

    if classification == "content_about_paper":
        difficulty = classify_difficulty(message, session_id)
        ensure_pdf_processed(session_id)
        if difficulty == "factual":
            print("DEBUG: Factual question detected.")
            # Special case for metadata questions
            if any(keyword in message.lower() for keyword in ["author", "authors", "who wrote", "title", "publication"]):
                # Very strict system prompt for metadata
                system_prompt = (
                    "You are a TA chatbot answering factual metadata questions about the uploaded TwIPS paper. "
                    "ONLY use the title page and the first page of the paper. "
                    "Ignore all references or citations. "
                    "If the requested information (like authorship or title) is not clearly stated, say so."
                )
                prompt = (
                    "Based solely on the front matter (title page and first page) of the uploaded TwIPS paper, "
                    f"answer the following factual question:\n\n{message}\n\n"
                    "If the information is unclear, say so politely."
                )
                answer = generate(
                    model='4o-mini',
                    system=system_prompt,
                    query=prompt,
                    session_id=session_id,
                    temperature=0.0,
                    lastk=5,
                    rag_usage=True,
                    rag_threshold=0.02,  # tighter threshold
                    rag_k=10  # broader top-k search
                )
                if isinstance(answer, dict):
                    answer = answer.get("response", "").strip()
                else:
                    answer = answer.strip()
            else:
                # Normal factual answering
                answer = generate_response("", f"Answer factually: {message}", session_id)
        else:
            print("DEBUG: Conceptual question detected.")
            answer = generate_response("", f"Answer conceptually in 1-2 sentences, then suggest where to look in the paper for details: {message}", session_id)
        conversation_history[session_id]["messages"].append(("bot", answer))
        return jsonify(show_buttons(answer, session_id, followup_button=True))

    if classification == "class_logistics":
        # Step 1: Try to give a short chatbot answer first
        short_answer = generate_response(
            "", 
            f"You are a TA chatbot for CS-150. The student asked: \"{message}\". "
            "Give a short, friendly, 1-2 sentence general tip, but do not make up specific class policies. "
            "If unsure, encourage them to ask the human TA for details.", 
            session_id
        )
        conversation_history[session_id]["messages"].append(("bot", short_answer))

        # Step 2: THEN offer human TA help
        conversation_history[session_id]["awaiting_ta_confirmation"] = True

        return jsonify({
            "text": f"{short_answer}\n\nWould you like to ask your TA for more clarification? 🧐",
            "attachments": [{
                "actions": [
                    {
                        "type": "button",
                        "text": "✅ Yes, Ask TA",
                        "msg": "yes",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "❌ No",
                        "msg": "no",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }],
            "session_id": session_id
        })

    if classification == "off_topic":
        text = "🚫 That seems off-topic! Let's focus on the research paper or class logistics."
        conversation_history[session_id]["messages"].append(("bot", text))
        return jsonify(show_buttons(text, session_id))

    # fallback
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