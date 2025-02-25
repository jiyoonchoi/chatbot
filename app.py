import os
import uuid
from flask import Flask, request, jsonify
from llmproxy import generate
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Environment variables for Google Custom Search API.
API_KEY = os.environ.get("googleApiKey")
CSE_ID = os.environ.get("googleSearchId")

# Global store for conversation history (keyed by session_id).
conversation_history = {}

def format_search_attachment(item):
    """
    Formats a single search result item into an attachment with action buttons.
    """
    title = item.get("title", "No title available")
    snippet = item.get("snippet", "No description available")
    link = item.get("link", "#")
    
    attachment = {
        "title": title,
        "text": snippet,
        "actions": [
            {
                "type": "button",
                "text": "View Paper",
                "url": link
            },
            {
                "type": "button",
                "text": "Summarize Paper",
                "url": f"command:summarize?link={link}"
            }
        ]
    }
    return attachment

def google_search(query, num_results=3):
    """
    Performs a Google Custom Search and returns a plain text summary (a list of top paper titles)
    and a list of attachments.
    """
    search_query = (
        f"{query} filetype:pdf OR site:researchgate.net OR site:ncbi.nlm.nih.gov OR site:data.gov "
        "OR site:arxiv.org OR site:worldbank.org OR site:europa.eu OR site:sciencedirect.com OR site:scholar.google.com"
    )
    print(f"[DEBUG] Google search query: {search_query}")
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'q': search_query, 'key': API_KEY, 'cx': CSE_ID, 'num': num_results}
    response = requests.get(url, params=params)
    print(f"[DEBUG] Google search API response status: {response.status_code}")
    attachments = []
    summary_texts = []
    
    if response.status_code != 200:
        print(f"[ERROR] Google Search API Error: {response.status_code}, {response.text}")
        return "No relevant research found.", []
    
    results = response.json().get("items", [])
    print(f"[DEBUG] Google search returned {len(results)} items.")
    for item in results:
        summary_texts.append(f"**{item.get('title', 'No title available')}**")
        attachments.append(format_search_attachment(item))
    
    summary_text = "\n".join(summary_texts) if summary_texts else "No relevant research found."
    print(f"[DEBUG] Google search summary text: {summary_text}")
    return summary_text, attachments

def classify_query(message):
    """
    Uses an LLM agent to classify the query.
    Returns 'greeting' if the message is a greeting; otherwise, defaults to research.
    """
    prompt = (
        f"Is the following message a simple greeting? Answer with just one word: 'greeting' if yes, otherwise answer 'not greeting'. "
        f"Message: \"{message}\""
    )
    print(f"[DEBUG] Classification prompt: {prompt}")
    classification = generate(
        model='4o-mini',
        system="You are a query classifier. Determine if the user message is a greeting.",
        query=prompt,
        temperature=0.0,
        lastk=0,
        session_id="classify_" + str(uuid.uuid4())
    )
    if isinstance(classification, dict):
        result = classification.get('response', '').strip().lower()
    else:
        result = classification.strip().lower()
    print(f"[DEBUG] Raw classification result: {result}")
    return "greeting" if result == "greeting" else "research"

@app.route('/query', methods=['POST'])
def query():
    print("[INFO] Received query request.")
    data = request.get_json()
    print(f"[DEBUG] Request data: {data}")

    # Handle summarization action.
    if data.get("action", "").lower() == "summarize":
        paper_link = data.get("link")
        if not paper_link:
            return jsonify({"error": "No paper link provided"}), 400

        summary_prompt = f"Please provide a concise summary of the paper available at: {paper_link}"
        summary_response = generate(
            model='4o-mini',
            system="You are an expert summarizer of academic papers. Provide a brief summary in plain text.",
            query=summary_prompt,
            temperature=0.0,
            lastk=0,
            session_id="summarize_" + str(uuid.uuid4())
        )
        if isinstance(summary_response, dict):
            summary_text = summary_response.get('response', '').strip()
        else:
            summary_text = summary_response.strip()
        print(f"[DEBUG] Summarization response: {summary_text}")
        # Return the summary directly as a chat message.
        return jsonify({"msg": summary_text})

    # Process as a normal chat/research query.
    user_id = data.get("user_id", "unknown_user")
    session_id = data.get("session_id", f"session_{user_id}_{str(uuid.uuid4())}")
    message = data.get("text", "")
    print(f"[DEBUG] user_id: {user_id}, session_id: {session_id}, message: {message}")

    if data.get("bot") or not message:
        print("[INFO] Message ignored because it's from bot or empty.")
        return jsonify({"status": "ignored"})

    if session_id not in conversation_history:
        intro_message = (
            "Hello! I am Research Assistant Bot. I specialize in retrieving and summarizing academic research, "
            "datasets, and scientific studies. How may I assist you today?"
        )
        conversation_history[session_id] = [("bot", intro_message)]
        print("[DEBUG] New session created with intro message.")
    else:
        intro_message = None

    conversation_history[session_id].append(("user", message))
    classification = classify_query(message)
    print(f"[DEBUG] Classification for message '{message}': {classification}")

    if classification == "greeting":
        query_with_context = "\n".join(text for _, text in conversation_history[session_id])
        print(f"[DEBUG] Context for general response:\n{query_with_context}")
        general_response = generate(
            model='4o-mini',
            system="You are a friendly chatbot assistant who will prompt the user to provide a research topic they're interested in.",
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
        result = {"msg": bot_reply}
        print(f"[DEBUG] General response: {bot_reply}")
    
    else:
        search_info, attachments = google_search(message, num_results=3)
        query_with_context = "\n".join(text for _, text in conversation_history[session_id])
        query_with_context += f"\nResearch Findings:\n{search_info}"
        print(f"[DEBUG] Context for research response:\n{query_with_context}")

        research_response = generate(
            model='4o-mini',
            system=(
                "You are a Research Assistant AI that specializes in retrieving and summarizing academic research, "
                "datasets, and scientific studies. Provide well-cited, fact-based insights from reputable sources."
            ),
            query=query_with_context,
            temperature=0.0,
            lastk=0,
            session_id=session_id
        )
        if isinstance(research_response, dict):
            response_text = research_response.get('response', "").strip()
        else:
            response_text = research_response.strip()
        print(f"[DEBUG] Research LLM response: {response_text}")
        if not response_text:
            response_text = "I'm sorry, I couldn't retrieve additional research findings for that topic."
            print("[DEBUG] Using fallback research response.")

        bot_reply = f"**📚 Top Results for Your Research:**\n{response_text}\n\n{search_info}"
        result = {"msg": bot_reply, "attachments": attachments}
        print(f"[DEBUG] Final research response: {bot_reply}")

    conversation_history[session_id].append(("bot", result.get("msg", "")))
    print(f"[DEBUG] Updated conversation history for session {session_id}: {conversation_history[session_id]}")
    if intro_message and len(conversation_history[session_id]) == 2:
        result["msg"] = f"{intro_message}\n\n{result['msg']}"
        print(f"[DEBUG] Prepending intro message: {result['msg']}")
    print(f"[INFO] Sending final response for session {session_id}")
    return jsonify(result)

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()