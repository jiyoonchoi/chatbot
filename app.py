import os
import uuid
import tempfile
from flask import Flask, request, jsonify, send_file
from llmproxy import generate
import requests
from fpdf import FPDF
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
    Formats a single search result item into a Rocket.Chat attachment with action buttons.
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
    Performs a Google Custom Search and returns a plain text summary and a list of attachments.
    """
    search_query = (
        f"{query} filetype:pdf OR site:researchgate.net OR site:ncbi.nlm.nih.gov OR site:data.gov "
        "OR site:arxiv.org OR site:worldbank.org OR site:europa.eu OR site:sciencedirect.com OR site:scholar.google.com"
    )
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': search_query,
        'key': API_KEY,
        'cx': CSE_ID,
        'num': num_results
    }
    
    response = requests.get(url, params=params)
    attachments = []
    summary_texts = []
    
    if response.status_code != 200:
        print(f"Google Search API Error: {response.status_code}, {response.text}")
        return "No relevant research found.", []
    
    results = response.json().get("items", [])
    for item in results:
        summary_texts.append(f"**{item.get('title', 'No title available')}**")
        attachments.append(format_search_attachment(item))
    
    summary_text = "\n".join(summary_texts) if summary_texts else "No relevant research found."
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
    return "greeting" if result == "greeting" else "research"

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()

    # Handle summarization action.
    if data.get("action", "").lower() == "summarize":
        paper_link = data.get("link")
        if not paper_link:
            return jsonify({"error": "No paper link provided"}), 400

        summary_prompt = f"Please provide a concise summary of the paper available at: {paper_link}"
        summary_response = generate(
            model='4o-mini',
            system="You are an expert summarizer of academic papers. Provide clear and concise summaries.",
            query=summary_prompt,
            temperature=0.0,
            lastk=0,
            session_id="summarize_" + str(uuid.uuid4())
        )
        if isinstance(summary_response, dict):
            summary_text = summary_response.get('response', '').strip()
        else:
            summary_text = summary_response.strip()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary_text)
        
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)
        temp_pdf.close()

        return send_file(temp_pdf.name, as_attachment=True, download_name="Paper_Summary.pdf")

    # Process as a normal chat/research query.
    user_id = data.get("user_id", "unknown_user")
    session_id = data.get("session_id", f"session_{user_id}_{str(uuid.uuid4())}")
    message = data.get("text", "")

    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    if session_id not in conversation_history:
        intro_message = (
            "Hello! I am Research Assistant Bot. I specialize in retrieving and summarizing academic research, "
            "datasets, and scientific studies. How may I assist you today?"
        )
        conversation_history[session_id] = [("bot", intro_message)]
    else:
        intro_message = None

    conversation_history[session_id].append(("user", message))
    classification = classify_query(message)

    if classification == "greeting":
        # Handle as a general conversational greeting.
        query_with_context = "\n".join(text for _, text in conversation_history[session_id])
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
        result = {"text": bot_reply, "session_id": session_id}
    
    else:  # Everything not a greeting is handled as research.
        summary_text, attachments = google_search(message, num_results=3)
        query_with_context = "\n".join(text for _, text in conversation_history[session_id])
        query_with_context += f"\nResearch Findings:\n{summary_text}"

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

        summary_heading = "**📚 Research Summary:**\n"
        papers_heading = "**🔗 Relevant Research Papers & Datasets:**\n"
        bot_reply = f"{summary_heading}{response_text}\n\n{papers_heading}{summary_text}"
        result = {"text": bot_reply, "session_id": session_id, "attachments": attachments}

    conversation_history[session_id].append(("bot", result["text"]))

    if intro_message and len(conversation_history[session_id]) == 2:
        result["text"] = f"{intro_message}\n\n{result['text']}"

    return jsonify(result)

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()