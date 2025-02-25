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

def classify_query(message):
    """
    Uses an LLM agent to classify the query.
    The LLM should return one word: 'greeting', 'research', or 'other'.
    """
    prompt = (
        f"Determine if the following message is a greeting, a research query, or something else. "
        f"Reply with just one word: 'greeting' if it's a simple greeting, 'research' if it's asking for research-related information, or 'other' if it is unrelated to research. "
        f"Message: \"{message}\""
    )
    classification = generate(
        model='4o-mini',
        system="You are a query classifier. Classify the user message as 'greeting', 'research', or 'other'.",
        query=prompt,
        temperature=0.0,
        lastk=0,
        session_id="classify_" + str(uuid.uuid4())
    )
    if isinstance(classification, dict):
        classification_text = classification.get('response', '').strip().lower()
    else:
        classification_text = classification.strip().lower()
    return classification_text

def google_search(query, num_results=3):
    """
    Performs a Google Custom Search focused on research papers and datasets.
    For each result, a 'View Paper' link is provided along with an interactive 
    'Summarize Paper' button.
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
    if response.status_code != 200:
        print(f"Google Search API Error: {response.status_code}, {response.text}")
        return []
    
    results = response.json().get("items", [])
    search_summaries = []
    for item in results:
        title = item.get("title", "No title available")
        snippet = item.get("snippet", "No description available")
        link = item.get("link", "#")
        # The "Summarize Paper" button is added as a secondary command.
        search_summaries.append(
            f"**{title}**\n{snippet}\n[🔗 View Paper]({link})  [Summarize Paper](command:summarize?link={link})\n"
        )
    return search_summaries

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()

    # Check if this is a summarization action.
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
            "datasets, and scientific studies. I can also chat generally. How may I assist you today?"
        )
        conversation_history[session_id] = [("bot", intro_message)]
    else:
        intro_message = None

    conversation_history[session_id].append(("user", message))

    classification = classify_query(message)
    
    if classification == "research":
        # Always use 3 results.
        search_results = google_search(message, num_results=3)
        search_info = "\n".join(search_results) if search_results else "No relevant research found."

        query_with_context = ""
        for sender, text in conversation_history[session_id]:
            query_with_context += f"{sender.capitalize()}: {text}\n"
        query_with_context += f"\nResearch Findings:\n{search_info}"

        research_response = generate(
            model='4o-mini',
            system=(
                "You are a Research Assistant AI that specializes in retrieving and summarizing "
                "academic research, datasets, and scientific studies. Provide well-cited, fact-based insights "
                "from reputable sources. If the query is general chat, respond as a friendly assistant."
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
        bot_reply = f"{summary_heading}{response_text}\n\n{papers_heading}{search_info}"
    
    elif classification == "greeting":
        # Use general conversation branch.
        query_with_context = ""
        for sender, text in conversation_history[session_id]:
            query_with_context += f"{sender.capitalize()}: {text}\n"
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
    
    elif classification == "other":
        # Out-of-scope response.
        bot_reply = (
            "I'm a research paper assistant and I specialize in academic research, datasets, and scientific studies. "
            "I'm sorry, but I can only help with research-related queries. Please ask me about research topics or papers."
        )
    else:
        # Fallback to general conversation.
        query_with_context = ""
        for sender, text in conversation_history[session_id]:
            query_with_context += f"{sender.capitalize()}: {text}\n"
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

    conversation_history[session_id].append(("bot", bot_reply))

    if intro_message and len(conversation_history[session_id]) == 2:
        bot_reply = f"{intro_message}\n\n{bot_reply}"

    return jsonify({"text": bot_reply, "session_id": session_id})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()