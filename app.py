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

def is_research_query(message):
    """
    Checks if the message contains research-related keywords.
    """
    keywords = ['paper', 'research', 'study', 'dataset', 'data', 'journal', 'article']
    return any(word in message.lower() for word in keywords)

def is_greeting_query(message):
    """
    Returns True if the message is a simple greeting.
    """
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    msg_lower = message.strip().lower()
    return any(msg_lower == greeting for greeting in greetings)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()

    # Check if this is a summarization action.
    if data.get("action", "").lower() == "summarize":
        paper_link = data.get("link")
        if not paper_link:
            return jsonify({"error": "No paper link provided"}), 400

        # Prepare the summarization prompt.
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

        # Generate a PDF from the summary text using FPDF.
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary_text)
        
        # Save the PDF to a temporary file.
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)
        temp_pdf.close()

        return send_file(temp_pdf.name, as_attachment=True, download_name="Paper_Summary.pdf")

    # Otherwise, process as a normal chat/research query.
    user_id = data.get("user_id", "unknown_user")
    # Generate a new session ID if one is not provided.
    session_id = data.get("session_id", f"session_{user_id}_{str(uuid.uuid4())}")
    message = data.get("text", "")

    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})

    # Initialize conversation history with an introduction if it's a new session.
    if session_id not in conversation_history:
        intro_message = (
            "Hello! I am Research Assistant Bot. I specialize in retrieving and summarizing academic research, "
            "datasets, and scientific studies. I can also chat generally. How may I assist you today?"
        )
        conversation_history[session_id] = [("bot", intro_message)]
    else:
        intro_message = None

    # Append the user's message to the conversation history.
    conversation_history[session_id].append(("user", message))

    # Determine whether to treat the query as research-related.
    if is_research_query(message) and not is_greeting_query(message):
        # Always use 3 results.
        search_results = google_search(message, num_results=3)
        search_info = "\n".join(search_results) if search_results else "No relevant research found."

        # Build conversation context.
        query_with_context = ""
        for sender, text in conversation_history[session_id]:
            query_with_context += f"{sender.capitalize()}: {text}\n"
        query_with_context += f"\nResearch Findings:\n{search_info}"

        # Call the LLM with a research assistant prompt.
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
    else:
        # General conversational branch.
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

    # Append the bot's reply to the conversation history.
    conversation_history[session_id].append(("bot", bot_reply))

    # For the very first interaction, prepend the introduction.
    if intro_message and len(conversation_history[session_id]) == 2:
        bot_reply = f"{intro_message}\n\n{bot_reply}"

    return jsonify({"text": bot_reply, "session_id": session_id})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    app.run()