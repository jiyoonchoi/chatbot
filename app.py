import os
import uuid
import io
import tempfile
from flask import Flask, request, jsonify
from llmproxy import generate
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin
import PyPDF2
from bs4 import BeautifulSoup

load_dotenv()

app = Flask(__name__)
app.config['PDF_FOLDER'] = os.path.join(os.getcwd(), 'static', 'pdfs')
os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)

# Environment variables for Google Custom Search API.
API_KEY = os.environ.get("googleApiKey")
CSE_ID = os.environ.get("googleSearchId")

# Global conversation history.
conversation_history = {}

def classify_query(message):
    prompt = (
        f"Determine if the following message is a greeting, a research query, or something else. "
        f"Reply with just one word: 'greeting' if it's a simple greeting, 'research' if it's asking for research-related information, or 'other' if it is unrelated to research. "
        f"Message: \"{message}\""
    )
    print(f"DEBUG: Classifying query: {message}")
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
    print(f"DEBUG: Classification result: {classification_text}")
    return classification_text

def google_search(query, num_results=3):
    """
    Perform a Google Custom Search for research-related PDFs and resources.
    Enhances the query to target research papers.
    Returns a list of result objects containing title, snippet, and link.
    """
    search_query = (
        f"{query} filetype:pdf OR site:researchgate.net OR site:ncbi.nlm.nih.gov OR site:data.gov "
        "OR site:arxiv.org OR site:worldbank.org OR site:europa.eu OR site:sciencedirect.com OR site:scholar.google.com"
    )
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'q': search_query, 'key': API_KEY, 'cx': CSE_ID, 'num': num_results}
    print(f"DEBUG: Performing Google Search with query: {search_query}")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"DEBUG: Google Search API Error: {response.status_code}, {response.text}")
        return []
    results = response.json().get("items", [])
    search_results = []
    for item in results:
        title = item.get("title", "No title available")
        snippet = item.get("snippet", "No description available")
        link = item.get("link", "#")
        search_results.append({
            "title": title,
            "snippet": snippet,
            "link": link
        })
    print(f"DEBUG: Google search results: {search_results}")
    return search_results

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        print(f"DEBUG: Extracted text from PDF at {pdf_path}")
    except Exception as e:
        print(f"DEBUG: Error reading PDF: {e}")
    return text

def fetch_paper_text(link):
    paper_text = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/98.0.4758.102 Safari/537.36"
    }
    print(f"DEBUG: Fetching paper text from URL: {link}")
    response = requests.get(link, headers=headers)
    if response.status_code != 200:
        print(f"DEBUG: Error fetching URL: {link} - Status code: {response.status_code}")
        return paper_text
    soup = BeautifulSoup(response.content, "html.parser")
    print(f"DEBUG: Fetched HTML snippet: {soup.get_text()[:300]}")
    pdf_anchor = soup.find("a", href=lambda href: href and ".pdf" in href.lower())
    if pdf_anchor:
        pdf_link = urljoin(link, pdf_anchor.get("href"))
        print(f"DEBUG: Found PDF link: {pdf_link}")
        pdf_response = requests.get(pdf_link, headers=headers)
        if pdf_response.status_code == 200:
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            with open(temp_pdf.name, "wb") as f:
                f.write(pdf_response.content)
            print(f"DEBUG: Downloaded PDF to temporary file: {temp_pdf.name}")
            paper_text = extract_text_from_pdf(temp_pdf.name)
        else:
            print(f"DEBUG: Error fetching PDF link: {pdf_link} - Status code: {pdf_response.status_code}")
    else:
        print("DEBUG: No PDF link found; using HTML text")
        paper_text = soup.get_text(separator="\n")
    print(f"DEBUG: Retrieved paper text length: {len(paper_text)} characters")
    return paper_text

def get_summary_text(paper_link, action_type):
    """
    Fetches the paper text, sends a summarization prompt to the LLM, and returns the summary as a string.
    """
    print(f"DEBUG: Getting summary for action: {action_type}, link: {paper_link}")
    paper_text = fetch_paper_text(paper_link)
    if not paper_text or len(paper_text.strip()) == 0:
        print("DEBUG: Could not retrieve paper content")
        return None

    excerpt = paper_text[:3000]
    if action_type == "summarize_abstract":
        summary_prompt = (
            f"Please provide a detailed summary focusing on the abstract of the following research paper text:\n\n{excerpt}"
        )
    else:  # summarize_full
        summary_prompt = (
            f"Please provide a detailed summary of the following research paper text, including key findings, methodology, and conclusions:\n\n{excerpt}"
        )

    print("DEBUG: Sending summarization prompt to LLM agent")
    summary_response = generate(
        model='4o-mini',
        system="You are an expert summarizer of academic papers. Provide a detailed and accurate summary.",
        query=summary_prompt,
        temperature=0.0,
        lastk=0,
        session_id="summarize_" + str(uuid.uuid4())
    )
    if isinstance(summary_response, dict):
        summary_text = summary_response.get('response', '').strip()
    else:
        summary_text = summary_response.strip()

    print(f"DEBUG: Received summary text: {summary_text[:300]}...")
    return summary_text

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    print(f"DEBUG: Received request data: {data}")
    message = data.get("text", "")

    # Check if the incoming message is a summarization command.
    if message.startswith("/summarize_abstract") or message.startswith("/summarize_full"):
        # Use an existing session ID if available or create a new one.
        session_id = data.get("session_id", f"session_{data.get('user_id', 'unknown')}_{str(uuid.uuid4())}")
        parts = message.split()
        action = parts[0][1:]  # Remove the leading '/'
        paper_link = " ".join(parts[1:])

        summary_text = get_summary_text(paper_link, action)
        if summary_text is None:
            return jsonify({"error": "Could not retrieve paper content"}), 400

        # Do not add the summarization command itself to the conversation history.
        # Instead, append the summary as a hidden entry.
        conversation_history.setdefault(session_id, [])
        conversation_history[session_id].append(("hidden_summary", summary_text))

        # Now have the main LLM generate a response that includes the summary.
        final_prompt = "Here is the summary of the requested paper:\n\n" + summary_text
        main_response = generate(
            model='4o-mini',
            system="You are a Research Assistant AI. Relay the following paper summary to the user in a clear and concise manner.",
            query=final_prompt,
            temperature=0.0,
            lastk=0,
            session_id=session_id
        )
        if isinstance(main_response, dict):
            response_text = main_response.get('response', '').strip()
        else:
            response_text = main_response.strip()

        conversation_history[session_id].append(("bot", response_text))
        return jsonify({"text": response_text, "session_id": session_id})

    # Handle interactive callbacks (from button presses) similarly.
    if data.get("interactive_callback"):
        action = data.get("action")
        paper_link = data.get("link")
        if action in ["summarize_abstract", "summarize_full"]:
            # Use an existing session or create one.
            session_id = data.get("session_id", f"session_{data.get('user_id', 'unknown')}_{str(uuid.uuid4())}")
            summary_text = get_summary_text(paper_link, action)
            if summary_text is None:
                return jsonify({"error": "Could not retrieve paper content"}), 400

            conversation_history.setdefault(session_id, [])
            conversation_history[session_id].append(("hidden_summary", summary_text))

            final_prompt = "Here is the summary of the requested paper:\n\n" + summary_text
            main_response = generate(
                model='4o-mini',
                system="You are a Research Assistant AI. Relay the following paper summary to the user in a clear and concise manner.",
                query=final_prompt,
                temperature=0.0,
                lastk=0,
                session_id=session_id
            )
            if isinstance(main_response, dict):
                response_text = main_response.get('response', '').strip()
            else:
                response_text = main_response.strip()

            conversation_history[session_id].append(("bot", response_text))
            return jsonify({"text": response_text, "session_id": session_id})
        else:
            return jsonify({"error": "Unknown action"}), 400

    # If the user clicked "Summarize Paper", return an interactive message with two buttons.
    if data.get("action", "").lower() == "summarize":
        paper_link = data.get("link")
        if not paper_link:
            print("DEBUG: No paper link provided in request")
            return jsonify({"error": "No paper link provided"}), 400
        interactive_message = {
            "text": "Would you like a summary of the abstract only or a full overview?",
            "attachments": [
                {
                    "actions": [
                        {
                            "type": "button",
                            "text": "Abstract Only",
                            "action": "summarize_abstract",  # new key instead of 'msg'
                            "link": paper_link,             # provide the paper link
                            "msg_in_chat_window": False,    # do not show command in chat
                            "msg_processing_type": "interactive_callback"  # trigger callback
                        },
                        {
                            "type": "button",
                            "text": "Full Overview",
                            "action": "summarize_full",
                            "link": paper_link,
                            "msg_in_chat_window": False,
                            "msg_processing_type": "interactive_callback"
                        }
                    ]
                }
            ]
        }
        print("DEBUG: Returning interactive button message")
        return jsonify(interactive_message)

    # For regular conversation queries, proceed with normal chat handling.
    user_id = data.get("user_id", "unknown_user")
    session_id = data.get("session_id", f"session_{user_id}_{str(uuid.uuid4())}")
    if data.get("bot") or not message:
        return jsonify({"status": "ignored"})
    if session_id not in conversation_history:
        intro_message = (
            "Hello! I am Research Assistant Bot. I specialize in retrieving and summarizing academic research, "
            "datasets, and scientific studies. I can also chat generally. How may I assist you today?"
        )
        conversation_history[session_id] = [("bot", intro_message)]
    conversation_history[session_id].append(("user", message))
    classification = classify_query(message)
    if classification == "research":
        search_results = google_search(message, num_results=3)
        # Build interactive attachments for each search result.
        interactive_attachments = []
        for result in search_results:
            attachment = {
                "text": f"*{result['title']}*\n{result['snippet']}\n[🔗 View Paper]({result['link']})",
                "actions": [
                    {
                        "type": "button",
                        "text": "Abstract Only",
                        "action": "summarize_abstract",
                        "link": result['link'],
                        "msg_in_chat_window": False,
                        "msg_processing_type": "interactive_callback"
                    },
                    {
                        "type": "button",
                        "text": "Full Overview",
                        "action": "summarize_full",
                        "link": result['link'],
                        "msg_in_chat_window": False,
                        "msg_processing_type": "interactive_callback"
                    }
                ]
            }
            interactive_attachments.append(attachment)

        query_with_context = "\n".join(text for _, text in conversation_history[session_id])
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
        conversation_history[session_id].append(("bot", response_text))
        return jsonify({"text": response_text, "attachments": interactive_attachments, "session_id": session_id})
    elif classification == "greeting":
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
    elif classification == "other":
        bot_reply = (
            "I'm a research paper assistant and I specialize in academic research, datasets, and scientific studies. "
            "I'm sorry, but I can only help with research-related queries. Please ask me about research topics or papers."
        )
    else:
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
    conversation_history[session_id].append(("bot", bot_reply))
    return jsonify({"text": bot_reply, "session_id": session_id})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()