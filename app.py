import os
import uuid
import io
import tempfile
from flask import Flask, request, jsonify, send_file
from llmproxy import generate
import requests
from fpdf import FPDF
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

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    print(f"DEBUG: Received request data: {data}")

    # Handle interactive callback from Rocket.Chat button presses.
    if data.get("interactive_callback"):
        action = data.get("action")
        paper_link = data.get("link")
        if action in ["summarize_abstract", "summarize_full"]:
            return handle_summarization(paper_link, action)
        else:
            return jsonify({"error": "Unknown action"}), 400

    # Summarize Paper Action (initial request) remains the same.
    if data.get("action", "").lower() == "summarize":
        paper_link = data.get("link")
        if not paper_link:
            print("DEBUG: No paper link provided in request")
            return jsonify({"error": "No paper link provided"}), 400
        # Send interactive message with Rocket.Chat buttons for a single paper.
        interactive_message = {
            "text": "Would you like a summary of the abstract only or a full overview?",
            "attachments": [
                {
                    "actions": [
                        {
                            "type": "button",
                            "text": "Abstract Only",
                            "msg": f"/summarize_abstract {paper_link}",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        },
                        {
                            "type": "button",
                            "text": "Full Overview",
                            "msg": f"/summarize_full {paper_link}",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage"
                        }
                    ]
                }
            ]
        }
        print("DEBUG: Returning interactive button message")
        return jsonify(interactive_message)

    # Conversation handling for research queries, greetings, etc.
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
                        "msg": f"/summarize_abstract {result['link']}",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    },
                    {
                        "type": "button",
                        "text": "Full Overview",
                        "msg": f"/summarize_full {result['link']}",
                        "msg_in_chat_window": True,
                        "msg_processing_type": "sendMessage"
                    }
                ]
            }
            interactive_attachments.append(attachment)

        # Build the query context with previous conversation.
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
        bot_reply = response_text
        conversation_history[session_id].append(("bot", bot_reply))
        if intro_message and len(conversation_history[session_id]) == 2:
            bot_reply = f"{intro_message}\n\n{bot_reply}"
        # Return both the bot reply and the interactive attachments for each search result.
        return jsonify({"text": bot_reply, "attachments": interactive_attachments, "session_id": session_id})
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

def handle_summarization(paper_link, action_type):
    """
    Helper function to fetch paper text and generate a summary.
    The action_type determines if we summarize just the abstract or provide a full overview.
    """
    print(f"DEBUG: Handling summarization for action: {action_type}, link: {paper_link}")
    paper_text = fetch_paper_text(paper_link)
    if not paper_text or len(paper_text.strip()) == 0:
        print("DEBUG: Could not retrieve paper content")
        return jsonify({"error": "Could not retrieve paper content"}), 400

    excerpt = paper_text[:3000]
    if action_type == "summarize_abstract":
        summary_prompt = (
            f"Please provide a detailed summary focusing on the abstract of the following research paper text:\n\n{excerpt}"
        )
    else:  # summarize_full
        summary_prompt = (
            f"Please provide a detailed summary of the following research paper text, including key findings, methodology, and conclusions:\n\n{excerpt}"
        )

    print("DEBUG: Sending summarization prompt to LLM")
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
    # Generate PDF in memory using FPDF.
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf_output = pdf.output(dest="S").encode("latin1")
    pdf_buffer = io.BytesIO(pdf_output)
    pdf_buffer.seek(0)
    print("DEBUG: PDF generated in memory, ready for download")
    return send_file(pdf_buffer, as_attachment=True, download_name="Paper_Summary.pdf", mimetype='application/pdf')

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()