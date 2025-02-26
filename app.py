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

# Global conversation history.
conversation_history = {}

def fetch_paper_text(link):
    """Fetches paper text from a given link, extracting either PDF or webpage content."""
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    print(f"DEBUG: Fetching paper text from URL: {link}")
    response = requests.get(link, headers=headers)
    if response.status_code != 200:
        print(f"DEBUG: Error fetching URL: {link} - Status code: {response.status_code}")
        return ""

    soup = BeautifulSoup(response.content, "html.parser")
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
            return extract_text_from_pdf(temp_pdf.name)
        else:
            print(f"DEBUG: Failed to fetch PDF. Status: {pdf_response.status_code}")
    
    print("DEBUG: No PDF found, extracting HTML text.")
    return soup.get_text(separator="\n")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        print(f"DEBUG: Extracted text from PDF at {pdf_path}")
    except Exception as e:
        print(f"DEBUG: Error reading PDF: {e}")
    return text

@app.route('/query', methods=['POST'])
def query():
    """Handles incoming queries, including interactive button presses."""
    data = request.get_json()
    print(f"DEBUG: Received request data: {data}")

    if data.get("interactive_callback"):
        action = data.get("action")
        paper_link = data.get("link")
        if action in ["summarize_abstract", "summarize_full"]:
            return handle_summarization(paper_link, action)
        return jsonify({"error": "Unknown action"}), 400

    if data.get("action", "").lower() == "summarize":
        paper_link = data.get("link")
        if not paper_link:
            return jsonify({"error": "No paper link provided"}), 400
        
        interactive_message = {
            "text": "Would you like a summary of the abstract only or a full overview?",
            "attachments": [
                {
                    "actions": [
                        {
                            "type": "button",
                            "text": "Abstract Only",
                            "msg": f"/query",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage",
                            "params": {"interactive_callback": True, "action": "summarize_abstract", "link": paper_link}
                        },
                        {
                            "type": "button",
                            "text": "Full Overview",
                            "msg": f"/query",
                            "msg_in_chat_window": True,
                            "msg_processing_type": "sendMessage",
                            "params": {"interactive_callback": True, "action": "summarize_full", "link": paper_link}
                        }
                    ]
                }
            ]
        }
        return jsonify(interactive_message)

    return jsonify({"error": "Unsupported query type"}), 400

def handle_summarization(paper_link, action_type):
    """
    Generates a summary of the requested paper text and returns it as a new message.
    """
    print(f"DEBUG: Handling summarization for {action_type} - {paper_link}")
    paper_text = fetch_paper_text(paper_link)

    if not paper_text.strip():
        return jsonify({"text": "❌ Error: Unable to retrieve paper content."})

    excerpt = paper_text[:3000]
    
    summary_prompt = (
        f"Summarize the abstract:\n\n{excerpt}" if action_type == "summarize_abstract"
        else f"Summarize the full paper while retaining all important details:\n\n{excerpt}"
    )

    print("DEBUG: Sending summary request to LLM model")
    summary_response = generate(
        model='4o-mini',
        system="You are an expert summarizer of academic papers.",
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

    return jsonify({"text": summary_text})

@app.errorhandler(404)
def page_not_found(e):
    return "Not Found", 404

if __name__ == "__main__":
    print("DEBUG: Starting Flask server...")
    app.run()