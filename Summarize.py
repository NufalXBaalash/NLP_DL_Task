from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# --- Initialize the model ---
def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

# --- Read Files Functions ---
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()   
    return docs

def merge_pages(docs):
    full_text = ""
    for doc in docs:
        page_num = doc.metadata["page"]
        full_text += f"\n[PAGE:{page_num}]\n" + doc.page_content
    return full_text

def read_word(file_path):
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    return docs[0].page_content

def chunk_text(full_text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([full_text])
    return docs

def extract_pages_from_chunk(chunk_text):
    pages = re.findall(r"\[PAGE:(\d+)\]", chunk_text)
    return list(sorted(set(map(int, pages))))

# --- Generate Summaries ---
def summarize_chunk(chunk_text, api_key, location=None):
    location_label = f"Location: {location}" if location else ""
    prompt = f"""
    You are an AI summarizer. Summarize the following text clearly and concisely.
    Include the main ideas and important points. Keep it professional and easy to understand.
    {location_label}

    Text:
    \"\"\"{chunk_text}\"\"\"
    """

    client = get_client(api_key)
    response = client.chat.completions.create(
        model="x-ai/grok-4.1-fast",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    summary = response.choices[0].message.content.strip()
    return summary

# --- Functions for PDF or Word ---
def pdf_to_summary(file_path, api_key):
    pages = read_pdf(file_path)          
    full_text = merge_pages(pages)       
    chunks = chunk_text(full_text)       

    summaries = []

    for chunk in chunks:
        text = chunk.page_content
        pages = extract_pages_from_chunk(text)
        location_str = f"Pages {', '.join(map(str, pages))}"
        summary = summarize_chunk(text, api_key, location=location_str)
        summaries.append(summary)

    # Combine chunk summaries into a final summary
    final_summary = "\n".join(summaries)
    return final_summary

def word_to_summary(file_path, api_key):
    full_text = read_word(file_path)
    chunks = chunk_text(full_text)

    summaries = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        location_str = f"Chunk {i+1}"
        summary = summarize_chunk(text, api_key, location=location_str)
        summaries.append(summary)

    final_summary = "\n".join(summaries)
    return final_summary

# --------- The Full Pipeline ---------
def summarize_pipeline(filepath, api_key):
    if filepath.endswith(".pdf"):
        return pdf_to_summary(filepath, api_key)
    elif filepath.endswith(".docx"):
        return word_to_summary(filepath, api_key)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")

