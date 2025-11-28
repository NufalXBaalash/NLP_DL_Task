from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import json

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

# --- Generate Keywords ---
def extract_keywords_chunk(chunk_text, api_key, location=None):
    location_label = f"Location: {location}" if location else ""
    prompt = f"""
    You are an AI keyword extractor. Extract the most important keywords and concepts from the following text.
    Return them as a bulleted list.
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

    keywords = response.choices[0].message.content.strip()
    return keywords

# --- Functions for PDF or Word ---
def pdf_to_keywords(file_path, api_key):
    pages = read_pdf(file_path)          
    full_text = merge_pages(pages)       
    chunks = chunk_text(full_text)       

    all_keywords = []

    for chunk in chunks:
        text = chunk.page_content
        pages = extract_pages_from_chunk(text)
        location_str = f"Pages {', '.join(map(str, pages))}"
        keywords = extract_keywords_chunk(text, api_key, location=location_str)
        all_keywords.append(keywords)

    final_keywords = "\n\n".join(all_keywords)
    return final_keywords

def word_to_keywords(file_path, api_key):
    full_text = read_word(file_path)
    chunks = chunk_text(full_text)

    all_keywords = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        location_str = f"Chunk {i+1}"
        keywords = extract_keywords_chunk(text, api_key, location=location_str)
        all_keywords.append(keywords)

    final_keywords = "\n\n".join(all_keywords)
    return final_keywords

# --------- The Full Pipeline ---------
def keyword_pipeline(filepath, api_key):
    if filepath.endswith(".pdf"):
        return pdf_to_keywords(filepath, api_key)
    elif filepath.endswith(".docx"):
        return word_to_keywords(filepath, api_key)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")
