from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from openai import OpenAI
import json
import re

# --- Initialize the model ---
def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# --- Read Files Functions ---

# Read the data from the PDF
def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()   # list of pages
    return docs            # each doc has metadata["page"]

# Merge the pages from PDFs
def merge_pages(docs):
    full_text = ""
    for doc in docs:
        page_num = doc.metadata["page"]
        full_text += f"\n[PAGE:{page_num}]\n" + doc.page_content
    return full_text

# Read the data from Doc
def read_word(file_path):
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    full_text = docs[0].page_content
    return full_text

# Chunk the text (for model tokens)
def chunk_text(full_text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs = splitter.create_documents([full_text])
    return docs

# Extract used pages to make Questions from PDFs
def extract_pages_from_chunk(chunk_text):
    pages = re.findall(r"\[PAGE:(\d+)\]", chunk_text)
    pages = list(sorted(set(map(int, pages))))  # unique, sorted
    return pages


# --- Generate Questions and Answers ---

# --- Generate Questions and Answers ---

def generate_qa_per_chunk(chunk_text, api_key, location=None):
    # Determine number of questions based on length
    # e.g., 1 question per 300 characters, min 2, max 5
    text_len = len(chunk_text)
    num_questions = max(2, min(5, text_len // 300))

    # Build location label for prompt
    location_label = f"Location: {location}" if location else ""

    prompt = f"""
    You are an AI question generator. Your task is to create **{num_questions} high-quality questions** based on the following text chunk.
    
    Mix the question types:
    - At least one **True/False** question.
    - The rest should be **Short Answer** questions.
    
    Questions must focus on the **main idea**, not minor or irrelevant details.

    If the text is empty, extremely short, or contains no meaningful information, return an empty list:
    []

    Return the output strictly as a valid JSON array of objects. 
    Each object must have the following structure:
    {{
        "question": "The question text here",
        "answer": "The answer here",
        "type": "short_answer" OR "true_false",
        "location": "{location if location else 'Unknown'}"
    }}

    Context:
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

    content = response.choices[0].message.content

    # Clean up markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
             # Handle case where model returns a single object instead of list
             return [data]
        else:
            return []
    except:
        # Fallback or error logging could go here
        return []


# --- Functions for PDF or Word ---
def pdf_to_qa(file_path, api_key):
    pages = read_pdf(file_path)          
    full_text = merge_pages(pages)       
    chunks = chunk_text(full_text)       

    all_qas = []

    for chunk in chunks:
        text = chunk.page_content
        pages = extract_pages_from_chunk(text)
        # Convert pages list to string for location
        location_str = f"Pages {', '.join(map(str, pages))}"
        
        qa_list = generate_qa_per_chunk(text, api_key, location=location_str)
        if qa_list:
            all_qas.extend(qa_list)

    return all_qas

def word_to_qa(file_path, api_key):
    full_text = read_word(file_path)
    chunks = chunk_text(full_text)

    all_qas = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        location_str = f"Chunk {i+1}"
        
        qa_list = generate_qa_per_chunk(text, api_key, location=location_str)
        if qa_list:
            all_qas.extend(qa_list)

    return all_qas


# --------- The Full Pipeline ---------
def qa_pipeline(filepath, api_key):
  if filepath.endswith(".pdf"):
    return pdf_to_qa(filepath, api_key)
  elif filepath.endswith(".docx"):
    return word_to_qa(filepath, api_key)