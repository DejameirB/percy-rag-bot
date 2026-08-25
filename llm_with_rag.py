import os
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Part 1: Knowledge Base ---
documents = [
    "ByteWorks offers a 30-day return window on unopened items. Opened items are eligible for store credit only, minus a 15% restocking fee.",
    "Standard domestic shipping takes 3-5 business days. Express shipping is available at checkout and delivers within 1-2 business days.",
    "All ByteWorks hardware products include a 1-year limited warranty covering manufacturing defects.",
    "We accept major credit cards (Visa, MasterCard, American Express), PayPal, and Apple Pay.",
    "Customer support is available via live chat and email from Monday to Friday, 9 AM to 6 PM EST.",
    "Custom PC builds require 5-7 business days for assembly and testing before shipment.",
    "Orders can be canceled within 2 hours of placement by accessing your account order history.",
    "International shipping is available to select countries, with delivery taking 7-14 business days.",
    "Trade-in programs allow customers to submit old hardware for store credit toward new purchases.",
    "Software licenses and digital download purchases are strictly non-refundable."
]

doc_ids = [f"doc_{i}" for i in range(len(documents))]

SYSTEM_PROMPT = (
    "You are ByteBot, a helpful customer service assistant for ByteWorks. "
    "Answer customer questions using ONLY the provided context. "
    "If the answer cannot be found in the context, politely state: 'I apologize but I do not know. Please contact customer support.'"
)

# --- Part 2: Vector Store (Local Mode) ---
def build_vector_store(documents, doc_ids):
    # Running locally without needing cloud API keys
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="customer_support_kb",
        embedding_function=None
    )

    vectorizer = TfidfVectorizer()
    doc_embeddings = vectorizer.fit_transform(documents).toarray().tolist()

    collection.add(
        documents=documents,
        embeddings=doc_embeddings,
        ids=doc_ids
    )

    return collection, vectorizer

# --- Part 3: Retrieval ---
def retrieve(collection, vectorizer, question, n_results=3):
    query_embedding = vectorizer.transform([question]).toarray().tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results["documents"][0]

# --- Part 4: Prompting & Generation ---
def build_prompt(question, retrieved_docs):
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    return f"Context:\n{context}\n\nQuestion: {question}"

def generate_answer(messages):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "(No DEEPSEEK_API_KEY found -- check your .env file)"

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages
    )

    return response.choices[0].message.content

# --- Part 5: End-to-End Chat ---
def chat(collection, vectorizer, question, history=None, n_results=3):
    retrieved_docs = retrieve(collection, vectorizer, question, n_results=n_results)
    current_prompt = build_prompt(question, retrieved_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": current_prompt})

    return generate_answer(messages)
    
