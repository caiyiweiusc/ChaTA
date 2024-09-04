import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from threading import Thread
import json
from bs4 import BeautifulSoup
import sqlite3
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import requests
import time

app = Flask(__name__)
CORS(app)
conversation_chain = None

db_texts = None
db_vectors = None

def initialize_data():
    """Initialize data loaded from the database."""
    global db_texts, db_vectors, db_links  # Add db_links as a global variable
    db_texts, db_vectors, db_links = read_vectors_from_db()
    print("Database data has been loaded.")

def get_db_connection(db_path="text_vectors.db"):
    """Connect to the SQLite database."""
    conn = sqlite3.connect(db_path)
    return conn

def read_vectors_from_db():
    """Read text blocks, corresponding vectors, and video links (if any) from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text, vector, video_time_link FROM text_vectors")
    db_data = cursor.fetchall()
    conn.close()
    texts, vectors, links = zip(*db_data)
    vectors = [np.fromstring(vector[1:-1], sep=', ') for vector in vectors]  # Convert string back to numpy array
    return texts, np.array(vectors), links

# Embedding PDF
def get_pdf_text(pdf_path):
    """Extract text from a PDF file at the specified path."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Add "or ''" to avoid None
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def save_text_to_db(text, db_path="piazza_posts.db"):
    """Save text to the database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)''')
    try:
        c.execute("INSERT INTO posts (content) VALUES (?)", (text,))
        conn.commit()
        print("Text saved successfully to database.")
    except sqlite3.IntegrityError as e:
        print(f"Error saving text to database: {e}")
    finally:
        conn.close()

def run_command_line_interaction():
    global conversation_chain
    while True:
        embed_pdf = input("Do you want to embed a PDF? (y/n): ")
        if embed_pdf.lower() == 'y':
            pdf_path = input("Enter the path to the PDF: ")
            if os.path.exists(pdf_path):
                pdf_text = get_pdf_text(pdf_path)
                # Save PDF text to the database
                save_text_to_db(pdf_text)
                print("PDF text embedding process completed.")
            else:
                print("PDF file does not exist. Please check the path.")
        elif embed_pdf.lower() == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

# Get Embedding
def get_embedding(text, model="text-embedding-3-small"):
    """
    Get the embedding vector for a given text.
    
    :param text: The text to get the embedding for.
    :param model: The name of the embedding model to use.
    :return: A list containing the embedding vector.
    """
    openai_api_key = ""  # Use your API key
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}
    data = {"input": text, "model": model}
    response = requests.post("https://api.openai.com/v1/embeddings", json=data, headers=headers)
    
    if response.status_code == 200:
        embedding_vector = response.json()["data"][0]["embedding"]
        return embedding_vector
    else:
        print("Failed to get embedding:", response.text)
        return None

# Compare similarity
def compare_user_question_to_db_vectors(user_question, db_vectors, db_texts, db_links, top_k=15):
    user_question_vector = get_embedding(user_question)

    if user_question_vector is not None:
        similarities = cosine_similarity([user_question_vector], db_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare to collect text and links
        top_texts = []
        top_links = []

        for index in top_indices:
            top_texts.append(db_texts[index])
            top_links.append(db_links[index])
            # Add logic to include previous and next texts
            if index > 0:
                top_texts.append(db_texts[index - 1])
                top_links.append(db_links[index - 1])
            if index < len(db_texts) - 1:
                top_texts.append(db_texts[index + 1])
                top_links.append(db_links[index + 1])

        # Check for video links in the top 5 results and return relevant text along with its surrounding text
        for i in range(min(5, len(top_links))):
            if top_links[i]:  # Assume that checking for a video link involves checking for specific characters or domains in the link
                # Found a video link, return the text for that link and its surrounding text
                linked_texts = [top_texts[i]]
                linked_texts_indices = [i]
                
                # Add previous text
                if i > 0:
                    linked_texts.insert(0, top_texts[i - 1])
                    linked_texts_indices.insert(0, i - 1)
                
                # Add next text
                if i < len(top_texts) - 1:
                    linked_texts.append(top_texts[i + 1])
                    linked_texts_indices.append(i + 1)

                print(linked_texts)
                return linked_texts, top_links[i]

        # If no video link is found in the top results, return the expanded list of texts and None
        return top_texts, None
    else:
        print("Failed to vectorize user question.")
        return [], None

# Merge texts in the get_vectorstore function
def get_vectorstore(texts):
    openai_api_key = ""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)  # Note that this is a list of individual strings
    return vectorstore

def get_conversation_chain(vectorstore):
    openai_api_key = ""
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
    )
    return conversation_chain

def start_cli_thread():
    cli_thread = Thread(target=run_command_line_interaction)
    cli_thread.daemon = True  # Set as a daemon thread so it will automatically end when the main program ends
    cli_thread.start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        start_time_total = time.time()
        user_message = request.json["message"]
        
        start_time_top_texts = time.time()
        top_texts, video_link = compare_user_question_to_db_vectors(user_message, db_vectors, db_texts, db_links)
        end_time_top_texts = time.time()
        print(f"Time taken to get top_texts: {end_time_top_texts - start_time_top_texts} seconds")
        
        if video_link:
            vectorstore = get_vectorstore(top_texts)
            conversation_chain = get_conversation_chain(vectorstore)
            response = conversation_chain({"question": user_message})
            # If a video link is found, embed the first text and provide the video link
            chatbot_response = response["answer"] + f"\n\nHere's a related video you might find helpful: {video_link}\n\n"
        else:
            # No video link found, process according to original logic
            vectorstore = get_vectorstore(top_texts)
            conversation_chain = get_conversation_chain(vectorstore)
            response = conversation_chain({"question": user_message})
            chatbot_response = response["answer"]

        end_time_total = time.time()
        print(f"Total time taken from receiving user question to responding: {end_time_total - start_time_total} seconds")

        print(f"Chatbot response: {chatbot_response}")
        
        return jsonify({"response": chatbot_response})
    except KeyError:
        return jsonify({"error": "Invalid request data"}), 400

if __name__ == "__main__":
    load_dotenv()
    start_cli_thread()
    initialize_data()  # Load data when the server starts
    app.run(debug=False, port=5001)
