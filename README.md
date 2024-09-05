# ChaTA: Intelligent Chat Assistant for Education with PDF and Video Integration

ChaTA is an advanced, AI-driven chat application designed specifically for educational support. It combines natural language processing, document analysis, and video content integration to provide intelligent, context-aware responses to student queries in real-time.

## Introduction

ChaTA represents a powerful AI tool for education, built using Python, Flask, JavaScript, and HTML. It integrates advanced AI-driven technologies to deliver an experience that effectively mirrors the support provided by a human Teaching Assistant. This platform is tailored to enhance student support in the education sector, providing instant, accurate responses to a wide range of academic queries. By leveraging cutting-edge AI and incorporating multimedia elements, ChaTA aims to create a comprehensive and engaging learning environment for students across various disciplines.

## Key Features

- Interactive chat interface with a user-friendly design
- Real-time, context-aware responses to student queries
- Instant answers to questions about assignments, course policies, and subject-specific content
- PDF text extraction and embedding for comprehensive understanding of course materials
- Integration with OpenAI's language models for intelligent, TA-like responses
- Similarity-based retrieval of relevant text segments from course documents
- Video link integration for enhanced learning experience:
  - Students can watch instructional videos directly within the chat interface
  - Helps in understanding complex topics through visual aids
- Conversation history management for continuous learning context
- Command-line interface for easy PDF embedding of course materials

## Enhanced Student Support

ChaTA is designed to significantly improve the student learning experience:

- Provides timely and accurate assistance comparable to that of a teaching assistant
- Addresses a wide range of student queries instantly, reducing wait times for support
- Offers 24/7 availability, allowing students to get help at any time
- Supports video responses, enabling multi-modal learning within the chat interface
- Maintains context across conversations, providing personalized and relevant support

## Technologies Used

- Backend:
  - Flask: A lightweight WSGI web application framework
  - OpenAI API: For GPT language models and text embeddings
  - LangChain: For building applications with language models
  - SQLite: For lightweight, serverless database management
  - PyPDF2: For extracting text from PDF files
  - FAISS: For efficient similarity search and clustering of dense vectors
- Frontend:
  - HTML/CSS: For structuring and styling the web interface
  - JavaScript: For dynamic client-side scripting
  - Fetch API: For making asynchronous HTTP requests

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chata.git
   cd chata
   ```

2. Install required Python packages:
   ```
   pip install flask flask-cors python-dotenv PyPDF2 langchain faiss-cpu openai requests beautifulsoup4 scikit-learn numpy
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5001`

3. Start chatting with the AI teaching assistant!

4. To embed course PDF content:
   - When prompted in the command line, enter 'y' to embed a PDF
   - Provide the path to your course PDF file

## Project Structure

- `app.py`: Main Flask application with routing and core logic
- `templates/index.html`: Frontend HTML template
- `text_vectors.db`: SQLite database for storing text vectors
- `piazza_posts.db`: SQLite database for storing PDF content

## Key Components

1. **PDF Processing**: 
   - Extracts text from course PDF files using PyPDF2
   - Saves extracted text to a SQLite database

2. **Text Embedding**:
   - Utilizes OpenAI's text embedding model to convert text into vector representations

3. **Similarity Search**:
   - Implements cosine similarity to find relevant text segments based on student queries

4. **Conversational AI**:
   - Employs LangChain and OpenAI's ChatGPT for generating contextual, TA-like responses

5. **Video Integration**:
   - Associates text segments with relevant instructional video links
   - Embeds YouTube videos in chat responses for visual learning

---

Thank you for your interest in ChaTA! I hope this tool enhances the educational experience for students and educators alike.

