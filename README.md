Resume AI Assistant
This is a full-stack application for uploading, analyzing, and querying resumes. The frontend is built with React, and the backend is a Python API using FastAPI and LangChain.

Features
Multi-Format Resume Upload: A sleek UI for uploading resumes in .txt, .pdf, .doc, or .docx formats.

AI Extraction: Uses a Large Language Model (LLM) to extract key information (e.g., name, experience, skills) from the resume and store it in a structured JSON format.

Markdown Generation: Automatically generates a professional, well-formatted markdown version of the resume.

RAG-based Q&A: A Retrieval-Augmented Generation (RAG) model allows you to ask questions about the uploaded resumes (e.g., "List all resumes with Python experience," "Who has worked at Google?").

Project Structure
.
├── backend/
│   ├── app.py           # The FastAPI backend application
│   ├── llm_provider.py  # Centralized LLM provider logic
│   └── requirements.txt # Python dependencies
├── frontend/
│   └── src/
│       └── App.jsx      # The single-file React frontend
├── .env.example         # Example environment file
└── README.md            # This README file

How to Run
1. Set up the Backend
Navigate to the backend directory.

Install the required Python packages:

pip install -r requirements.txt

Create a .env file in the backend directory based on .env.example.

cp .env.example .env

Get your API key from Groq and your connection string from your MongoDB Atlas account.

In the .env file, replace the placeholder values with your actual keys and connection string.

GROQ_API_KEY="your_groq_api_key_here"
MONGO_DB_URI="your_mongodb_connection_string_here"

Run the FastAPI application with Uvicorn:

uvicorn app:app --reload

The backend will be available at http://localhost:8000.

2. Run the Frontend
The React frontend is a single file and can be run in any environment that supports React.

Navigate to the frontend directory.

Install the dependencies:

npm install

Run the app:

npm start

The frontend will be available at http://localhost:3000.

Note: For local development, the frontend is configured to make API calls to http://localhost:8000.

Deployment
Render
Frontend: For the frontend, you can use a static site host like Render's Static Sites.

Backend: The Python backend can be deployed as a Render Web Service. Configure the build command as pip install -r requirements.txt and the start command as uvicorn app:app --host 0.0.0.0 --port $PORT. Remember to add your GROQ_API_KEY and MONGO_DB_URI as environment variables in the Render dashboard.

Cloudflare
Frontend: The frontend can be deployed using Cloudflare Pages.

Backend: The Python backend can be deployed using Cloudflare Workers with Python support or a similar service.

Important Notes on MongoDB & Vector Search
The provided app.py uses an in-memory dictionary (resume_database) and an in-memory vector store (Chroma) for demonstration.

For a production application, you should:

Install the pymongo library.

Replace the resume_database dictionary with a MongoClient connection to your MongoDB Atlas cluster.

Load documents from your MongoDB collection.

Utilize MongoDB Atlas Vector Search for your RAG model. This is the most efficient and scalable solution.
