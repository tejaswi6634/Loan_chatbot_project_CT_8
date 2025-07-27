# 💬 Loan Q&A Chatbot using RAG with Streamlit GUI

This project is an interactive **Loan-related Question and Answer Chatbot** that uses **Retrieval-Augmented Generation (RAG)** to answer both **loan-specific** and **general questions**. It includes a user-friendly GUI built with **Streamlit**, making it easy to interact with the model via a web interface.

---

## 🚀 Features

-  Ask questions about loans or general topics.
-  Uses RAG to retrieve and generate high-quality, context-aware answers.
-  Incorporates a lightweight model to run on limited memory.
-  Streamlit-based interface for seamless chatting experience.
-  Maintains chat history during a session.
-  Clear History button to reset the conversation when needed.

---

## 📁 Project Files

- `Loan Q&A Chatbot with RAG.ipynb`: Jupyter Notebook containing model setup, document processing, and basic RAG implementation.
- `your_rag_module.py`: Python module with functions to load the model and generate answers.
- `app.py`: Streamlit web application script to run the GUI interface.

---

## 🛠️ Setup Instructions
pip install streamlit transformers sentence-transformers faiss-cpu


---
## Running the Application
streamlit run app.py
The Streamlit app will open in your browser at:
http://localhost:8501
