import streamlit as st
from your_rag_module import (
    load_model_and_retriever,
    retrieve_top_k,
    generate_answer,
    load_general_model,
    generate_general_answer
)

st.set_page_config(page_title="Loan Q&A Chatbot", layout="centered")
st.title("📄 Loan Q&A Chatbot with RAG")

if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []

@st.cache_resource
def setup():
    return load_model_and_retriever()

@st.cache_resource
def setup_general():
    return load_general_model()

documents, embedding_model, index, generator = setup()
general_tokenizer, general_model = setup_general()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask your loan-related or general question:")

if st.button("Get Answer") and user_query:
    with st.spinner("Thinking..."):
        if any(x in user_query.lower() for x in ["loan", "credit", "applicant", "income", "property"]):
            top_docs = retrieve_top_k(user_query, embedding_model, index, documents)
            answer = generate_answer(user_query, top_docs, generator)
            context = top_docs
        else:
            answer = generate_general_answer(user_query, general_tokenizer, general_model)
            context = ["N/A (general question)"]

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": answer,
            "context": context
        })

if st.session_state.chat_history:
    st.subheader("🗂️ Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}: {chat['question']}**")
        st.markdown(f"*Answer:* {chat['answer']}")
        with st.expander("Context Used"):
            for j, doc in enumerate(chat["context"]):
                st.markdown(f"**Document {j+1}:**")
                st.text(doc)
        st.markdown("---")
