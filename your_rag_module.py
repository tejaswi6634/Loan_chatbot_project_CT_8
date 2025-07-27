import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def load_model_and_retriever():
    df = pd.read_csv("Training Dataset.csv")

    documents = []
    for _, row in df.iterrows():
        doc = f"""
        Applicant ID: {row['Loan_ID']}
        Gender: {row['Gender']}
        Married: {row['Married']}
        Education: {row['Education']}
        Self Employed: {row['Self_Employed']}
        Applicant Income: {row['ApplicantIncome']}
        Coapplicant Income: {row.get('CoapplicantIncome', 'N/A')}
        Loan Amount: {row['LoanAmount']}
        Loan Amount Term: {row.get('Loan_Amount_Term', 'N/A')}
        Credit History: {row.get('Credit_History', 'N/A')}
        Property Area: {row.get('Property_Area', 'N/A')}
        Loan Status: {row['Loan_Status']}
        """
        documents.append(doc.strip())

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedding_model.encode(documents)

    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))

    generator = pipeline("text2text-generation", model="google/flan-t5-small")

    return documents, embedding_model, index, generator

def retrieve_top_k(query, embedding_model, index, documents, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [documents[i] for i in indices[0]]

def generate_answer(query, context_docs, generator):
    context = "\n".join(context_docs)
    prompt = f"Answer the question based on the context below:\n{context}\n\nQuestion: {query}"
    result = generator(prompt, max_length=256, do_sample=False)
    return result[0]['generated_text']

def load_general_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

def generate_general_answer(query, tokenizer, model, max_length=80):
    input_ids = tokenizer.encode(query, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
