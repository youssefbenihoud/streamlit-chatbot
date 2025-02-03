import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Load the pre-trained model for generating embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load the GPT-2 text generation model
@st.cache_resource
def load_gpt2():
    return pipeline('text-generation', model='gpt2')

# Load the knowledge base
@st.cache_data
def load_knowledge_base(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

# Perform semantic search on the knowledge base
def retrieve_relevant_documents(query, knowledge_base, model, top_k=1):
    # Generate embeddings for the query and knowledge base
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(knowledge_base, convert_to_tensor=True)

    # Compute cosine similarity between query and documents
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    # Get the top-k most similar documents
    top_results = torch.topk(cos_scores, k=top_k)
    return [knowledge_base[idx] for idx in top_results.indices]

# Generate a response using GPT-2
def generate_response(query, context, generator):
    # Combine the query and context to form the input prompt
    prompt = f"Question: {query}\nContext: {context}\nAnswer: "
    
    # Generate text using GPT-2
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text'].split("Answer: ")[-1]

# Set the page title and header
st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")

# Load the models and knowledge base
model = load_model()
gpt2_generator = load_gpt2()
knowledge_base = load_knowledge_base('knowledge_base.txt')

# Add a text input for user queries
user_input = st.text_input("Ask me anything!")

if user_input:
    st.write(f"You asked: {user_input}")

    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_documents(user_input, knowledge_base, model)

    if relevant_docs:
        # Display retrieved documents
        st.write("Relevant Information:")
        for doc in relevant_docs:
            st.write(f"- {doc}")

        # Generate a response using GPT-2
        context = " ".join(relevant_docs)  # Combine all relevant documents
        response = generate_response(user_input, context, gpt2_generator)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Chatbot: I don't have enough information to answer that.")