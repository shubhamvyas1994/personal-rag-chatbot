import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="My RAG Chatbot", page_icon="🤖")
st.title("🤖 My Personal RAG Chatbot")

# Debug: Show if key loaded
st.write(f"API Key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")

# Load DB only once
@st.cache_resource
def load_qa_chain():
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt_template = """You are a friendly personal assistant chatbot.

Rules:
1. For greetings, casual chat, or general questions like "how are you", respond naturally and conversationally.
2. For questions about specific facts, numbers, or details, use ONLY the context below.
3. If the context doesn't contain the answer to a factual question, say: "I don't have that information in the provided documents."
4. Do NOT make up facts from the documents.
5. Keep answers concise.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
        )
        return qa
    except Exception as e:
        st.error(f"Error loading QA chain: {e}")
        st.stop()

qa = load_qa_chain()

st.success("Chatbot ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        result = qa.invoke(prompt)
        response = result['result']
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})