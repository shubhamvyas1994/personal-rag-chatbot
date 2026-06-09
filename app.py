import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

DB_PATH = "faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# THIS IS THE MISSING PIECE - SYSTEM PROMPT WITH RULES
prompt_template = """You are a helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Use ONLY information from the context below to answer.
2. If the answer is not in the context, say: "I don't have that information in the provided documents."
3. Do NOT make up facts, numbers, or details.
4. Keep answers concise - 3 to 5 sentences max.
5. If asked about something unrelated to the documents, say: "I can only answer questions about your uploaded documents."

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False
)

print("RAG Chatbot ready! Ask questions about your PDFs. Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    answer = qa.invoke(query)
    print(f"\nBot: {answer['result']}\n")