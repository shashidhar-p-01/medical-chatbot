import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
Answer in a clear, complete, and well-explained way using only the information given in the context.
Do not just copy a single sentence — synthesize the relevant details into a full answer.
If the answer is not contained in the context, say you don't know — do not make anything up.

Context: {context}
Question: {question}

Answer:
"""

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.title("Ask Medibot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask a medical question here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            db = get_vectorstore()
            retriever = db.as_retriever(search_kwargs={'k': 3})

            llm = ChatGroq(
                api_key=os.environ.get("GROQ_API_KEY"),
                model="llama-3.1-8b-instant",
                temperature=0.5,
                max_tokens=512,
            )

            chat_prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | chat_prompt
                | llm
                | StrOutputParser()
            )

            response = rag_chain.invoke(prompt)

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()