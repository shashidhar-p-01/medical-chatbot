import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# laod secrets

load_dotenv()

# load the saved faiss index from the disk


DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  # the FAISS index only stores vectors,
# not the model that produced them. To later embed a new user question into the same vector space (so it's comparable), we need the
# identical embedding model loaded again. If you used a different model here than during ingestion, the vectors wouldn't be comparable at all

print("loading FAISS model")
db = FAISS.load_local(
    DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
)  # allow_dangerous_deserialization - Pickle can
# execute arbitrary code if loaded from an untrusted source, so LangChain requires you to explicitly opt in with this flag

print("FAISS index loaded successfully")


# bringing in the LLM


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL_NAME,
    temperature=0.5,  # control randomness
    max_tokens=512,  # maximum length of the generated answer
)

print("LLM initiated successfully", llm)


# retrieval + generation


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
Answer in a clear, complete, and well-explained way using only the information given in the context.
Do not just copy a single sentence — synthesize the relevant details into a full answer.
If the answer is not contained in the context, say you don't know — do not make anything up.

Context: {context}
Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

retriever = db.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# testing
response = rag_chain.invoke("What is cancer?")
print(response)
