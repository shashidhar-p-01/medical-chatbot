from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# imports the environment variables


load_dotenv()
print("loaded environment variables from .env file")


# LOADING THE PDF
# PyPDFLoader - extracts text from pdf , one page at a time
# DirectoryLoader - scans folder and finds everything matching the filetye


DATA_PATH = "data/"
print("loading pdf files ")


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


documents = load_pdf_files(DATA_PATH)
print("pdf files loaded ")

print(f"number of pages loaded : {len(documents)}")

# CHUNKING
# ResursiveCharacterTextSplitter - is "recursive" because it tries to cut at the best possible boundary: first tries paragraph
# breaks (\n\n), then line breaks (\n), then sentences, then words, only falling back to a hard character cut as a last resort


print("making chunks from loaded pdf ")


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # max ~500 characters per chunk (not words, not tokens — raw characters)
        chunk_overlap=50,  # the last 50 characters of one chunk get repeated as the first 50 characters of the next chunk.
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(documents)
print("chunks created")
print("number of chunks created : ", len(text_chunks))


# embeddings - turninf text into vectors

print("getting embedding model ")


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


embedding_model = get_embedding_model()


# build and save FAISS index


print("creating FAISS index ")

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("FAISS index created ")

print("FAISS index saved to : ", DB_FAISS_PATH)
