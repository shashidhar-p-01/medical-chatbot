STEPS FOLLOWED :
created project folder 
initialized empty git repo
setting python version $ uv python install 3.13
$ uv venv --python 3.13
$ uv init --python 3.13
$ uv add python-dotenv
created github repo and added it as origin 
added langchain and langchain-community $ uv add langchain langchain-community
added langchain huggingface llm connecter $ uv add langchain-huggingface
added langchain groq $ uv add langchain-groq
added sentence transformers and faiss-cpu $ uv add sentence-transformers faiss-cpu
added streamlit $ uv add streamlit
created .venv and added huddingface and groq api keys 
created .gitignore to hide secrets from being pushed to github 
created data/ directory and added the medical encyclopedia pdf 
created a jupyter notebook to test and document each step for create_memory_for_llm.py
    creating_memory_for_llm.ipynb
    - import environment variables 
    - loading the pdf , extracting pdf (got an import error , so i installed pypdf )
    - chunking 
    - embeddings (got an module not found error for langchain_huggingface , so i did pip install in jupyter notebook)
    - building and saving FAISS index (creates the vectorstore\db_faiss)
the creating memory for llm is for ingestion phase , we only run it once 
now we create a notebook for retrieval phase - retrieval_qa.py
    retrieval_qa.ipynb
    - load secrets (environment variables )
    - load the saved faiss index from the disk 
    - bring in the LLM ( initializing the LLM )
    - retrieval + generation 
    - tested if its working fine 

problems faced 
- the PyPDFLoader function , needs pypdf installed to run , which was not downloaded , it raised an import error and module not found error 
- the jupyter notebook keeps on throwing module not found error again and again , then i realised that i installed the libraries in the uv environment , and opened the ipynb file in global environment , this was causeing it , later i fixed it by opening the jupyter notebook from the venv ($ uv run python -m jupyter notebook)
- during the retrieval and generation i was importing hub from langchain_community , but it didnt work , so i used langchain_core.prompts , .runnables , .output_parsers -- using the help of AI , i still dont understand them completely but i know what they are doing 