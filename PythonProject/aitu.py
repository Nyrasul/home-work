import pypdf
import streamlit as st
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
import warnings
import logging
import shutil

# Setup logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_pdf_text(file):
    try:
        reader = pypdf.PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    except Exception as e:
        logging.error(f"PDF Reading Error: {type(e).__name__}: {e}")
        st.error(f"PDF Reading Error: {type(e).__name__}: {e}")
        return None

def get_chroma_db(_embeddings, texts, collection_name):
    persist_directory = "db"
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)  # Clear existing DB before each run
            logging.info(f"Deleted existing db at {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)

        logging.info("Creating Chroma client...")
        client = chromadb.Client(chromadb.Settings(persist_directory=persist_directory, anonymized_telemetry=False))
        logging.info("Chroma client created.")
        client.heartbeat()
        logging.info("Chroma heartbeat successful.")

        collections = client.list_collections()
        if any(c.name == collection_name for c in collections):
            logging.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(name=collection_name)
            logging.info(f"Collection {collection_name} deleted.")

        if texts:
            logging.info(f"Creating collection {collection_name} with {len(texts)} texts")
            Chroma.from_texts(texts, _embeddings, collection_name=collection_name, client=client)
            logging.info(f"Collection '{collection_name}' created.")
        else:
            st.warning("No text to add to Chroma DB")
            return None

        docsearch = Chroma(client=client, collection_name=collection_name, embedding_function=_embeddings)
        return docsearch

    except Exception as e:
        logging.error(f"ChromaDB Error: {type(e).__name__}: {e}")
        st.error(f"ChromaDB Error: {type(e).__name__}: {e}")
        return None

# Initialize Streamlit session state
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "texts" not in st.session_state:
    st.session_state.texts = None
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit App Title
st.title("PDF Q&A with Ollama and Chroma")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            pdf_text = load_pdf_text(uploaded_file)
            if pdf_text is None:
                st.error("Failed to extract text from PDF.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
            texts = [text for text in text_splitter.split_text(pdf_text) if text.strip() and len(text) > 10]
            texts = [re.sub(r'[^\w\s]', '', text).strip() for text in texts]

            if not texts:
                st.error("No valid text extracted from PDF.")
                st.stop()

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    embeddings.embed_documents(batch)
                    st.write(f"Processed batch {i // batch_size + 1}/{int(len(texts) / batch_size + 1)}")
                except Exception as e:
                    logging.error(f"Embedding Error (Batch {i // batch_size + 1}): {type(e).__name__}: {e}")
                    st.error(f"Embedding Error (Batch {i // batch_size + 1}): {type(e).__name__}: {e}")
                    st.stop()

            docsearch = get_chroma_db(embeddings, texts, "pdf_collection")

            if not docsearch:
                st.error("Failed to initialize Chroma DB.")
                st.stop()

            # Save data in session state
            st.session_state.pdf_uploaded = True
            st.session_state.texts = texts
            st.session_state.docsearch = docsearch

            st.success("PDF processed successfully! You can now ask questions based on the uploaded document.")

        except Exception as e:
            logging.error(f"File Processing Error: {type(e).__name__}: {e}")
            st.error(f"File Processing Error: {type(e).__name__}: {e}")
            st.stop()

else:
    st.info("Upload a PDF file to enable PDF-based Q&A.")

# Initialize Ollama LLM
try:
    ollama_model = Ollama(model="llama3.2")
except Exception as e:
    logging.error(f"Ollama Initialization Error: {type(e).__name__}: {e}")
    st.error(f"Ollama Initialization Error: {type(e).__name__}: {e}")
    st.stop()

# Chat interaction
if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                use_context = st.session_state.pdf_uploaded and st.session_state.docsearch is not None and any(
                    word in prompt.lower() for word in ["what", "who", "when", "where", "why", "how", "summarize", "explain", "list", "describe"]
                )
                if use_context:
                    retrieved_docs = st.session_state.docsearch.similarity_search(prompt, k=3)
                    context = "\n".join([doc.page_content for doc in retrieved_docs])

                    prompt_with_context = f"""Use the following context to answer the question at the end. If the answer cannot be found within the context, respond with "I don't know based on the context provided.".

                    Context:
                    {context}

                    Question:
                    {prompt}

                    Answer:
                    """
                    answer = ollama_model(prompt_with_context)
                else:
                    answer = ollama_model(prompt)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                logging.error(f"LLM Processing Error: {type(e).__name__}: {e}")
                st.error(f"LLM Processing Error: {type(e).__name__}: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})

