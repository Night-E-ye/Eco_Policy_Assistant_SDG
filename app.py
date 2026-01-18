import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ibm import WatsonxLLM

# Page Setup
st.set_page_config(page_title="Eco-Policy Assistant", page_icon="🌱")
st.title("🌱 Eco-Policy Assistant")
st.markdown("### Ask questions about India's Sustainability Landscape")

load_dotenv()

# --- 1. RAG Setup (PDF Processing) ---
@st.cache_resource # Caches the PDF data so it doesn't reload every time you ask a question
def get_vectorstore():
    loader = PyPDFLoader("sustainability_policy.pdf")
    data = loader.load()
    # Smaller chunks help the AI find specific rules more accurately
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = get_vectorstore()

# --- 2. IBM Granite Initialization ---
llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct", # Using the latest 3.0 model for better reasoning
    url=os.getenv("WATSONX_URL"),
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("PROJECT_ID"),
    params={"max_new_tokens": 500, "temperature": 0.1} # Lower temperature for factual accuracy
)

# --- 3. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Eco-Policy India"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Bot Logic
    with st.chat_message("assistant"):
        # SEARCH: Find relevant parts of the PDF
        docs = vectorstore.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # AGENTIC PROMPT: Define the AI's persona to avoid safety refusals
        system_prompt = f"""
        You are a helpful Sustainability Policy Expert. Your goal is to help users understand environmental rules.
        Use only the following context from the 'Sustainability reporting landscape in India' report to answer the question.
        If the information is not in the context, say you don't know based on this document.
        
        Context: {context}
        
        User Question: {user_query}
        Assistant Answer:"""
        
        with st.spinner("Consulting Policy Documents..."):
            response = llm.invoke(system_prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})