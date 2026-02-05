import streamlit as st
import os
import ssl
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector

# --- 1. SSL BYPASS ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# --- 2. CONFIGURATION ---
CONNECTION_STRING = "postgresql+psycopg://user:password@localhost:5433/rag_db"
COLLECTION_NAME = "my_docs_collection"

st.set_page_config(page_title="JDE Support Expert", page_icon="🏗️", layout="wide")
st.title("🏗️ JDE Production Support Expert")
st.markdown("---")

# --- 3. RESOURCES ---
@st.cache_resource
def load_resources():
    return {
        "llm": ChatOllama(model="llama3.2", temperature=0),
        "embeddings": OllamaEmbeddings(model="bge-m3"),
        "vs": PGVector(
            connection=CONNECTION_STRING,
            embeddings=OllamaEmbeddings(model="bge-m3"),
            collection_name=COLLECTION_NAME,
        )
    }

res = load_resources()

# --- 4. THE MASTER PROMPT (NO HISTORY) ---
MASTER_PROMPT = """
### ROLE
You are a JD Edwards (JDE) Production Support Expert. Your goal is to provide technical resolutions for Sales Order Management issues based ONLY on the provided documentation.

### TECHNICAL GUIDELINES
- Always include Application IDs (e.g., P4210), Versions (e.g., ZJDE0001), and Table names (e.g., F4211).
- Prioritize specific versions mentioned in the documents.
- Maintain a professional, technical, and concise tone.

### FORMAT REQUIREMENTS
1. **Analysis**: A brief (1-2 sentence) explanation of the root cause.
2. **Resolution**: A numbered, step-by-step guide to fixing the issue.

---

### CONTEXT DATA
{context_docs}

### CURRENT USER QUERY
{user_query}

### ASSISTANT RESPONSE
"""

# --- 5. UI DISPLAY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. DYNAMIC INTERACTION LOGIC ---
if user_input := st.chat_input("Describe the JDE issue..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Intent Check to handle greetings naturally
        intent_prompt = f"System: Is the user asking a technical JDE question or just greeting? Answer 'TECHNICAL' or 'CHAT'.\nUser: {user_input}"
        intent = res["llm"].invoke(intent_prompt).content.strip().upper()

        response_placeholder = st.empty()
        full_response = ""

        if "CHAT" in intent:
            greet_prompt = f"You are a JDE Expert. Respond briefly to: '{user_input}'. Ask how you can help with Sales Order issues."
            for chunk in res["llm"].stream(greet_prompt):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
        
        else:
            with st.spinner("Searching Sales Order Documentation..."):
                # Retrieve Docs
                docs = res["vs"].similarity_search(user_input, k=3)
                context_docs = "\n\n".join([d.page_content for d in docs])
                
                # Format Prompt without history
                final_query = MASTER_PROMPT.format(
                    context_docs=context_docs,
                    user_query=user_input
                )

                # Stream Technical Response
                for chunk in res["llm"].stream(final_query):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- 7. UI CONTROLS ---
with st.sidebar:
    st.header("Settings")
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.rerun()