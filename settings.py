import os
import streamlit as st

# Gemini secrets
ENGINE_ADA = "text-embedding-ata-002"
GPT_DEFAULT = "4"
USE_GEMINI = st.secrets.get("USE_GEMINI", False)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = st.secrets.get("GEMINI_CHAT_MODEL", "")
GEMINI_EMBEDDING_MODEL = st.secrets.get("GEMINI_EMBEDDING_MODEL", "")
