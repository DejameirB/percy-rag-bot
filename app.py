import streamlit as st
from llm_with_rag import build_vector_store, chat, retrieve, documents, doc_ids

st.title("💬 Customer Support Chat")
st.write("This is ByteBot, your assistant for ByteWorks inquiries.")

@st.cache_resource
def get_collection():
    return build_vector_store(documents, doc_ids)

collection, vectorizer = get_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources used"):
                for doc in msg["sources"]:
                    st.write(f"- {doc}")

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_docs = retrieve(collection, vectorizer, question)
            answer = chat(collection, vectorizer, question, history=st.session_state.messages[:-1])
            
        st.write(answer)
        with st.expander("Sources used"):
            for doc in retrieved_docs:
                st.write(f"- {doc}")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": retrieved_docs
    })
