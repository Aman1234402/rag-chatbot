import streamlit as st
from langchain_core.messages import HumanMessage

import backend

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")


@st.cache_resource
def get_chatbot():
    """Refresh the PDF index once, then reuse the chatbot across reruns."""
    backend.ingest_pdf()
    return backend.create_chatbot()


chatbot = get_chatbot()

st.title("PDF RAG Chatbot")
st.caption(f"Ask questions about `{backend.PDF_PATH.name}`. Answers use only the PDF.")

with st.sidebar:
    st.subheader("Document")
    st.write(backend.PDF_PATH.name)
    st.write(f"Embedding model: `{backend.EMBED_MODEL}`")
    st.write(f"Answer model: `{backend.GROQ_MODEL}`")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["message_history"] = []
        st.rerun()

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


def render_extras(message: dict):
    """Show confidence + retrieved chunks under an assistant message."""
    confidence = message.get("confidence")
    context = message.get("context") or []

    if not confidence or not context:
        return

    st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.2f}")

    with st.expander(f"Retrieved context ({len(context)} chunks)"):
        for c in context:
            st.markdown(
                f"**Page {c['page']}** · chunk `{c['chunk_id']}` · score `{c['score']:.3f}`"
            )
            st.write(c["text"])
            st.divider()


# replay history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_extras(message)


user_input = st.chat_input("Ask a question about the PDF")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the PDF and writing an answer..."):
                response = chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                )

            answer = response["messages"][-1].content
            context = response.get("context", [])
            confidence = response.get("confidence", 0.0)
        except Exception as error:
            st.error(f"Unable to answer this query: {error}")
            st.stop()

        st.markdown(answer)

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "context": context,
            "confidence": confidence,
        }
        render_extras(assistant_message)

    st.session_state["message_history"].append(assistant_message)