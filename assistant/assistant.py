
import streamlit as st
from google import genai

from assistant.prompts import SYSTEM_PROMPT


MODEL_NAME = "gemini-3.8-flash"


@st.cache_resource
def get_gemini_client():
    """Create and cache the Gemini API client."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing from .streamlit/secrets.toml"
        )

    return genai.Client(api_key=api_key)


def get_assistant_response(question, analysis_context, chat_history):
    """Generate a response using Gemini and the current MRI context."""
    client = get_gemini_client()

    context = analysis_context or (
        "No MRI analysis results have been provided in this conversation."
    )

    contents = []

    # Supply the current analysis context to the assistant.
    contents.append({
        "role": "user",
        "parts": [{
            "text": (
                "CURRENT APPLICATION ANALYSIS CONTEXT:\n"
                f"{context}\n\n"
                "Use this information only when answering questions "
                "about the user's current analysis."
                )
        }]
    })

    contents.append({
        "role": "model",
        "parts": [{
            "text": (
                "I will use the supplied application results as context. "
                "I will not invent missing results or treat predictions "
                "as a medical diagnosis."
            )
        }]
    })

    # Include earlier messages for follow-up questions.
    for message in chat_history:
        contents.append({
            "role": (
                "user" if message["role"] == "user" else "model"
            ),
            "parts": [{"text": message["content"]}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.4,
        },
    )

    answer = response.text

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    return answer


def render_assistant(analysis_context=""):
    """Display the CereVexa AI chat interface in Streamlit."""
    st.subheader("💬 CereVexa AI Assistant")
    st.caption(
        "Ask about your model results, Grad-CAM heatmap, "
        "segmentation, or brain MRI concepts."
    )

    if "cerevexa_chat" not in st.session_state:
        st.session_state.cerevexa_chat = []

    with st.expander("Suggested questions"):
        st.markdown(
            "- What does my model's confidence score mean?\n"
            "- How should I interpret the Grad-CAM heatmap?\n"
            "- What does a segmentation mask show?\n"
            "- What are the limitations of AI-based MRI analysis?"
        )

    for message in st.session_state.cerevexa_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask CereVexa AI a question..."
    )

    if question:
        st.session_state.cerevexa_chat.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("CereVexa is thinking..."):
                    answer = get_assistant_response(
                        question=question,
                        analysis_context=analysis_context,
                        chat_history=st.session_state.cerevexa_chat[:-1],
                    )

                st.markdown(answer)

                st.session_state.cerevexa_chat.append({
                    "role": "assistant",
                    "content": answer,
                })

            except Exception as exc:
                st.error(f"Gemini error: {type(exc).__name__}: {exc}")
                print(f"CereVexa assistant error: {type(exc).__name__}: {exc}")

    if st.button("Clear chat", key="clear_cerevexa_chat"):
        st.session_state.cerevexa_chat = []
        st.rerun()

