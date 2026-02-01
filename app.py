import gradio as gr
from dotenv import load_dotenv
from rag_setup import RAG_Setup
from tools import MedicalTools
from graph_setup import GraphSetup
from chat_handler import ChatHandler
from audio_handler import AudioHandler
from user_data import initialize_db, add_user, create_session
import uuid

load_dotenv(override=True)
initialize_db()

rag = RAG_Setup()
medical_tools = MedicalTools(rag)
tools = medical_tools.get_tools()
graph_setup = GraphSetup(tools)
graph = graph_setup.get_graph()
chat_handler = ChatHandler(graph, rag)
audio_handler = AudioHandler()


def transcribe_audio_wrapper(audio, current_text, file_input, message_history, user_state, session_state):
    if not user_state or not session_state:
        warning = {
            "role": "assistant",
            "content": "Please log in and start a session before using voice input."
        }
        return message_history + [warning], current_text, None, file_input, user_state, session_state

    return audio_handler.transcribe_audio(
        audio, 
        current_text, 
        file_input, 
        message_history, 
        user_state,
        session_state,
        chat_handler.chat
    )

def handle_login(user_identifier):
    if not user_identifier or not user_identifier.strip():
        return (
            "Please enter a user name or email.",
            None,
            None
        )

    user_id = user_identifier.strip().lower()
    session_id = str(uuid.uuid4())

    add_user(user_id, user_identifier)
    create_session(user_id, session_id)

    session_md = f"**Active user:** {user_id}<br>**Session:** {session_id}"

    return (
        session_md,
        {"user_id": user_id, "name": user_identifier},
        {"session_id": session_id}
    )


def handle_logout():
    return "No active session.", None, None

with gr.Blocks(title="Medical Assistant") as demo:
    user_state = gr.State(value=None)
    session_state = gr.State(value=None)

    gr.Markdown("# 🏥 Medical Assistant")
    gr.Markdown("Ask questions using text, voice, or upload medical documents")
    
    with gr.Accordion("User Login", open=True):
        user_input = gr.Textbox(label="Enter email or username", placeholder="name@example.com")
        with gr.Row():
            login_button = gr.Button("Start Session", variant="primary")
            logout_button = gr.Button("End Session", variant="stop")
        session_display = gr.Markdown("No active session.")
    
    login_button.click(
        handle_login,
        inputs=[user_input],
        outputs=[session_display, user_state, session_state],
    )

    logout_button.click(
        handle_logout,
        outputs=[session_display, user_state, session_state],
    )

    chatbot = gr.Chatbot(label="Conversation", height=400)
    
    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                placeholder="Type your medical question here...",
                label="Text Input",
                lines=2
            )
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Voice"
            )
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📄 Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.ClearButton([chatbot, text_input, audio_input, file_input])
    
    gr.Markdown("### Tips:\n- Upload medical records (PDFs) and I'll process them automatically\n- Ask about medications, interactions, or symptoms\n- I can store new medical information you share")
    
    submit_btn.click(
        chat_handler.chat,
        inputs=[text_input, file_input, chatbot, user_state, session_state],
        outputs=[chatbot, text_input, file_input],
    )
    
    text_input.submit(
        chat_handler.chat,
        inputs=[text_input, file_input, chatbot, user_state, session_state],
        outputs=[chatbot, text_input, file_input]
    )

    audio_input.change(
        transcribe_audio_wrapper,
        inputs=[audio_input, text_input, file_input, chatbot, user_state, session_state],
        outputs=[chatbot, text_input, audio_input, file_input] 
    )

if __name__ == "__main__":
    demo.launch(share=True)