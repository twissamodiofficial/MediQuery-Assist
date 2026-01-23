import gradio as gr
from dotenv import load_dotenv
from rag_setup import RAG_Setup
from tools import MedicalTools
from graph_setup import GraphSetup
from chat_handler import ChatHandler
from audio_handler import AudioHandler


load_dotenv(override=True)

rag = RAG_Setup()
medical_tools = MedicalTools(rag)
tools = medical_tools.get_tools()
graph_setup = GraphSetup(tools)
graph = graph_setup.get_graph()
chat_handler = ChatHandler(graph, rag)
audio_handler = AudioHandler()


def transcribe_audio_wrapper(audio, current_text, file_input, message_history):
    return audio_handler.transcribe_audio(
        audio, 
        current_text, 
        file_input, 
        message_history, 
        chat_handler.chat
    )


with gr.Blocks(title="Medical Assistant") as demo:
    gr.Markdown("# 🏥 Medical Assistant")
    gr.Markdown("Ask questions using text, voice, or upload medical documents")
    
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
        inputs=[text_input, file_input, chatbot],
        outputs=[chatbot, text_input, file_input]
    )
    
    text_input.submit(
        chat_handler.chat,
        inputs=[text_input, file_input, chatbot],
        outputs=[chatbot, text_input, file_input]
    )

    audio_input.change(
        transcribe_audio_wrapper,
        inputs=[audio_input, text_input, file_input, chatbot],
        outputs=[chatbot, text_input, audio_input, file_input] 
    )


if __name__ == "__main__":
    demo.launch(share=True)