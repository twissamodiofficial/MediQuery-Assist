from transformers import pipeline


class AudioHandler:
    def __init__(self):
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    
    def transcribe_audio(self, audio, current_text, file_input, message_history, chat_func):
        if audio is None:
            return message_history, current_text, None, file_input
        
        transcript = self.transcriber(audio)["text"].strip()
        
        updated_history, cleared_text, cleared_file = chat_func(
            transcript, 
            file_input, 
            message_history
        )
        
        return updated_history, current_text, None, cleared_file