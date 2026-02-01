import json
import uuid
from langgraph.errors import GraphRecursionError
from prompts import REACT_SYSTEM_PROMPT


class ChatHandler:
    def __init__(self, graph, rag_setup):
        self.graph = graph
        self.rag = rag_setup
        self.session_id = str(uuid.uuid4())
        print(self.session_id)
    
    def chat(self, user_message, uploaded_file, message_history, user_state, session_state):
        if not user_state or not session_state:
            warning = {
                "role": "assistant",
                "content": "Please log in and start a session before chatting."
            }
            return message_history + [warning], user_message, uploaded_file, user_state, session_state
        
        user_query_parts = []
        try: 
            if user_message and user_message.strip():
                user_query_parts.append(user_message)
            
            if uploaded_file is not None:
                result = self.rag.store_data(uploaded_file)
                result_str = json.dumps(result, indent=2)
                user_query_parts.append(f"""A medical document was uploaded. Here are the upload details: {result_str} Please inform the user about the upload status in a friendly, professional way.""")

            if not user_query_parts:
                return message_history, "", None, None
            
            user_query = (' ').join(user_query_parts)
            
            config = {"configurable": {"thread_id": self.session_id}, "recursion_limit" : 25}
            current_state = self.graph.get_state(config)
            
            if not current_state.values.get("messages"):
                messages = {
                    "messages": [
                        {"role": "system", "content": REACT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_query}
                    ]
                }
            else:
                messages = {"messages": [{"role": "user", "content": user_query}]}

            result = self.graph.invoke(
                messages,
                config=config
            )
            
            last_message = result["messages"][-1].content
            
            updated_history = message_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": last_message}
            ]
            
            return updated_history, "", None
            
        except GraphRecursionError:
            error_message = "This query is too complex and exceeded the reasoning limit. Please simplify or break it into smaller questions."
            return message_history + [
                {"role": "assistant", "content": error_message}
            ], "", None
           
        except Exception as e:
            error_message = f"Error: {str(e)}"
            return message_history + [
                {"role": "assistant", "content": error_message}
            ], "", None