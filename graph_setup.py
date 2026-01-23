import sqlite3
from typing_extensions import TypedDict, Annotated
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class State(TypedDict):
    messages: Annotated[list, add_messages]


class GraphSetup:
    def __init__(self, tools):
        self.tools = tools
        self.llm = self._setup_llm()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = self._setup_memory()
        self.graph = self._build_graph()
        
    def _setup_llm(self):
        llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-V3",
            task="text-generation",
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03,
            provider="auto", 
        )
        return ChatHuggingFace(llm=llm)
    
    def _setup_memory(self):
        db_path = 'data/long_term_memory.db'
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    
    def _personal_assistant(self, state: State):
        print("assistant responses:")
        print(state["messages"])
        messages = state["messages"]
        return {
            "messages": self.llm_with_tools.invoke(messages)
        }
    
    def _build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("personal_assistant", self._personal_assistant)
        graph_builder.add_node("tools", ToolNode(self.tools))
        graph_builder.add_conditional_edges("personal_assistant", tools_condition, {"tools": "tools", "__end__": END})
        graph_builder.add_edge(START, "personal_assistant")
        graph_builder.add_edge("tools", "personal_assistant")
        
        return graph_builder.compile(checkpointer=self.memory)
    
    def get_graph(self):
        return self.graph