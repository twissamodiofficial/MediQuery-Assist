from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper


class MedicalTools:
    def __init__(self, rag_setup):
        self.rag = rag_setup
        self.serper = GoogleSerperAPIWrapper()
        
    def get_tools(self):
        @tool
        def check_medical_history(query: str):
            '''Retrieves relevent medical history of the user

            Args:
                query: medical history to be searched for
            '''
            return self.rag.retrieve_info(query)
        
        @tool
        def web_search(query: str):
            ''' Search web for answering queries with latest information
            Args:
                query: query to be searched on the web
            '''
            print("Websearch tool calling")
            return self.serper.run(query)
        
        return [web_search, check_medical_history]