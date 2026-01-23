import hashlib
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class RAG_Setup:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = Chroma(
            collection_name="medical_history_collection",
            embedding_function=self.embeddings,
            persist_directory="data/patient_record_db", 
        )

    def _calculate_file_hash(self, file_path):
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _is_file_uploaded(self, file_hash):
        results = self.vector_store.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(results['ids']) > 0
    
    def _extract_content(self, file_path):
        pdf_loader = PyPDFLoader(file_path)
        content = pdf_loader.load()
        return content

    def _split_content(self, content):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, add_start_index=True)
        chunks = text_splitter.split_documents(content)
        return chunks

    def _embed_content(self, chunks):
        self.vector_store.add_documents(chunks)

    def store_data(self, file_path):
        file_hash = self._calculate_file_hash(file_path)
        
        if self._is_file_uploaded(file_hash):
            return {
                "status": "skipped",
                "message": f"File already exists in database"
            }
        
        try:
            content = self._extract_content(file_path)
            chunks = self._split_content(content)
            
            for chunk in chunks:
                chunk.metadata.update({
                    'file_hash': file_hash
                })
            
            self._embed_content(chunks)
            
            return {
                "status": "success",
                "message": f"File successfully uploaded",
                "chunks": len(chunks)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to upload file: {str(e)}"
            }

    def retrieve_info(self, query: str):
        try:
            results = self.vector_store.similarity_search(query, k=5)
            print("printing tool results", results)
            
            if not results:
                return "No medical history found for this query."
            
            content = "\n\n---DOCUMENT---\n\n".join([doc.page_content for doc in results])
            
            return content
        
        except Exception as e:
            return "Failed to retrieve medical record"