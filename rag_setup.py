import hashlib
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from document_classifier import DocumentClassifier


class RAG_Setup:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = Chroma(
            collection_name="medical_history_collection",
            embedding_function=self.embeddings,
            persist_directory="data/patient_record_db", 
        )
        
        self.classifier = DocumentClassifier(
            pages_per_group=2,
            min_confidence=0.35,
            max_workers=4,
            model_name="cross-encoder/nli-deberta-v3-small"
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

    def store_data(self, file_path, user_id=None):
        file_hash = self._calculate_file_hash(file_path)
        
        if self._is_file_uploaded(file_hash):
            return {
                "status": "skipped",
                "message": f"File already exists in database"
            }
        
        try:
            print(f"[RAG] Classifying document...")
            classification = self.classifier.classify_document(file_path)
            page_map = classification['page_classifications']
            
            print(f"[RAG] Primary type: {classification['primary_type']}")
            print(f"[RAG] Found {len(classification['all_types'])} document types "
                  f"across {classification['total_pages']} pages in {classification['processing_time']}s")
            if len(classification['all_types']) > 1:
                print(f"[RAG] All types: {', '.join(classification['all_types'])}")
            
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            all_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100,
                add_start_index=True
            )
            
            for i, page in enumerate(pages):
                page_num = i + 1
                
                page_class = page_map.get(page_num, {'type': 'other', 'confidence': 0.0})
                page_chunks = text_splitter.split_documents([page])
                
                for chunk in page_chunks:
                    chunk.metadata.update({
                        'file_hash': file_hash,
                        'page_number': page_num,
                        'doc_type': page_class['type'],
                        'classification_confidence': page_class['confidence'],
                        'primary_doc_type': classification['primary_type'],
                        'all_doc_types': ','.join(classification['all_types'])
                    })
                    if user_id:
                        chunk.metadata['user_id'] = user_id
                
                all_chunks.extend(page_chunks)
            
            self._embed_content(all_chunks)
            
            print(f"[RAG] Stored {len(all_chunks)} chunks with page-specific labels")
            
            return {
                "status": "success",
                "message": f"File successfully uploaded",
                "chunks": len(all_chunks),
                "primary_type": classification['primary_type'],
                "all_types": classification['all_types'],
                "processing_time": classification['processing_time']
            }
        except Exception as e:
            print(f"[RAG] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Failed to upload file: {str(e)}"
            }

    def retrieve_info(self, user_id:str, query: str):
        try:
            print(f"[RAG] Retrieving for user_id: {user_id}, query: {query}")
            results = self.vector_store.similarity_search(query, k=5,  filter={"user_id": user_id})
            print(f"[RAG] Found {len(results)} results")
            
            if not results:
                return "No medical history found for this query."
            
            content = "\n\n---DOCUMENT---\n\n".join([doc.page_content for doc in results])
            
            return content
        
        except Exception as e:
            print(f"[RAG] Error retrieving medical record: {str(e)}")
            return f"Failed to retrieve medical record: {str(e)}"