from typing import List, Dict
from src.vector_store import VectorStoreManager
from src.llm import LLMEngine
from src.prompts import construct_rag_prompt

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.llm_engine = LLMEngine()
        
    def get_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieves relevant chunks from the vector store."""
        docs = self.vector_store.query_similarity(query, k=k)
        return [doc.page_content for doc in docs]

    def construct_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Constructs the prompt for Phi-3."""
        context_text = "\n\n".join(context_chunks)
        return construct_rag_prompt(query, context_text)

    def answer_question(self, query: str, k: int = 5, source_filter: List[str] = None) -> Dict:
        """
        End-to-end RAG pipeline: Retrieve -> Generate.
        Returns dictionary with answer and source context.
        source_filter: Optional list of filenames to restrict search to
        """
        # 1. Retrieve
        if source_filter:
            docs = self.vector_store.query_similarity_filtered(query, source_filter=source_filter, k=k)
        else:
            docs = self.vector_store.query_similarity(query, k=k)
        context_chunks = [doc.page_content for doc in docs]
        sources = list(set([doc.metadata.get('source', 'unknown') for doc in docs]))
        
        if not context_chunks:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": []
            }

        # 2. Construct Prompt
        prompt = self.construct_prompt(query, context_chunks)
        
        # 3. Generate
        try:
            response = self.llm_engine.generate_response(prompt)
        except FileNotFoundError:
            return {
                "answer": "Error: Model file not found. Please check the 'models' directory.",
                "sources": []
            }
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": []
            }
            
        return {
            "answer": response,
            "sources": sources,
            "context": context_chunks # Optional: return context for debugging
        }

    def answer_question_stream(self, query: str, k: int = 5, source_filter: List[str] = None):
        """
        Streams the answer. Returns (generator, sources).
        source_filter: Optional list of filenames to restrict search to
        """
        # 1. Retrieve
        if source_filter:
            docs = self.vector_store.query_similarity_filtered(query, source_filter=source_filter, k=k)
        else:
            docs = self.vector_store.query_similarity(query, k=k)
        context_chunks = [doc.page_content for doc in docs]
        sources = list(set([doc.metadata.get('source', 'unknown') for doc in docs]))
        
        if not context_chunks:
            # Return a dummy generator
            def empty_gen():
                yield {'choices': [{'text': "No relevant documents found in the knowledge base."}]}
            return empty_gen(), []

        # 2. Construct Prompt
        prompt = self.construct_prompt(query, context_chunks)
        
        # 3. Generate Stream
        try:
            stream = self.llm_engine.generate_response(prompt, stream=True)
            return stream, sources
        except Exception as e:
            def error_gen():
                yield {'choices': [{'text': f"Error: {str(e)}"}]}
            return error_gen(), []
