import streamlit as st
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.ingest import process_uploaded_file, process_local_file
from src.rag import RAGPipeline
from src.vector_store import VectorStoreManager
from src.config import MODELS_DIR, MODEL_NAME, DATA_DIR

# Page Config
st.set_page_config(
    page_title="Research Paper Knowledge Base",
    page_icon="📚",
    layout="wide"
)

# Initialize Session State
# Initialize vector store first
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

# Initialize RAG pipeline with the same vector store instance
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
    # Share the same vector store
    st.session_state.rag_pipeline.vector_store = st.session_state.vector_store

# Track files uploaded in this session
if "uploaded_files_this_session" not in st.session_state:
    st.session_state.uploaded_files_this_session = []

# Sidebar
st.sidebar.title("📚 Research RAG")
st.sidebar.markdown("---")

# API Key Input (only if not in .env)
if not os.getenv("GROQ_API_KEY"):
    api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get one for free at console.groq.com")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
else:
    st.sidebar.success("✅ Groq API Key loaded")

st.sidebar.markdown("---")

# Model Status Check (Skipped for Groq)
# model_path = os.path.join(MODELS_DIR, MODEL_NAME)
# model_exists = os.path.exists(model_path)
#
# if model_exists:
#     st.sidebar.success(f"✅ Local Model Found: {MODEL_NAME}")
# else:
#     st.sidebar.warning(f"⚠️ Local Model Not Found: {MODEL_NAME}")
#     st.sidebar.info("Using Groq API for inference.")

st.sidebar.markdown("---")
nav = st.sidebar.radio("Navigation", ["Chat & Query", "Upload Documents", "Manage Knowledge Base"])

# --- Upload Page ---
if nav == "Upload Documents":
    st.header("📤 Upload Research Papers")
    
    tab1, tab2 = st.tabs(["Drag & Drop", "Load from 'data' folder"])
    
    with tab1:
        st.markdown("Support formats: **PDF, TXT, Images (OCR)**")
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=['pdf', 'txt', 'png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )
        
        if st.button("Process & Ingest Uploads"):
            print(f"\n{'='*50}")
            print(f"UPLOAD BUTTON CLICKED")
            print(f"uploaded_files: {uploaded_files}")
            print(f"Number of files: {len(uploaded_files) if uploaded_files else 0}")
            print(f"{'='*50}\n")
            
            if not uploaded_files:
                st.warning("Please upload at least one file.")
                print("WARNING: No files uploaded")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_files = len(uploaded_files)
                processed_count = 0
                
                for file in uploaded_files:
                    try:
                        print(f"\n--- Processing file: {file.name} ---")
                        status_text.text(f"Processing {file.name}...")
                        
                        filename, text = process_uploaded_file(file)
                        print(f"Filename: {filename}")
                        print(f"Text length: {len(text) if text else 0}")
                        
                        if text:
                            st.info(f"Extracted {len(text)} characters from {filename}")
                            print(f"Adding to vector store...")
                            
                            chunks_added = st.session_state.vector_store.add_document(filename, text)
                            print(f"Chunks added: {chunks_added}")
                            
                            # Track this file as uploaded in this session
                            if filename not in st.session_state.uploaded_files_this_session:
                                st.session_state.uploaded_files_this_session.append(filename)
                            
                            st.success(f"✅ {filename}: Added {chunks_added} chunks to database")
                        else:
                            print(f"ERROR: No text extracted from {filename}")
                            st.error(f"❌ {filename}: No text extracted (empty file or OCR failed)")
                    except Exception as e:
                        print(f"EXCEPTION: {str(e)}")
                        st.error(f"❌ {filename}: Error - {str(e)}")
                        import traceback
                        traceback.print_exc()
                        st.code(traceback.format_exc())
                    
                    processed_count += 1
                    progress_bar.progress(processed_count / total_files)
                    
                print(f"\n{'='*50}")
                print(f"UPLOAD COMPLETE - Processed {total_files} files")
                print(f"{'='*50}\n")
                
                status_text.text("Ingestion Complete!")
                st.success(f"✅ Processed {total_files} files. Check 'Manage Knowledge Base' to verify.")
                st.balloons()

    with tab2:
        st.markdown(f"Files in `{DATA_DIR}`:")
        
        # List files in data dir
        try:
            local_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
            valid_extensions = ['.pdf', '.txt', '.png', '.jpg', '.jpeg']
            local_files = [f for f in local_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
        except Exception as e:
            st.error(f"Error accessing data directory: {e}")
            local_files = []
            
        if local_files:
            for f in local_files:
                st.text(f"📄 {f}")
            
            if st.button("Ingest All from Data Folder"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                processed_count = 0
                
                for f in local_files:
                    status_text.text(f"Processing {f}...")
                    file_path = os.path.join(DATA_DIR, f)
                    filename, text = process_local_file(file_path)
                    
                    if text:
                        chunks_added = st.session_state.vector_store.add_document(filename, text)
                        st.success(f"✅ {filename}: Added {chunks_added} chunks.")
                    else:
                        st.error(f"❌ {filename}: Failed to extract text.")
                        
                    processed_count += 1
                    progress_bar.progress(processed_count / len(local_files))
                
                status_text.text("Batch Ingestion Complete!")
                st.balloons()
        else:
            st.info("No supported files found in the 'data' folder.")

# --- Chat Page ---
elif nav == "Chat & Query":
    st.header("💬 Chat with your Papers")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show active file filter
    if st.session_state.uploaded_files_this_session:
        with st.expander(f"🎯 Querying ONLY: {len(st.session_state.uploaded_files_this_session)} recently uploaded file(s)", expanded=False):
            for f in st.session_state.uploaded_files_this_session:
                st.text(f"📄 {f}")
            if st.button("🗑️ Clear filter & Delete uploaded files", key="clear_filter", type="primary"):
                # Delete the files from database
                st.session_state.vector_store.delete_documents(st.session_state.uploaded_files_this_session)
                st.success(f"Deleted {len(st.session_state.uploaded_files_this_session)} file(s) from database")
                # Clear the session list
                st.session_state.uploaded_files_this_session = []
                st.rerun()

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Initialize stop flag
            if "stop_generation" not in st.session_state:
                st.session_state.stop_generation = False
            
            # Create columns for stop button
            col_response, col_stop = st.columns([10, 1])
            
            with col_stop:
                if st.button("⏹️", key=f"stop_{len(st.session_state.messages)}", help="Stop generation"):
                    st.session_state.stop_generation = True
            
            if st.session_state.stop_generation:
                st.warning("⏹️ Generation stopped by user")
                answer = "[Stopped by user]"
                st.session_state.stop_generation = False
            else:
                with col_response:
                    try:
                        print(f"\n{'='*50}")
                        print(f"QUERY: {prompt}")
                        print(f"Vector store instance: {id(st.session_state.rag_pipeline.vector_store)}")
                        print(f"Session vector store instance: {id(st.session_state.vector_store)}")
                        print(f"Session uploaded files: {st.session_state.uploaded_files_this_session}")
                        
                        # Use source filter if files were uploaded this session
                        source_filter = st.session_state.uploaded_files_this_session if st.session_state.uploaded_files_this_session else None
                        
                        # Use streaming method with filter
                        stream, sources = st.session_state.rag_pipeline.answer_question_stream(
                            prompt, 
                            source_filter=source_filter,
                            k=10  # Increase context window
                        )
                        
                        print(f"Sources found: {sources}")
                        print(f"{'='*50}\n")
                        
                        # Display sources first
                        if sources:
                            st.info(f"📚 Sources: {', '.join(sources)}")
                        else:
                            st.warning("No specific sources found.")

                        # Stream the response
                        placeholder = st.empty()
                        full_response = ""
                        
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}][LLM] Starting generation...")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] Prompt length: {len(prompt)} characters")
                        
                        # Iterate over the stream with stop check
                        token_count = 0
                        print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] Entering stream loop...")
                        
                        for i, chunk in enumerate(stream):
                            if i == 0:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] First chunk received: {type(chunk)}")
                            
                            # Check stop flag
                            if st.session_state.stop_generation:
                                st.session_state.stop_generation = False
                                full_response += "\n\n[Generation stopped]"
                                print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] Generation stopped by user after {token_count} tokens")
                                break
                            
                            # Handle Groq chunk format
                            text_chunk = ""
                            try:
                                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                                    delta = chunk.choices[0].delta
                                    if delta.content:
                                        text_chunk = delta.content
                                # Fallback for dictionary format
                                elif isinstance(chunk, dict) and 'choices' in chunk:
                                    if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                                        text_chunk = chunk['choices'][0]['delta']['content']
                            except Exception as e:
                                print(f"Error parsing chunk: {e}")
                                    
                            if text_chunk:
                                full_response += text_chunk
                                placeholder.markdown(full_response + "▌")
                                token_count += 1
                                
                                # Log progress every 20 tokens
                                if token_count % 20 == 0:
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] Generated {token_count} tokens...")
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] Generation complete!")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] Generated {token_count} tokens")
                        print(f"[{datetime.now().strftime('%H:%M:%S')}][LLM] Response length: {len(full_response)} characters")
                        print(f"{'='*50}\n")
                                
                        placeholder.markdown(full_response)
                        answer = full_response
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        answer = f"Error: {e}"
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Manage Page ---
elif nav == "Manage Knowledge Base":
    st.header("🗂️ Manage Knowledge Base")
    
    docs = st.session_state.vector_store.list_documents()
    
    st.subheader(f"Stored Documents ({len(docs)})")
    if docs:
        for doc in docs:
            st.text(f"📄 {doc}")
    else:
        st.info("No documents found in the database.")
        
    st.markdown("---")
    st.subheader("Danger Zone")
    
    if st.button("🗑️ Reset Database", type="primary"):
        st.session_state.vector_store.reset_db()
        st.success("Database has been reset!")
        st.experimental_rerun()
