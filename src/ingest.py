import os
import pdfplumber
import pytesseract
from PIL import Image
from typing import List, Tuple, Union
import io

def extract_text_from_pdf(file_stream) -> str:
    """Extracts text from a PDF file stream."""
    text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    # Fallback to OCR for uploaded stream
                    try:
                        im = page.to_image(resolution=300).original
                        text += pytesseract.image_to_string(im) + "\n"
                    except Exception as e:
                        print(f"OCR failed for uploaded PDF page: {e}")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text

def extract_text_from_image(file_stream) -> str:
    """Extracts text from an image file stream using OCR."""
    try:
        image = Image.open(file_stream)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing image OCR: {e}")
        return ""

def extract_text_from_txt(file_stream) -> str:
    """Extracts text from a text file stream."""
    try:
        # Check if it's bytes or string
        if isinstance(file_stream, io.IOBase):
             content = file_stream.read()
             if isinstance(content, bytes):
                 return content.decode("utf-8")
             return content
        return ""
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""

def process_file(file_input, filename: str) -> str:
    """
    Generic processing function for both Streamlit uploads and local files.
    file_input: Can be a file path (str) or a file-like object.
    """
    file_type = filename.split('.')[-1].lower()
    text = ""
    
    # If file_input is a path string, open it
    if isinstance(file_input, str):
        if not os.path.exists(file_input):
            return ""
        
        if file_type == 'pdf':
            # pdfplumber can open paths directly
            with pdfplumber.open(file_input) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t: 
                        text += t + "\n"
                    else:
                        # Fallback to OCR if text extraction fails (scanned PDF)
                        # Convert page to image
                        try:
                            im = page.to_image(resolution=300).original
                            text += pytesseract.image_to_string(im) + "\n"
                        except Exception as e:
                            print(f"OCR failed for page in {file_input}: {e}")
        elif file_type in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            image = Image.open(file_input)
            text = pytesseract.image_to_string(image)
        elif file_type == 'txt':
            with open(file_input, 'r', encoding='utf-8') as f:
                text = f.read()
                
    # If file_input is a file-like object (Streamlit)
    else:
        if file_type == 'pdf':
            text = extract_text_from_pdf(file_input)
        elif file_type in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            text = extract_text_from_image(file_input)
        elif file_type == 'txt':
            text = extract_text_from_txt(file_input)
            
    return text

def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """
    Wrapper for Streamlit uploads.
    """
    filename = uploaded_file.name
    text = process_file(uploaded_file, filename)
    return filename, text

def process_local_file(file_path: str) -> Tuple[str, str]:
    """
    Wrapper for local files.
    """
    filename = os.path.basename(file_path)
    text = process_file(file_path, filename)
    return filename, text
