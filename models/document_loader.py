# models/document_loader.py
import os
import PyPDF2

DATA_DIR = "data"

def load_pdf(file_name: str):
    """Extract text from a PDF file in the data folder"""
    path = os.path.join(DATA_DIR, file_name)
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    return " ".join(text.split())

def load_and_clean(file_name: str):
    text = load_pdf(file_name)
    return clean_text(text)
