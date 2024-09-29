from transformers import AutoModel, AutoTokenizer,AutoProcessor
import streamlit as st
import os
from PIL import Image
import torch
from torchvision import io
import torchvision.transforms as transforms
import random
import easyocr
import numpy as np

def start():
    st.session_state.start = True

def reset():
    del st.session_state['start']

@st.cache_data
def get_text(image_file, _model, _tokenizer):
    res = _model.chat(_tokenizer, image_file, ocr_type='ocr')
    return res

@st.cache_data
def extract_text_easyocr(_image):
    try:
        reader = easyocr.Reader(['hi'], gpu=False)
        if reader is None:
            raise ValueError("Failed to create EasyOCR reader.")
        results = reader.readtext(np.array(_image))
        return " ".join([result[1] for result in results])
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""


@st.cache_resource
def model():
    tokenizer = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
    model = AutoModel.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval()
    return model, tokenizer

@st.cache_resource
def highlight_keywords(text, keywords):
    colors = generate_unique_colors(len(keywords))
    highlighted_text = text
    found_keywords = []
    for keyword, color in zip(keywords, colors):
        if keyword.lower() in text.lower():
            highlighted_text = highlighted_text.replace(keyword, f'<mark style="background-color: {color};">{keyword}</mark>')
            found_keywords.append(keyword)
    return highlighted_text, found_keywords

def search():
    st.session_state.search = True

@st.cache_data
def generate_unique_colors(n):
    colors = []
    for i in range(n):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        while color in colors:
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        colors.append(color)
    return colors


with st.sidebar:
    st.title("Instructions")

    st.write("1. Choose a language(English or Hindi)")
    st.write("2. Upload an image in JPG, PNG, or JPEG format.")
    st.write("3. The app will extract text from the image using OCR.")
    st.write("4. Enter keywords to search within the extracted text.")
    st.write("5. If needed, click 'Reset' to upload a new image/change language.")

st.title("A Web-Based Text Extraction and Retrieval System")

language = st.selectbox("Select a language:", ["English", "Hindi"])

if language == "English":
    st.subheader("You selected English!")
    st.button("Let's get started", on_click=start)
    
    if 'start' not in st.session_state:
        st.session_state.start = False
    
    if 'search' not in st.session_state:
        st.session_state.search = False
    
    if 'reset' not in st.session_state:
        st.session_state.reset = False
    
    if st.session_state.start:
        uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
        if uploaded_file is not None:
            st.subheader("Uploaded Image:")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
            MODEL, TOKENIZER = model()
    
            if not os.path.exists("images"):
                os.makedirs("images")
            with open(f"images/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
    
            extracted_text = get_text(f"images/{uploaded_file.name}", MODEL, TOKENIZER)
    
            st.subheader("Extracted Text")
            st.write(extracted_text)
    
            keywords_input = st.text_input("Enter keywords to search within the extracted text (comma-separated):")
    
            if keywords_input:
                keywords = [keyword.strip() for keyword in keywords_input.split(',')]
                highlighted_text, found_keywords = highlight_keywords(extracted_text, keywords)
                st.button("Search", on_click=search)
    
                if st.session_state.search:
                    st.subheader("Search Results:")
                    if found_keywords: 
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        st.write(f"Found keywords: {', '.join(found_keywords)}")
                    else:
                        st.warning("No keywords were found in the extracted text.")  
    
                    not_found_keywords = set(keywords) - set(found_keywords)
                    if not_found_keywords:
                        st.error(f"Keywords not found: {', '.join(not_found_keywords)}")
    
            st.button("Reset", on_click=reset)
    
elif language == "Hindi":
    st.subheader("You selected HINDI!")
    st.button("Let's get started", on_click=start)
    
    if 'start' not in st.session_state:
        st.session_state.start = False
    
    if 'search' not in st.session_state:
        st.session_state.search = False
    
    if 'reset' not in st.session_state:
        st.session_state.reset = False
    
    if st.session_state.start:
        uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
          st.subheader("Uploaded Image:")
          st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
          image = Image.open(uploaded_file)
          if not os.path.exists("images"):
                os.makedirs("images")
          with open(f"images/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
          extracted_text_hindi =extract_text_easyocr(image)
          st.subheader("Extracted Text:")
          st.write(extracted_text_hindi)

          keywords_input = st.text_input("Enter keywords to search within the extracted text (comma-separated):")
          if keywords_input:
                keywords = [keyword.strip() for keyword in keywords_input.split(',')]
                highlighted_text, found_keywords = highlight_keywords(extracted_text_hindi, keywords)
                st.button("Search", on_click=search)
    
                if st.session_state.search:
                    st.subheader("Search Results:")
                    if found_keywords: 
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        st.write(f"Found keywords: {', '.join(found_keywords)}")
                    else:
                        st.warning("No keywords were found in the extracted text.")  
    
                    not_found_keywords = set(keywords) - set(found_keywords)
                    if not_found_keywords:
                        st.error(f"Keywords not found: {', '.join(not_found_keywords)}")
          st.button("Reset", on_click=reset)
