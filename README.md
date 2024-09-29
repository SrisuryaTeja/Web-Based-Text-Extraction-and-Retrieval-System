# Web-Based Text Extraction and Retrieval System

This project is a web application that performs Optical Character Recognition (OCR) on images and highlights keywords within the extracted text. The system supports both English and Hindi languages, allowing users to upload images, extract text, and search for specific keywords within the extracted content.

## Features
- **Language Support**: English and Hindi
- **OCR**: Extracts text from uploaded images.
- **Keyword Search**: Highlights specified keywords in the extracted text.
- **Multiple Image Formats**: Supports PNG, JPG, and JPEG image formats.

## Tech Stack
- **Python**
- **Streamlit**: Web interface for interactive image upload and keyword search.
- **Hugging Face Transformers**: Used for text extraction in English.
- **EasyOCR**: For Hindi text extraction from images.
- **PIL**: To handle image uploads.
- **Torch**: For working with the model and tokenizers.
- **Numpy**: For image processing.

## How it Works
### English OCR Flow:
1. Upload an image containing text.
2. The application uses a Hugging Face pre-trained model to extract text.
3. The extracted text is displayed, and users can search for keywords.
4. The keywords are highlighted within the extracted text.

### Hindi OCR Flow:
1. Upload an image with Hindi text.
2. EasyOCR is used to detect and extract Hindi text from the image.
3. Users can search for Hindi keywords, which will be highlighted in the extracted content.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone <https://github.com/SrisuryaTeja/Web-Based-Text-Extraction-and-Retrieval-System>
    ```

2. **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use myenv\Scripts\activate
    ```

3. **Install Dependencies**:
    Install the required packages listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```bash
    streamlit run app.py
    ```
