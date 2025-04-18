import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load models and encoders
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    content = file.read()
    try:
        return content.decode('utf-8')
    except UnicodeDecodeError:
        return content.decode('latin-1')

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Optional: Function to get confidence if using predict_proba
def get_confidence(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    if hasattr(svc_model, "predict_proba"):
        probs = svc_model.predict_proba(vectorized_text)
        return max(probs[0]) * 100
    return None

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("📄 Resume Category Prediction App")
    st.markdown("Upload a resume in **PDF**, **DOCX**, or **TXT** format and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted the text from the uploaded resume.")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.success(f"🔍 The predicted category of the uploaded resume is: **{category}**")

            confidence = get_confidence(resume_text)
            if confidence:
                st.caption(f"🧠 Model confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
