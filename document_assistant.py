import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq


from pipeline import (
    load_document,
    split_documents,
    load_embedding_model,
    VectorStore,
    AdvancedRAGPipeline
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(" GROQ_API_KEY not found in .env")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)



from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(chat_history, filename="chat_history.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    for i, chat in enumerate(chat_history):
        content.append(Paragraph(f"<b>Q{i+1}:</b> {chat['question']}", styles["Normal"]))
        content.append(Spacer(1, 10))

        content.append(Paragraph(f"<b>Answer:</b> {chat['answer']}", styles["Normal"]))
        content.append(Spacer(1, 10))

        if chat.get("summary"):
            content.append(Paragraph(f"<b>Summary:</b> {chat['summary']}", styles["Italic"]))
            content.append(Spacer(1, 10))

        content.append(Spacer(1, 20))

    doc.build(content)
    return filename


# UI

st.set_page_config(page_title="AskMyDocs", layout="wide")
st.markdown("""
<h1 style='text-align: center;'>📄 AskMyDocs</h1>
<h3 style='text-align: center;'> Chat with your PDFs, Word files & more using AI</h3>
<p style='text-align: center; color: gray;'>
Ask questions, get answers with citations & summaries
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt", "html", "png"]
)



st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    .answer-box {
        background-color: #1e1e1e;
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }

    .summary-box {
        color: #cccccc;
    }

    .stButton button {
        background-color: #1f2937;
        color: white;
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)


# load model 

@st.cache_resource
def load_model():
    return load_embedding_model()

embedding_model = load_model()



if uploaded_file:
    import os

    #  Get file extension dynamically
    file_ext = os.path.splitext(uploaded_file.name)[1]
    file_path = f"temp{file_ext}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ {uploaded_file.name} uploaded!")

    documents = load_document(file_path)
    chunks = split_documents(documents)

    vectorstore = VectorStore(embedding_model)
    vectorstore.add_documents(chunks)

    st.info(f"📚 {len(chunks)} chunks created")


    # model
    model_name = "llama-3.3-70b-versatile"

    #  Create pipeline
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = AdvancedRAGPipeline(
            vectorstore,
            client,
            model_name
        )

    pipeline = st.session_state.pipeline



    summarize = st.checkbox("Generate Summary")

    with st.sidebar:
        st.title("⚙️ Settings")


        model_name = st.selectbox(
            "Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        )

        k = st.slider("Top-K Chunks", 1, 10, 3)

        summarize = st.checkbox("Enable Summary")

        st.markdown("---")
        # Clear chat 
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []   

        st.markdown("---")
        st.info("📄 Upload a document and start asking questions!")


    # chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # input chat
    query = st.chat_input("Ask something about your Document...")

    if query:
        result = pipeline.query(query, k=k, summarize=summarize)

        st.session_state.chat_history.append({
            "question": query,
            "answer": result["answer"],
            "summary": result["summary"],
            "sources": result["sources"]
        })

    # displaying chat
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.markdown(
                f"<div class='answer-box'>{chat['answer']}</div>",
                unsafe_allow_html=True
            )

            if chat["summary"]:
                st.caption("📌 Summary:")
                st.markdown(
                f"<div class='summary-box'>{chat['summary']}</div>",
                unsafe_allow_html=True
            )

            with st.expander("📚 Sources"):
                for i, src in enumerate(chat["sources"]):
                    st.markdown(f"""
                    **[{i+1}] {src['source']} (Page {src['page']})**  
                    > {src['preview']}
                    """)

    #  download chat as PDF
    if st.session_state.chat_history:
        if st.button("📥 Download Chat as PDF"):
            pdf_file = generate_pdf(st.session_state.chat_history)

            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="⬇️ Click to Download",
                    data=f,
                    file_name="chat_history.pdf",
                    mime="application/pdf"
                )