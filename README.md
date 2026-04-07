# 📄 AskMyDocs

# 🚀Live Demo: https://askmydocs07.streamlit.app/


An AI-powered **Document Question Answering System** built with Retrieval-Augmented Generation (RAG).
Upload documents and get intelligent, context-aware answers instantly.

---

## 🚀 Features

* 📄 Supports PDF, DOCX, TXT, HTML
* 🔍 Semantic Search using FAISS
* 🤖 LLM-powered answers (Groq API)
* 📚 Source citations
* 🧠 Chat history tracking
* ✨ Answer summarization

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **RAG Framework:** LangChain
* **Vector DB:** FAISS
* **Embeddings:** Sentence Transformers
* **LLM:** Groq

---

## 🚀 Quick Start

### 🔧 Prerequisites

* Python 3.9+
* pip

---

### 📥 Installation

```bash
git clone https://github.com/Masum0895/Ask_My_Docs_RAG.git
cd Ask_My_Docs_RAG
```

---

### 🧪 Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # Mac/Linux
```

---

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔐 Setup Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### ▶️ Run the Application

```bash
streamlit run main.py
```

---

## 📁 Project Structure

```
Rag_prac1/
├── main.py                 # Streamlit UI
├── pipeline.py             # RAG pipeline logic
├── document_assistant.py   # Core assistant logic
├── requirements.txt        # Dependencies
├── .gitignore              # Ignore rules
└── README.md               # Documentation
```

---

## ⚙️ Configuration

You can customize:

* Chunk size for document splitting
* Embedding model
* LLM model (Groq)
* Retrieval parameters (Top-K)

---

## ⚠️ Important Notes

* ❌ Do NOT upload API keys
* ❌ Do NOT commit sensitive documents
* ⚡ Be aware of API rate limits

---

## 🐛 Troubleshooting

**Module not found**

```bash
pip install -r requirements.txt --upgrade
```

**API Key Error**

* Ensure `.env` file exists
* Verify API key is correct

**Document not loading**

* Check file format
* Ensure file is not corrupted

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch
3. Commit changes
4. Push and open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 📞 Support

* Open an issue on GitHub
* Check existing issues

---

---

## 📬 Connect & Support

**Questions or feedback?**  
📧 Email: [masumbilla0895@gmail.com](mailto:masumbilla0895@gmail.com)  
🔗 Project: [github.com/Masum0895/Ask_My_Docs_RAG](https://github.com/Masum0895/Ask_My_Docs_RAG)

**Found this helpful?**  
⭐ Star this repository to show support!  
🔄 Share it with others who might benefit  
🐛 Report issues to help improve the project

---

*Built with ❤️ by Masum*
