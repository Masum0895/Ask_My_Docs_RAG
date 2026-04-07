import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#  load any PDF

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
from PIL import Image
import pytesseract
import os

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()

    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif ext == ".txt":
        loader = TextLoader(file_path)
        return loader.load()

    elif ext == ".html":
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()

    elif ext in [".png", ".jpg", ".jpeg"]:
        # OCR for images
        text = pytesseract.image_to_string(Image.open(file_path))
        return [{
            "page_content": text,
            "metadata": {"source": file_path, "page": 1}
        }]

    else:
        raise ValueError("Unsupported file type")


#  split the document

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


#  embedding model

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


#  VecorStoreDB

class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.texts = []
        self.metadata = []
        self.index = None

    def add_documents(self, docs):
        self.texts = [doc.page_content for doc in docs]
        
        # ✅ Store metadata (IMPORTANT)
        self.metadata = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown")
            }
            for doc in docs
        ]

        embeddings = self.embedding_model.encode(self.texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

    def search(self, query, k=5):
        q_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(q_embedding), k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "content": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(distances[0][i])
            })

        return results

#  answering the questions

def generate_answer(query, retrieved_chunks, client, model_name):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer ONLY from the context below.
If not found, say "Not available in document".

Context:
{context}

Question: {query}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


class AdvancedRAGPipeline:
    def __init__(self, vectorstore, client, model_name):
        self.vectorstore = vectorstore
        self.client = client
        self.model_name = model_name
        self.history = []

    def query(self, question, k=3, summarize=False):
        results = self.vectorstore.search(question, k)

        if not results:
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "summary": None,
                "history": self.history
            }

        # ✅ Build context
        context = "\n\n".join([doc["content"] for doc in results])

        # ✅ Prepare sources
        sources = [
            {
                "source": doc["metadata"]["source"],
                "page": doc["metadata"]["page"],
                "score": doc["score"],
                "preview": doc["content"][:120] + "..."
            }
            for doc in results
        ]

        # ✅ Prompt
        prompt = f"""
Use the following context to answer the question concisely.

Context:
{context}

Question: {question}
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content

        # ✅ Add citations
        citations = [
            f"[{i+1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]

        answer_with_citations = answer + "\n\n📚 Citations:\n" + "\n".join(citations)

        # ✅ Optional summary
        summary = None
        if summarize:
            summary_prompt = f"Summarize this in 2 sentences:\n{answer}"
            summary_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary = summary_resp.choices[0].message.content

        # ✅ Store history
        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary,
        })

        return {
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history
        }