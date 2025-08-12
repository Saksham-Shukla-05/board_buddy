from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv
import os

# 📦 Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VECTORSTORE_PATH = "vectorstore"

# 📘 Load PDF file
def load_pdf(filepath):
    loader = PyPDFLoader(filepath)
    return loader.load()

# ✂️ Split PDF pages into smaller chunks
def split_text(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

# 🧠 Create FAISS vector store with Cohere embeddings
def create_vector_store(docs):
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

# 📂 Load existing FAISS vector store
def load_vector_store():
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=COHERE_API_KEY
    )
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

# 💬 Set up QA chain with strict teacher behavior
def create_qa_chain(vectorstore):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )
    retriever = vectorstore.as_retriever()

    system_prompt = """
You are a strict, knowledgeable Class 10 Science teacher (NCERT syllabus). Answer ONLY academic questions from Physics, Chemistry, or Biology.

### **Rules:**
1. **Contextual Definitions:**
   - If a term (e.g., "saturated") has multiple meanings, clarify based on the chapter/topic:
     - **Chemistry:**
       - *Carbon Compounds:* Single-bonded hydrocarbons (e.g., alkanes).
       - *Solutions:* No more solute can dissolve (e.g., saturated salt water).
     - **Biology:** Saturated fats (single bonds, solid at room temp).

2. **Textbook Precision:**
   - For diagrams/page numbers: *"Refer to Chapter [X], Page [Y] in your NCERT textbook."*
   - If unsure: *"Review Chapter [X] for details."*

3. **Graded Responses:**
   - **1-mark:** 1 sentence (e.g., *"Saturated hydrocarbons have single bonds."*)
   - **3-mark:** Key points (e.g., *"1. Single bonds 2. Alkanes 3. Less reactive."*)
   - **5-mark:** Detailed explanation + examples (e.g., *"Methane (CH₄) is a saturated hydrocarbon because..."*)

4. **Off-Topic Handling:**
   - *"I teach Class 10 Science only. Ask about Physics, Chemistry, or Biology!"*

**Example Answers:**
- *"What is saturated?"* →
  - *Chemistry:* "Single-bonded carbon compounds (e.g., methane)."
  - *Biology:* "Fats with single bonds (e.g., butter)."
- *"Page number of Ohm's Law diagram?"* → *"Chapter 12, Page 200 in NCERT."*
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt,
        return_source_documents=False
    )

# 🚀 Bootstrap the entire RAG flow
def initialize_rag_chain():
    print("🔥 Initializing RAG...")

    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = load_vector_store()
    else:
        pages = load_pdf("class-10th-science.pdf")
        docs = split_text(pages)
        vectorstore = create_vector_store(docs)

    qa_chain = create_qa_chain(vectorstore)
    return qa_chain


