import os
import httpx
import tarfile
from pathlib import Path
from dotenv import load_dotenv

# FIX: Removed langchain_community entirely.
# We load PDFs using standard 'pypdf' and map them directly into LangChain's splitter.
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


#------------------------
# setup/config
#------------------------
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION   = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM   = os.getenv("EMBEDDING_DIM", 768)


#----------------------
# helper functions
#----------------------
def download_and_extract(url: str, dest_path: str="./data") -> str:
    """Downloads and extracts tarball from URL"""
    path = Path(dest_path)
    path.mkdir(parents=True, exist_ok=True)
    
    local_tar = "data.tgz"
    
    print(f"Downloading data tarball from {url}...")
    
    with httpx.Client(follow_redirects=True) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with open(local_tar, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=8192):
                    f.write(chunk)
    
    print(f"Extracting to {dest_path}...")
    with tarfile.open(local_tar, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if not Path(m.name).name.startswith("._") and "__MACOSX" not in m.name
        ]
        tar.extractall(path=dest_path, members=members)
    
    os.remove(local_tar) # Cleanup
    return dest_path


# FIX: Replaces DirectoryLoader and PyPDFLoader natively without deprecations
def load_pdfs_from_dir(directory_path: str) -> list[Document]:
    """Scans directory for PDFs, extracts text, and wraps them as LangChain Documents"""
    documents = []
    dir_path = Path(directory_path)
    
    # Recursively find all PDFs
    for file_path in dir_path.rglob("*.pdf"):
        try:
            reader = PdfReader(file_path)
            # Combine text across all pages in the PDF
            text = "".join([page.extract_text() or "" for page in reader.pages])
            
            # Format exactly how LangChain expects it
            documents.append(Document(
                page_content=text,
                metadata={"source": str(file_path)}
            ))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return documents


#-------------------
# main
#-------------------
if __name__ == "__main__":
    # extracting PDFs
    DATA_URL = "https://storage.googleapis.com/public-file-server/genai-downloads/bc_hr_policies.tgz"
    DATA_DIR = download_and_extract(DATA_URL)

    print("> Processing PDFs...")
    # FIX: Native loading mechanism
    docs = load_pdfs_from_dir(DATA_DIR)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIM,
        project=PROJECT_ID,
        location=LOCATION,
        vertexai=True,                  
        task_type="retrieval_document"
    )

    print(f"> Indexing {len(chunks)} chunks into Chroma DB...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("> Done! Database initialized without warnings.")
