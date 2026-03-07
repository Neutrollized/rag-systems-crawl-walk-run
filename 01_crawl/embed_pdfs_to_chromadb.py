import os
import httpx
import tarfile
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma


#------------------------
# setup/config
#------------------------
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION   = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
# common dimension values are: 384, 768 (balanced), 1536, 3072 (default)
EMBEDDING_DIM   = os.getenv("EMBEDDING_DIM", 768)


#----------------------
# helper functions
#----------------------
def download_and_extract(url: str, dest_path: str="./data") -> str:
    """Downloads and extracts tarball from URL

    Args:
        url: URL of the data tarball
        dest_path: Local path to extra the files to

    Returns:
        The path which the files were extract to,
        which is just the 'dest_path' arg
    """
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
        # exclude AppleDouble files
        members = [
            m for m in tar.getmembers()
            if not Path(m.name).name.startswith("._") and "__MACOSX" not in m.name
        ]
        tar.extractall(path=dest_path, members=members)
    
    os.remove(local_tar) # Cleanup
    return dest_path


#-------------------
# main
#-------------------
if __name__ == "__main__":
    # extracting PDFs
    DATA_URL = "https://storage.googleapis.com/public-file-server/genai-downloads/bc_hr_policies.tgz"
    DATA_DIR = download_and_extract(DATA_URL)

    print("> Processing PDFs...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="./**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()

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
        vertexai=True,                  # This tells LangChain to use Vertex AI, not AI Studio
        task_type="retrieval_document"
    )

    print(f"> Indexing {len(chunks)} chunks into Chroma DB...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
