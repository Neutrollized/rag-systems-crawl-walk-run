import os
import sys
import httpx
import tarfile
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

import cohere

import lancedb
from lancedb.pydantic import LanceModel, Vector


#--------------------------------------------------------------
# Settings for Apple Metal GPUs
# - get the num_threads from About This Mac > 
#   System Report... > 
#   under Graphics/Displays > 
#   Total Number of Cores
#--------------------------------------------------------------
accel_options = AcceleratorOptions(
    device=AcceleratorDevice.MPS,
    num_threads=10  # Adjust based on your Mac's performance cores
)

pipeline_options = PdfPipelineOptions(accelerator_options=accel_options)
pipeline_options.do_ocr = False


#----------------------
# setup/config
#----------------------
load_dotenv()
COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-english-light-v3.0")
EMBEDDING_DIM   = os.getenv("EMBEDDING_DIM", 384)


#------------------------
# helper functions
#------------------------
def download_and_extract(url: str, dest_path: str="./data") -> str:
    """Downloads and extracts tarball from URL

    Args:
        url (str): URL of the data tarball
        dest_path (str): Local path to extra the files to

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
        members = [
            m for m in tar.getmembers()
            if not Path(m.name).name.startswith("._") and "__MACOSX" not in m.name
        ]
        tar.extractall(path=dest_path, members=members)

    os.remove(local_tar) # Cleanup
    return dest_path


def docling_chunk_pdf(file: str) -> tuple[list, list]:
    """Chunks PDF file using Docling

    Args:
        file (str): Name of file to process

|   Returns:
        A tuple of two lists
        - A nested list where each sub-list represents a chunk and all its contents (metadata, text, etc.)
        - A nested list where each sub-list represents a chunk of (only) processed text containing individual strings
    """
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(file)
    doc = result.document
    chunker = HybridChunker()
    chunks = list(chunker.chunk(doc))
    chunk_texts = [c.text for c in chunks]
    
    return chunks, chunk_texts


def cohere_embedding(chunk_texts: list, embedding_model: str, output_dimension: int) -> list:
    """Using Cohere embedding model to embed text
    
    Args:
        chunk_texts (list): Chunked text
        embedding_model (str): Cohere embedding model
        output_dimension (int): Embedding model output dimension

    Returns:
        A nested list where each sub-list represents a vector embedding of the chunked text
    """
    co = cohere.ClientV2()

    response = co.embed(
        texts=chunk_texts,
        model=embedding_model,
        input_type="search_document",
        output_dimension=output_dimension,
        embedding_types=["float"]
    )

    vector = response.embeddings.float

    return vector


def lancedb_insert(file: str, chunks: list, chunk_texts: list, vectors: list):
    db = lancedb.connect(DB_FILE)

    data = []
    for i, vector in enumerate(vectors):
        data.append({
            "vector": vector,
            "text": chunk_texts[i],
            "source": file,
            "heading": chunks[i].meta.headings if chunks[i].meta.headings else [""],
            "page_no": chunks[i].meta.doc_items[0].prov[0].page_no if chunks[i].meta.doc_items else 0
        })

    tbl = db.open_table(TABLE_NAME)
    tbl.add(data)

    print(f"Successfully indexed {len(data)} chunks into {DB_FILE}")


#----------------
# main
#----------------
if __name__ == "__main__":
    # extracting PDFs
    DATA_URL = "https://storage.googleapis.com/public-file-server/genai-downloads/bc_hr_policies.tgz"
    DATA_DIR = download_and_extract(DATA_URL)

    print(EMBEDDING_MODEL)
    # LanceDB init
    DB_FILE = "lancedb_data"
    TABLE_NAME = "bc_hr_policies"
    class DocumentSchema(LanceModel):
        vector: Vector(int(EMBEDDING_DIM))  # <--- Specifying dimensions here
        text: str
        source: str
        heading: list[str]
        page_no: int

    print("> Initializing LanceDB...")
    db = lancedb.connect(DB_FILE)
    tbl = db.create_table(TABLE_NAME, schema=DocumentSchema, mode="overwrite")

    print("> Processing PDFs...")
    data_path = Path(DATA_DIR)
    pdf_files = list(data_path.glob("*.pdf"))
    
    for file in pdf_files:
        print(file.name)
        chunks, chunk_texts = docling_chunk_pdf(file)
        vectors = cohere_embedding(chunk_texts, EMBEDDING_MODEL, int(EMBEDDING_DIM))
        lancedb_insert(str(file.name), chunks, chunk_texts, vectors)
