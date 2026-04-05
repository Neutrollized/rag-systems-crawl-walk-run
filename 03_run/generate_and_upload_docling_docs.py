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

from google.cloud import storage


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

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)


#----------------------
# setup/config
#----------------------
load_dotenv()
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCS_BUCKET_PATH = os.getenv("GCS_BUCKET_PATH", "")


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


def upload_folder_to_gcs(local_path: str, bucket_name: str, gcs_folder_prefix=""):
    """Uploads all files from a local directory to a GCS bucket.
    
    Args:
        local_path (str): Path to the local folder (e.g., './my_data')
        bucket_name (str): Name of your GCS bucket (without gs://)
        gcs_folder_prefix (str): Optional 'folder' path inside the bucket
    """
    # Initialize the client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # Convert string path to a Path object
    base_path = Path(local_path)
    markdown_files = list(base_path.glob("*.md"))

    for file in markdown_files:
        try:  # Skip directories themselves
            # Calculate the path relative to the base directory
            relative_path = file.relative_to(base_path)
            
            # Combine with prefix and ensure forward slashes for GCS
            blob_name = str(Path(gcs_folder_prefix) / relative_path).replace("\\", "/")

            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file))
            #print(f"Uploaded: {file} -> {blob_name}")

        except Exception as e:
            print(f"Error on {file}: {e}")


#----------------
# main
#----------------
if __name__ == "__main__":
    # extracting PDFs
    DATA_URL = "https://storage.googleapis.com/public-file-server/genai-downloads/bc_hr_policies.tgz"
    DATA_DIR = download_and_extract(DATA_URL)

    docling_docs = Path("./docling_docs")
    docling_docs.mkdir(parents=True, exist_ok=True)

    print("> Processing PDFs...")
    data_path = Path(DATA_DIR)
    pdf_files = list(data_path.glob("*.pdf"))
    
    for file in pdf_files:
        try:
            #print(file.name)
            result = converter.convert(file)
            markdown_content = result.document.export_to_markdown()
            #print(markdown_content)

            docling_doc_name = Path(file).stem + ".md"
            #print(docling_doc_name)
            with open(f"docling_docs/{docling_doc_name}", "w", encoding="utf-8") as f:
                f.write(markdown_content)

        except Exception as e:
            print(f"Error on {file}: {e}")

    print("> Uploading Docling docs to GCS...")
    upload_folder_to_gcs("./docling_docs", GCS_BUCKET, GCS_BUCKET_PATH)
