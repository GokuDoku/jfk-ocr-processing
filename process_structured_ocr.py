#!/usr/bin/env python3
"""
Script to process JFK Assassination Records PDFs using Mistral OCR API
with structured output extraction based on Mistral's cookbook approach
"""

import os
import json
import time
import base64
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("structured_ocr_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("structured-mistral-ocr")

# Constants
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
OCR_ENDPOINT = f"{MISTRAL_API_BASE}/ocr"
CHAT_ENDPOINT = f"{MISTRAL_API_BASE}/chat/completions"
SMALL_PDF_DIR = "jfk_pdfs_organized/small"
MEDIUM_PDF_DIR = "jfk_pdfs_organized/medium"
OCR_RESULTS_DIR = "structured_ocr_results"
MAX_MEDIUM_PDFS = 10
RETRY_DELAY = 5  # Seconds between retries

# Define structured output model (following the cookbook approach)
class JFKDocumentMetadata(BaseModel):
    document_date: Optional[str] = None
    document_type: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    subject: Optional[str] = None
    agencies_mentioned: Optional[List[str]] = None
    locations_mentioned: Optional[List[str]] = None
    document_id: Optional[Union[str, List[str]]] = None
    classification: Optional[str] = None
    key_individuals: Optional[List[str]] = None
    content_summary: Optional[str] = None

    def clean_data(self):
        """Clean data to ensure consistent format"""
        # Convert document_id to string if it's a list
        if isinstance(self.document_id, list) and self.document_id:
            self.document_id = ', '.join(str(id) for id in self.document_id)
        
        # Clean up other fields as needed
        return self

def extract_metadata_from_ocr(ocr_response: Dict[str, Any], api_key: str) -> JFKDocumentMetadata:
    """
    Extract structured metadata from OCR output using Mistral's chat API
    """
    # Get the markdown content from the OCR response
    markdown_content = ""
    for page in ocr_response.get("pages", []):
        if "markdown" in page:
            markdown_content += page["markdown"] + "\n\n"
    
    # Prepare the prompt for metadata extraction
    system_prompt = """You are an expert document analyzer specialized in historical documents from the JFK era.
Extract key metadata from historical documents related to the JFK assassination in a structured way.
Focus on extracting: dates, document types, names, agencies/departments, locations, and document IDs/numbers.
Provide the output as a JSON object with these fields, using null for missing information.

IMPORTANT: Ensure document_id is a single string value. If multiple IDs are found, combine them with commas.
"""
    
    user_prompt = f"""Extract structured information from this historical document:

{markdown_content[:4000]}  # Limit to first 4000 chars to avoid token limits

Return ONLY a JSON object with the following fields:
- document_date: The date of the document (format: YYYY-MM-DD or null if uncertain)
- document_type: Type of document (memo, letter, report, etc.)
- sender: Person or organization sending/authoring document
- recipient: Person or organization receiving document
- subject: The document's subject or title
- agencies_mentioned: List of government agencies mentioned
- locations_mentioned: List of locations mentioned
- document_id: Any reference/document numbers (single string, comma-separated if multiple)
- classification: Any classification markings (SECRET, CONFIDENTIAL, etc.)
- key_individuals: List of important individuals mentioned
- content_summary: A brief 1-2 sentence summary of the document contents
"""

    # Prepare request to the chat API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    # Make API request
    logger.info("Sending request to Mistral Chat API for metadata extraction")
    
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                json_str = result["choices"][0]["message"]["content"]
                try:
                    metadata_dict = json.loads(json_str)
                    logger.info(f"Successfully extracted metadata")
                    
                    # Handle potential formatting issues
                    if "document_id" in metadata_dict and isinstance(metadata_dict["document_id"], list):
                        metadata_dict["document_id"] = ", ".join(str(id) for id in metadata_dict["document_id"])
                    
                    # Convert the dictionary to our Pydantic model and clean it
                    metadata = JFKDocumentMetadata(**metadata_dict)
                    return metadata.clean_data()
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {json_str}")
                except Exception as e:
                    logger.error(f"Error processing metadata: {str(e)}")
            else:
                logger.error(f"Unexpected response format: {result}")
        else:
            logger.error(f"Error from Chat API: {response.status_code} - {response.text}")
    
    except Exception as e:
        logger.error(f"Exception during Chat API call: {e}")
    
    return JFKDocumentMetadata()  # Return empty model if extraction failed

def process_pdf_with_ocr(api_key: str, pdf_path: Path, model: str = "mistral-ocr-latest") -> Dict[str, Any]:
    """Process a single PDF with Mistral OCR API"""
    logger.info(f"Processing: {pdf_path}")
    
    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    # Encode in base64 for the API request
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare payload - using the recommended document_url format from the cookbook
    payload = {
        "model": model,
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_base64}"
        },
        "include_image_base64": True  # Include image data for better context
    }
    
    # Make API request with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Mistral OCR API for {pdf_path.name} (attempt {attempt+1})")
            response = requests.post(
                OCR_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            # Check response
            if response.status_code == 200:
                logger.info(f"Successfully processed {pdf_path.name}")
                return response.json()
            elif response.status_code == 429:  # Rate limit
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error processing {pdf_path.name}: {response.status_code} - {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return None
        
        except Exception as e:
            logger.error(f"Exception during API call: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None
    
    return None

def save_results(ocr_results: Dict[str, Any], metadata: JFKDocumentMetadata, output_path: Path):
    """Save OCR results and metadata to a JSON file"""
    combined_results = {
        "ocr_results": ocr_results,
        "extracted_metadata": json.loads(metadata.model_dump_json())
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def process_single_pdf(pdf_path: str, api_key: str):
    """Process a single PDF file for testing purposes"""
    results_dir = Path(OCR_RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    
    output_filename = f"{pdf_path.stem}_structured.json"
    output_path = results_dir / output_filename
    
    try:
        # Process the PDF
        logger.info(f"Processing single PDF for testing: {pdf_path}")
        ocr_results = process_pdf_with_ocr(api_key, pdf_path)
        
        if ocr_results:
            # Extract metadata from OCR results
            metadata = extract_metadata_from_ocr(ocr_results, api_key)
            
            # Save the combined results
            save_results(ocr_results, metadata, output_path)
            
            logger.info(f"Test complete. Results saved to {output_path}")
            print(f"Test complete. Results saved to {output_path}")
            
            # Print a summary of the metadata extracted
            print("\nExtracted Metadata Summary:")
            print(f"Document Type: {metadata.document_type}")
            print(f"Document Date: {metadata.document_date}")
            print(f"Sender: {metadata.sender}")
            print(f"Recipient: {metadata.recipient}")
            print(f"Subject: {metadata.subject}")
            print(f"Classification: {metadata.classification}")
            print(f"Document ID: {metadata.document_id}")
            print(f"Content Summary: {metadata.content_summary}")
            
            return True
        else:
            logger.error(f"Failed to process {pdf_path.name}")
            return False
    
    except Exception as e:
        logger.error(f"Exception processing {pdf_path.name}: {e}")
        return False

def process_pdfs(api_key: str, include_images: bool = True):
    """Process small PDFs and a limited number of medium PDFs"""
    # Create results directory if it doesn't exist
    results_dir = Path(OCR_RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    
    # Create a summary file for easy reference of all processed documents
    summary_file = results_dir / "metadata_summary.jsonl"
    
    # Get list of small PDFs
    small_pdfs = list(Path(SMALL_PDF_DIR).glob("*.pdf"))
    logger.info(f"Found {len(small_pdfs)} small PDFs")
    
    # Get list of medium PDFs (limited)
    medium_pdfs = list(Path(MEDIUM_PDF_DIR).glob("*.pdf"))[:MAX_MEDIUM_PDFS]
    logger.info(f"Selected {len(medium_pdfs)} medium PDFs")
    
    # Combine the lists
    all_pdfs = small_pdfs + medium_pdfs
    
    # Process PDFs with progress bar
    successful_count = 0
    failed_count = 0
    
    for pdf_path in tqdm(all_pdfs, desc="Processing PDFs"):
        output_filename = f"{pdf_path.stem}_structured.json"
        output_path = results_dir / output_filename
        
        # Skip if already processed
        if output_path.exists():
            logger.info(f"Skipping {pdf_path.name} - already processed")
            successful_count += 1
            continue
        
        try:
            # Process the PDF
            ocr_results = process_pdf_with_ocr(api_key, pdf_path)
            
            if ocr_results:
                # Extract metadata from OCR results
                metadata = extract_metadata_from_ocr(ocr_results, api_key)
                
                # Save the combined results
                save_results(ocr_results, metadata, output_path)
                
                # Write a summary entry
                with open(summary_file, 'a') as f:
                    summary_entry = {
                        "filename": pdf_path.name,
                        "processed_at": datetime.now().isoformat(),
                        "metadata": json.loads(metadata.model_dump_json()),
                    }
                    f.write(json.dumps(summary_entry) + "\n")
                
                successful_count += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
            else:
                failed_count += 1
                logger.error(f"Failed to process {pdf_path.name}")
        
        except Exception as e:
            failed_count += 1
            logger.error(f"Exception processing {pdf_path.name}: {e}")
    
    # Log final statistics
    logger.info(f"Processing complete. Successfully processed: {successful_count}, Failed: {failed_count}")
    print(f"Processing complete. Successfully processed: {successful_count}, Failed: {failed_count}")
    print(f"Results saved to {results_dir}")

def main():
    """Main function to parse arguments and start processing"""
    parser = argparse.ArgumentParser(description="Process JFK PDFs with Mistral OCR and extract structured metadata")
    parser.add_argument("--api-key", help="Your Mistral API key (can also be set in .env file)")
    parser.add_argument("--no-images", action="store_true", help="Don't include images in OCR output")
    parser.add_argument("--small-only", action="store_true", help="Process only small PDFs")
    parser.add_argument("--test", help="Process a single PDF file for testing")
    
    args = parser.parse_args()
    
    # Get API key from command line or environment variable
    api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        print("Error: Mistral API key is required. Set it in .env file or provide via --api-key")
        return
    
    # Test a single file if requested
    if args.test:
        process_single_pdf(args.test, api_key)
    else:
        process_pdfs(api_key, not args.no_images)

if __name__ == "__main__":
    main() 