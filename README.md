# JFK Assassination Records OCR Processing

This repository contains tools for processing and extracting structured metadata from JFK Assassination Records PDFs using the Mistral AI OCR and Language Model APIs.

## Overview

The JFK Assassination Records Collection consists of thousands of previously classified documents related to the assassination of President John F. Kennedy. This project aims to:

1. Process these historical documents using OCR technology
2. Extract structured metadata (dates, names, agencies, etc.)
3. Make the information more accessible and searchable

## Features

- Processes both small and medium-sized PDFs
- Extracts structured metadata using Mistral's AI models
- Handles mixed content (typed text, handwritten notes, diagrams)
- Maintains a log of processed documents
- Creates summary files for easy reference

## Requirements

- Python 3.8+
- Mistral AI API key
- Required Python packages (see `requirements.txt`)
- JFK Assassination Records PDFs (acquired separately)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/GokuDoku/jfk-ocr-processing.git
cd jfk-ocr-processing
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Mistral API key:
```
MISTRAL_API_KEY=your_api_key_here
```

5. Download the JFK Assassination Record PDFs (not included in this repository due to size constraints):
   - JFK PDFs can be obtained from the National Archives at https://www.archives.gov/research/jfk/release
   - Create directories `jfk_pdfs/` and `jfk_pdfs_organized/` to store the PDFs

## PDF Organization

Once you have downloaded the PDFs, organize them into the following structure:
- `jfk_pdfs/`: Contains all raw PDF files
- `jfk_pdfs_organized/`: Contains PDFs organized by size (small, medium, large)

You can use the script from our companion repository to organize PDFs by size:
```bash
python organize_pdfs_by_size.py
```

## Usage

### Process a single PDF for testing:

```bash
python process_structured_ocr.py --test path/to/your/file.pdf
```

### Process all small PDFs and a sample of medium PDFs:

```bash
python process_structured_ocr.py
```

### Options:

- `--api-key KEY`: Directly provide your Mistral API key (alternative to using .env)
- `--no-images`: Exclude image data from OCR output
- `--small-only`: Process only small PDFs

## Output

The script generates structured output in JSON format with the following information:

- Document date
- Document type
- Sender/author
- Recipient
- Subject
- Agencies mentioned
- Locations mentioned
- Document ID/reference numbers
- Classification markings
- Key individuals mentioned
- Content summary

## Project Structure

- `process_structured_ocr.py`: Main script for OCR processing
- `structured_ocr_results/`: Output directory for processed results

## Example Output

Here's an example of the structured metadata extracted from a document:
```json
{
  "document_date": "1962-04-05",
  "document_type": "MEMORANDUM",
  "sender": "FBI",
  "recipient": "FBI HQ",
  "subject": null,
  "agencies_mentioned": [
    "FBI",
    "CIA",
    "Department of State",
    "Department of Defense"
  ],
  "locations_mentioned": [
    "New York",
    "Colombia",
    "Argentina",
    "Cuba",
    "Florida",
    "Guantanamo",
    "Oriente Province",
    "Key Biscayne"
  ],
  "document_id": "124-10326-10149,105-35253-991,NY 105-35253",
  "classification": "Secret",
  "key_individuals": [
    "JFK",
    "LBJ",
    "ROBERT F. KENNEDY",
    "HUMPHREY",
    "MC CARTHY"
  ],
  "content_summary": "Memorandum discussing political opinions and potential actions regarding Cuba, including conversations with CIA contacts and individuals within the Kennedy administration."
}
```

## License

[MIT License]

## Acknowledgments

- Mistral AI for their OCR and language model APIs
- National Archives for making the JFK Assassination Records available 