import os
import re
import requests
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image, UnidentifiedImageError
import json
import time
from regex_patterns import POSITIVE_PATTERN, NEGATIVE_PATTERN, QUANTUM_GATES_PATTERN, QUANTUM_ALGO_PROBLEM_PATTERN, ABBREVIATION_MAPPING, GATE_NORMALIZATION_MAP, CAPTION_REGEX
import csv
import random


# Output directories and global variables
OUT_DATA_FILE = "./captions_for_regex.json"
ARXIV_IDS_FILE = "./paper_list_71.txt"
DOWNLOAD_TIMEOUT = 30
fig_counter = 1



def download_pdf_to_bytes(arxiv_id):
    """
    Download the PDF for an arXiv paper and return it as a BytesIO object.
    Returns None if the download fails or is not a valid PDF.
    
    Parameters
    ----------
    arxiv_id : str
        The arXiv ID of the paper to download.

    Returns
    -------
    pdf_bytes : BytesIO
        The PDF as a BytesIO object.
    """

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except Exception as e:
        print(f"Request failed for {arxiv_id}: {e}")
        return None

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code} for {arxiv_id}")
        return None

    # Verify PDF magic bytes
    if not resp.content.startswith(b"%PDF"):
        print(f"Downloaded file for {arxiv_id} is not a PDF!")
        return None

    print(f"Downloaded PDF successfully for {arxiv_id}")
    return BytesIO(resp.content)



def overlap_x_fraction(rect1, rect2):
    """Calculates the horizontal overlap fraction between two rectangles."""
    x1 = max(rect1.x0, rect2.x0)
    x2 = min(rect1.x1, rect2.x1)
    if x2 <= x1:
        return 0.0

    width1 = rect1.width
    if width1 == 0:
         return 0.0

    return (x2 - x1) / width1


def bbox_distance_vertical(upper_rect, lower_rect):
    """distance in vertical axis from bottom of upper_rect to top of lower_rect (could be negative if overlap)"""
    return lower_rect.y0 - upper_rect.y1




def extract_text_blocks(page):
    """
    Extract text blocks from the page.

    Parameters
    ----------
    page : fitz.Page
        The page to extract text blocks from.

    Returns
    -------
    text_blocks : list
        The list of text blocks.
    """ 
    blocks = page.get_text("blocks")
    text_blocks = []
    for b in blocks:
        x0, y0, x1, y1, text, block_no = b[:6]
        # FIX 1: Replace newlines with spaces to handle multi-line captions cleaner
        clean_text = text.replace("\n", " ").strip()
        text_blocks.append({
            "rect": fitz.Rect(x0, y0, x1, y1),
            "text": clean_text,
        })
    text_blocks.sort(key=lambda tb: tb["rect"].y0)

    return text_blocks



def clean_all_text(text):
    """
    Clean text by removing prefixes, hyphens, sub-labels, and line breaks.
    
    Parameters
    ----------
    text : str
        The text to clean.

    Returns
    -------
    text : str
        The cleaned text.
    """
    
    # 1. Remove hyphenation from line breaks (e.g., quan-\ntum → quantum)
    text = re.sub(r'-\s*\n\s*', '', text)

    # 2. Remove Figure / Fig at start of caption
    text = re.sub(
        r'^\s*(figure|fig)\s*\.?\s*\d*\s*[:.\-]\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # 3. Remove sub-labels like "(a)", "(b)" at start of each sentence
    text = re.sub(
        r'(^|[.!?]\s+)\([a-zA-Z]\)\s*',
        r'\1',
        text
    )

    # 4. Remove all newline characters safely
    #    (handles both between words and inside words)
    text = re.sub(r'\s*\n\s*', ' ', text)

    # 5. Normalize multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()



def extract_caption_blocks(text_blocks):
    """
    Extract caption blocks from the text blocks

    Parameters
    ----------
    text_blocks : list
        The list of text blocks.

    Returns
    -------
    caption_blocks : list
        The list of caption blocks.
    """
    caption_blocks = []
    for idx, tb in enumerate(text_blocks):
        # Check if it is a "pure caption" - MUST start with the keyword
        # No text should exist before the keyword.
        if CAPTION_REGEX.match(tb["text"]):
            # Use the new cleaning helper
            clean_caption = tb["text"].replace("\n", " ").strip()

            caption_blocks.append({
                "cap_rect": tb["rect"],
                "text": clean_caption,
                "index": idx
            })

    return caption_blocks



def map_abbrev_to_full(matches):
    """
    Map abbreviations to their full forms

    Parameters
    ----------
    matches : list
        The list of abbreviations to map.

    Returns
    -------
    mapped : list
        The list of full forms.
    """
    mapped = []
    for m in matches:
        key = m.upper()  # normalize case for dictionary lookup
        if key in ABBREVIATION_MAPPING:
            mapped.append(ABBREVIATION_MAPPING[key])
        else:
            mapped.append(m)
    return mapped



def extract_noun_phrases(text):
    """
    Extract noun phrases from the text using Noun-Phrase Chunking technique

    Parameters
    ----------
    text : str
        The text to extract noun phrases from.

    Returns
    -------
    noun_chunks : list
        The list of noun phrases.
    """
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]



def get_quantum_problem(descriptions, caption_text):
    """
    Get the quantum problem being solved in the quantum circuit figure

    Parameters
    ----------
    descriptions : list
        The list of descriptive texts.
    caption_text : str
        The caption text from below the figure.

    Returns
    -------
    problems : list
        The list of quantum problems.
    """
    # 1. Clean all descriptions
    descriptions_clean = [clean_all_text(desc) for desc in descriptions]
    # 2. Join descriptions into a single string
    all_text = " ".join(descriptions_clean)
    
    # 3. Also include the cleaned caption in the search
    caption_clean = clean_all_text(caption_text)
    all_text += " " + caption_clean

    # 4. Find matches using the quantum algorithm problem pattern
    matches = QUANTUM_ALGO_PROBLEM_PATTERN.findall(all_text)
    if matches:
        problems_list = map_abbrev_to_full(list(dict.fromkeys(matches))) # Deduplicate while preserving order
        return ",".join(problems_list)

    # 5. If no standard quantum algorithm problem is found, try to extract noun phrases from the caption
    noun_chunks = extract_noun_phrases(caption_clean)
    if not noun_chunks:
        return "Generic Quantum Implementation"
    
    # 6. Safely build a string based on available noun chunks to build a coherent quantum problem description
    if len(noun_chunks) >= 4:
        return f"{noun_chunks[0]} of {noun_chunks[1]} with {noun_chunks[2]} and {noun_chunks[3]}"
    elif len(noun_chunks) == 3:
        return f"{noun_chunks[0]} of {noun_chunks[1]} with {noun_chunks[2]}"
    elif len(noun_chunks) == 2:
        return f"{noun_chunks[0]} of {noun_chunks[1]}"
    else:
        return noun_chunks[0]



def normalize_and_deduplicate_gates(gate_list):
    """
    Map full gate names to abbreviations and deduplicate the gates

    Parameters
    ----------
    gate_list : list
        The list of gates.

    Returns
    -------
    gates : list
        The list of normalized and deduplicated gates.
    """
    normalized = set()

    # 1. Traverse the gate list
    for gate in gate_list:
        g = gate.strip().upper()

        # 2. Normalize whitespace
        g = re.sub(r"\s+", " ", g)

        # 3. Parameterized gates (keep as-is)
        if re.match(r"C?[A-ZΑ-Ωα-ω]*R[XYZ]\s*\(.*\)", g) or re.match(r"U[123]\s*\(.*\)", g):
            normalized.add(g)
            continue

        # 4. Indexed gates (X1, CNOT_01 → base gate)
        g = re.sub(r"[_\d]+$", "", g)

        # 5. Map full names → abbreviations
        g = GATE_NORMALIZATION_MAP.get(g, g)

        # 6. Add to the set
        normalized.add(g)

    return sorted(normalized)



def get_quantum_gates(descriptions):
    """
    Get the quantum gates from the descriptive texts

    Parameters
    ----------
    descriptions : list
        The list of descriptive texts.

    Returns
    -------
    gates : list
        The list of quantum gates.
    """
    # 1. Clean all descriptions
    descriptions_clean = [clean_all_text(desc) for desc in descriptions]

    # 2. Join descriptions into a single string
    descriptions_clean = " ".join(descriptions_clean)

    # 3. Find all quantum gates using regex
    gates = normalize_and_deduplicate_gates(
        re.findall(QUANTUM_GATES_PATTERN, descriptions_clean)
    )
    return gates




def extract_descriptions_with_pos(all_blocks_with_pos, fig_number, caption_text, caption_rect, page_num, descriptions, text_positions):
    """
    Extract the descriptive texts which are related to the figure above the caption

    Parameters
    ----------
    all_blocks_with_pos : list
        The list of all text blocks with their positions.
    fig_number : int
        The figure number.
    caption_text : str
        The caption text.
    caption_rect : fitz.Rect
        The caption rectangle.
    page_num : int
        The page number.
    descriptions : list
        An empty array to store the descriptive texts.
    text_positions : list
        An empty array to store the text positions.

    Returns
    -------
    descriptions : list
        The list of descriptive texts.
    text_positions : list
        The list of positions of the descriptive texts.
    """

    # 1. Traverse all text blocks and find the caption block
    for block in all_blocks_with_pos:
        is_caption_block = False
        block_clean = block['text'].replace('\n', ' ').strip()
        # 2. For each block, check if it is the caption block
        if block_clean == caption_text:
            is_caption_block = True

        # 3. If the block is not the caption block, check if it is the figure block
        if block['page'] == page_num:
            if block['rect'].intersects(caption_rect):
                if abs(len(block['text']) - len(caption_text)) < 50:
                    is_caption_block = True

        if is_caption_block:
            caption_out_positions = (block['start'], block['end'])
            continue

        if fig_number:
            ref_regex = re.compile(rf"(Figure|Fig\.?)\s*{re.escape(fig_number)}(?!\d)", re.IGNORECASE)
            if ref_regex.search(block['text']):
                # NEW: Sentence-based windowing (+/- 5 sentences) with accurate offsets
                # Use finditer to keep track of character offsets for each sentence
                sentence_matches = list(re.finditer(r'[^.!?]+(?:[.!?]\s*|$)', block['text']))
                sentences_info = [(m.group(), m.start(), m.end()) for m in sentence_matches]
                
                match_idx = -1
                for idx, (sent_text, s, e) in enumerate(sentences_info):
                    if ref_regex.search(sent_text):
                        match_idx = idx
                        break
                
                if match_idx != -1:
                    win_start_idx = max(0, match_idx - 5)
                    win_end_idx = min(len(sentences_info), match_idx + 6)
                    
                    # Extract windowed text
                    window_sents = sentences_info[win_start_idx:win_end_idx]
                    desc_text = " ".join([si[0] for si in window_sents]).strip()
                    
                    # Calculate accurate document positions
                    doc_start = block['start'] + window_sents[0][1]
                    doc_end = block['start'] + window_sents[-1][2]
                    
                    descriptions.append(desc_text)
                    text_positions.append((doc_start, doc_end))

    return descriptions, text_positions, caption_out_positions





def process_pdf_bytes(arxiv_id, pdf_bytes_io):
    """
    Process a PDF and call various functions to extract images, captions, descriptive text and other metadata.

    Parameters
    ----------
    arxiv_id : str
        The arxiv id of the paper.
    pdf_bytes_io : io.BytesIO
        The PDF file as a bytes IO stream.

    Returns
    -------
    results : list
        A list of dictionaries, each containing the metadata for a relevant quantum circuit figure.
    """
    global fig_counter

    # 1. Preprocessing and file opening
    arxiv_id = arxiv_id.replace(":", "-")
    doc = fitz.open(stream=pdf_bytes_io.read(), filetype="pdf")
    results = []

    full_document_text = ""
    all_blocks_with_pos = []

    # 2. Traverse each page of the pdf
    for pno in range(doc.page_count):
        page = doc[pno]
        # 3. Extract text blocks from the page
        blocks = page.get_text("blocks")
        for b in blocks:
            text_content = b[4].strip()
            if not text_content:
                continue

            # 4. Calculate the start and end indices of the text block
            start_idx = len(full_document_text)
            full_document_text += text_content + "\n"
            end_idx = len(full_document_text)

            all_blocks_with_pos.append({
                "text": text_content,
                "start": start_idx,
                "end": end_idx,
                "rect": fitz.Rect(b[:4]),
                "page": pno + 1
            })

    print(f"Index built. Total text length: {len(full_document_text)} chars.")

    # 5. Re-traverse each page of the pdf
    for pno in range(doc.page_count):
        try: 
            page = doc[pno]
            page_num = pno + 1
            print(f"Processing Page {page_num}/{doc.page_count}")

            # 6) extract text blocks for this page
            text_blocks = extract_text_blocks(page)

            # 7) find those blocks from text blocks that have captions for an image
            caption_blocks = extract_caption_blocks(text_blocks)


            # 8) Traverse the caption blocks
            for ca in caption_blocks:
                caption_text = ca["text"]
                caption_rect = ca["cap_rect"]

                # 9) Check if the caption is a caption for a quantum circuit figure. If not, skip to the next caption block.
  

                # 10) Extract the figure number from the caption text
                fig_match = CAPTION_REGEX.search(caption_text)
                fig_number = fig_match.group(2) if fig_match else None

                descriptions = []
                text_positions = []

                caption_out_text = caption_text
                caption_out_positions = ()

                # 11) Extract other descriptive texts which are related to the quantum circuit figure along with their positions
                descriptions, text_positions, caption_out_positions = extract_descriptions_with_pos(all_blocks_with_pos, fig_number, caption_text, caption_rect, page_num, descriptions, text_positions)

                # 12) Get the quantum problem and quantum gates from all the descriptive texts

                results.append({
                    "fig counter": fig_counter,
                    "arxiv paper no.": arxiv_id.replace("-",":"),
                    "page number": page_num,
                    "figure number": int(fig_number) if fig_number and fig_number.isdigit() else "",
                    "caption text": caption_text,
                    "caption positions": caption_out_positions,
                    "descriptions": descriptions,
                    "text positions": text_positions
                    })

                fig_counter += 1

        except Exception as e:
            print(f"Skipping Page {pno+1} of {arxiv_id} due to processing error: {e}")
            continue

    doc.close()
    return results




if __name__ == "__main__":
    """
    Traverses the list of paper ids, calls various processing functions, updates the of each paper pdfs to JSON and CSV files.
    """
    random.seed(42)
    #3. Read the list of arxiv ids
    with open(ARXIV_IDS_FILE, "r", encoding="utf-8") as f:
        arxiv_ids = [line.strip() for line in f if line.strip()]
    files_count = 0
    images_count = 0
    all_results = []
    csv_data = []
    last_idx = 0

    arxiv_ids_chunk = arxiv_ids[25000:26000]

    # Uncomment the following line to process a random sample of 1000 arxiv ids
    # arxiv_ids_chunk = random.sample(arxiv_ids[20000:26000], 1000)
    
    #4. Traverse the list of arxiv ids
    for idx, arxiv_id in enumerate(arxiv_ids_chunk):
        time.sleep(3) 
        print(f"\n=== Processing {arxiv_id} ===")
        
        #5. Download the pdf
        pdf_io = download_pdf_to_bytes(arxiv_id)
        if not pdf_io:
            print(f" Could not download PDF for {arxiv_id}, skipping.")
            continue

        try:
            #6. Process the pdf
            results = process_pdf_bytes(arxiv_id, pdf_io)
            all_results.extend(results)
            with open(OUT_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            files_count += 1
            print(f" Finished {arxiv_id}: extracted {len(results)} items.")
            print(f"Total files processed: {files_count}")
        except Exception as e:
            print(f" Error processing {arxiv_id}: {e}")
