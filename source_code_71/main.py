import os
import re
import requests
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image, UnidentifiedImageError
import json
import requests
import time
import spacy
import spacy.cli
from regex_patterns import POSITIVE_PATTERN, NEGATIVE_PATTERN, QUANTUM_GATES_PATTERN, QUANTUM_ALGO_PROBLEM_PATTERN, ABBREVIATION_MAPPING, GATE_NORMALIZATION_MAP, CAPTION_REGEX
import csv


# Output directories and global variables
OUT_IMAGES_DIR = "./images_71"
OUT_DATA_FILE = "./dataset_71.json"
ARXIV_IDS_FILE = "./paper_list_71.txt"
RENDER_DPI = 300
IMAGE_LIMIT = 250
CSV_FILE = "./paper_list_counts_71.csv"
DOWNLOAD_TIMEOUT = 30



def reproduce_text_from_positions(arxiv_id, text_positions):
    """
    Downloads an arXiv PDF, reconstructs the document text precisely as in the original pipeline,
    and extracts text snippets based on the provided (start, end) index pairs.
    
    Parameters
    ----------
    arxiv_id : str
        The arXiv ID of the paper to download.
    text_positions : list
        A list of list/tuples containing [start_index, end_index].

    Returns
    -------
    list
        A list of extracted strings.
    """
    # 1. Download PDF
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    print(f"Downloading {url}...")
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Failed to download PDF: HTTP {resp.status_code}")
    
    # 2. Reconstruct Full Text
    doc = fitz.open(stream=resp.content, filetype="pdf")
    full_document_text = ""
    
    # 3. Reconstruct Full Text
    for pno in range(doc.page_count):
        page = doc[pno]
        blocks = page.get_text("blocks")
        for b in blocks:
            text_content = b[4].strip()
            if not text_content:
                continue
            full_document_text += text_content + "\n"
            
    doc.close()
    
    # 3. Extract Snippets
    reproduced_texts = []
    for start, end in text_positions:
        # Slice the full text
        snippet = full_document_text[start:end]
        # Standard cleaning often applied in the original script: strip and space normalization
        # However, to "reproduce" exactly what was likely in the JSON, 
        # we return the slice as is, or with minor cleanup if the user wants parity.
        snippet = clean_all_text(snippet)
        reproduced_texts.append(snippet.strip())
        
    return reproduced_texts

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



def check_quantum_caption(caption_text):
    """
    Check if the caption text is a caption of a quantum circuit figure

    Parameters
    ----------
    caption_text : str
        The caption text to check.

    Returns
    -------
    bool
        True if the caption text is a quantum caption, False otherwise.
    """
    caption_clean = clean_all_text(caption_text)
    return bool(POSITIVE_PATTERN.search(caption_clean) and not NEGATIVE_PATTERN.search(caption_clean))


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
                    
                    desc_text_cleaned = clean_all_text(desc_text)
                    descriptions.append(desc_text_cleaned)
                    text_positions.append((doc_start, doc_end))

    return descriptions, text_positions, caption_out_positions




def crop_and_rasterize(ca, text_blocks, page, fig_number, arxiv_id, fig_counter):
    """
    Crop and get the bounding boxes of the figure

    Parameters
    ----------
    ca : dict
        The caption block.
    text_blocks : list
        The list of text blocks.
    page : fitz.Page
        The page.
    fig_number : int
        The figure number.
    arxiv_id : str
        The arxiv id.
    fig_counter : int
        The figure counter.

    Returns
    -------
    crop_rect : fitz.Rect
        The crop rectangle.
    """

    # ChatGPT generated code: To extract the image from the paper
    cap_y0 = ca["cap_rect"].y0
    crop_y1 = max(1, cap_y0 - 2)
    found_top_limit = False
    prev_y = 0.0
    safe_zone_top = max(0, cap_y0 - 200)
    current_tb_idx = -1
    for i, tb in enumerate(text_blocks):
        if tb["rect"] == ca["cap_rect"]:
            current_tb_idx = i
            break
    if current_tb_idx > 0:
        for i in range(current_tb_idx - 1, -1, -1):
            tb = text_blocks[i]
            txt = tb["text"].strip()
            if tb["rect"].y1 > safe_zone_top:
                continue
            is_long = len(txt) > 30
            is_sentence = txt.endswith(( '?', '!'))
            if is_long or is_sentence:
                prev_y = tb["rect"].y1
                found_top_limit = True
                break
    if not found_top_limit:
            prev_y = max(0.0, crop_y1 - 400)
    crop_y0 = min(prev_y, safe_zone_top)
    crop_y0 = max(0.0, crop_y0)
    if crop_y0 >= crop_y1:
            crop_y0 = max(0.0, crop_y1 - 100)

    margin_w = 10
    crop_x0 = max(0, ca["cap_rect"].x0 - margin_w)
    crop_x1 = min(page.rect.width, ca["cap_rect"].x1 + margin_w)
    crop_rect = fitz.Rect(crop_x0, crop_y0, crop_x1, crop_y1)

    return crop_rect




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


            fig_counter = 1
            # 8) Traverse the caption blocks
            for ca in caption_blocks:
                caption_text = ca["text"]
                caption_rect = ca["cap_rect"]

                # 9) Check if the caption is a caption for a quantum circuit figure. If not, skip to the next caption block.
                caption_decision = check_quantum_caption(caption_text)
                if not caption_decision:
                    continue

                # 10) Extract the figure number from the caption text
                fig_match = CAPTION_REGEX.search(caption_text)
                fig_number = fig_match.group(2) if fig_match else None

                descriptions = []
                text_positions = []

                caption_out_text = caption_text
                caption_out_positions = ()

                # 11) Extract other descriptive texts which are related to the quantum circuit figure along with their positions
                descriptions, text_positions, caption_out_positions = extract_descriptions_with_pos(all_blocks_with_pos, fig_number, caption_text, caption_rect, page_num, descriptions, text_positions)

                caption_out_text = clean_all_text(caption_out_text)
                descriptions.append(caption_out_text)
                if caption_out_positions:
                    text_positions.append(caption_out_positions)
                else:
                    text_positions.append((0, 0)) # Fallback if not found

                # 12) Get the quantum problem and quantum gates from all the descriptive texts
                quantum_problem = get_quantum_problem(descriptions, caption_out_text)
                gates = get_quantum_gates(descriptions)

                def add_result(img_path, img_ext, img_name_only):
                    results.append({
                        "image_filename": img_name_only,
                        "arxiv paper no.": arxiv_id.replace("-",":"),
                        "page number": page_num,
                        "figure number": int(fig_number) if fig_number and fig_number.isdigit() else "",
                        "quantum gates": gates,
                        "quantum problem": quantum_problem,
                        "descriptions": descriptions,
                        "text positions": text_positions
                    })

                # 13) Crop and rasterize the figure above the caption
                crop_rect = crop_and_rasterize(ca, text_blocks, page, fig_number, arxiv_id, fig_counter)

                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                try:
                    pix = page.get_pixmap(matrix=mat, clip=crop_rect, alpha=False)
                    fname_fig_num = fig_number if fig_number else f"fallback_{fig_counter}"
                    base_name = f"{arxiv_id.replace('/','_')}_p{page_num}_fig{fname_fig_num}.png"
                    out_img_path = os.path.join(OUT_IMAGES_DIR, base_name)
                    pix.save(out_img_path)
                    add_result(out_img_path, "png", base_name)
                except Exception as e:
                    print(f"Error rasterizing region: {e}")

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

    #1. Download the utilities for spacy and nltk
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
    #2. Create the output directory for images
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
    
    #3. Read the list of arxiv ids
    with open(ARXIV_IDS_FILE, "r", encoding="utf-8") as f:
        arxiv_ids = [line.strip() for line in f if line.strip()]
    files_count = 0
    images_count = 0
    all_results = {}
    csv_data = []
    last_idx = 0
    
    #4. Traverse the list of arxiv ids
    for idx, arxiv_id in enumerate(arxiv_ids):
        if images_count == IMAGE_LIMIT:
            last_idx = idx
            print("Image limit reached, stopping.")
            break
        time.sleep(3) 
        print(f"\n=== Processing {arxiv_id} ===")
        
        #5. Download the pdf
        pdf_io = download_pdf_to_bytes(arxiv_id)
        if not pdf_io:
            print(f" Could not download PDF for {arxiv_id}, skipping.")
            csv_data.append({
                "arxiv paper no.": arxiv_id.replace("-", ":"),
                "images_count": ""
            })
            continue

        try:
            #6. Process the pdf
            results = process_pdf_bytes(arxiv_id, pdf_io)
            if images_count + len(results) >= IMAGE_LIMIT:
                results = results[:IMAGE_LIMIT - images_count]
            
            #7. Update the results from each pdf
            for item in results:
                image_file = item["image_filename"]
                all_results[image_file] = {
                        "arxiv paper no.":item["arxiv paper no."],
                        "page number": item["page number"],
                        "figure number": item["figure number"],
                        "quantum gates": item["quantum gates"],
                        "quantum problem": item["quantum problem"],
                        "descriptions": item["descriptions"],
                        "text positions": item["text positions"]
                    }
            with open(OUT_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            files_count += 1
            images_count += len(results)
            csv_data.append({
                "arxiv paper no.": arxiv_id.replace("-", ":"),
                "images_count": len(results)
            })
            print(f" Finished {arxiv_id}: extracted {len(results)} items.")
            print(f"Total files processed: {files_count}, total images extracted: {images_count}")
        except Exception as e:
            print(f" Error processing {arxiv_id}: {e}")

    for i in range(last_idx,len(arxiv_ids)):
        csv_data.append({
            "arxiv paper no.": arxiv_ids[i].replace("-", ":"),
            "images_count": ""
        })

    with open(CSV_FILE, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)