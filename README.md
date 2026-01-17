# Quantum-Circuit Dataset Extraction

This project is an end-to-end NLP and data extraction pipeline designed to build a structured dataset of quantum circuit images and their metadata from research papers. It identifies, extracts, and filters relevant images and descriptions to prepare high-quality data for training Image-to-Text models.

# Problem Statement

The goal of this project is to compile a dataset of 1000 quantum-circuit images, along with their metadata, like descriptions, gates used in the circuit, quantum problem being solved, etc. This data has to be extracted from a pool of 27000 multi-domain research papers. Additionally, the number of papers utilized to fetch this data has to be as low as possible.

# Example Quantum Circuit Images
<img width="503" height="247" alt="arXiv-2412 07844_p14_fig6" src="https://github.com/user-attachments/assets/12bc6668-dc20-40cd-87ba-4b25b1ecf65d" />
<img width="468" height="463" alt="arXiv-2505 09320_p10_fig15" src="https://github.com/user-attachments/assets/2c5181df-751e-4e47-bc86-45a784fa4cd3" />
<img width="891" height="432" alt="arXiv-2504 15841_p6_fig3" src="https://github.com/user-attachments/assets/957b4a8b-5de1-4f53-b41a-c94063565041" />





# 🚀 Key Features

* Domain Classification: Uses Psuedo-labeling based Semi-Supervised approach, with the help of Sci-BERT model, to classify research papers into quantum circuit and non-quantum circuit domains, giving a precision of 95%.

* Automated Data Extraction: Processes research papers page-by-page to identify image captions and the corresponding circuit images located above them.

* Contextual Metadata Collection: Searches the entire document to gather related descriptive text for every extracted image.

* Intelligent Filtering: Uses a custom regex-based heuristic to score word importance, ensuring only relevant quantum circuit data is kept.

* Spatial Analysis: Employs spatial coordinates to accurately link captions with their respective images.

* High-Quality Output: Prioritizes reducing "False Positives" to ensure the resulting dataset is reliable for machine learning training. Final dataset achieves a precision of 80% (i.e. 80% out of 1000 images are relevant quantum circuit images).
