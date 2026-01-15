# Quantum-Circuit Dataset Extraction

This project is an end-to-end NLP and data extraction pipeline designed to build a structured dataset of quantum circuit images and their metadata from research papers. It identifies, extracts, and filters relevant images and descriptions to prepare high-quality data for training Image-to-Text models.

# 🚀 Key Features

* Domain Classification: Uses Psuedo-labeling based Semi-Supervised approach, with the help of Sci-BERT model, to classify research papers into quantum circuit and non-quantum circuit domains, giving a precision of 95%.

* Automated Data Extraction: Processes research papers page-by-page to identify image captions and the corresponding circuit images located above them.

* Contextual Metadata Collection: Searches the entire document to gather related descriptive text for every extracted image.

* Intelligent Filtering: Uses a custom regex-based heuristic to score word importance, ensuring only relevant quantum circuit data is kept.

* Spatial Analysis: Employs spatial coordinates to accurately link captions with their respective images.

* High-Quality Output: Prioritizes reducing "False Positives" to ensure the resulting dataset is reliable for machine learning training. Final dataset achieves a precision of 80% (i.e. 80% out of 250 images are relevant quantum circuit images).