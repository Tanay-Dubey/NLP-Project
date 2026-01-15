import numpy as np
import pandas as pd
import re
from collections import Counter
import unicodedata
import shutil
import os
import json
import nltk
from nltk.corpus import stopwords
from regex_patterns import (
    POSITIVE_PATTERN,
    QUANTUM_GATES_PATTERN,
    NEGATIVE_PATTERN
)

from sklearn.feature_extraction.text import TfidfVectorizer

def process_captions(df):
    """
    Process and clean the captions in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the captions.

    Returns
    -------
    matched_captions : pd.DataFrame
        The DataFrame containing the positive captions.
    negative_df : pd.DataFrame
        The DataFrame containing the negative samples.
    """
    print("Shape of the DataFrame is: ", df.shape)
    df['caption_clean'] = df['caption text'].str.lower()
    df['caption_clean'] = (
        df['caption_clean']
        # fix hyphenation from line breaks
        .str.replace(r'-\s+', '', regex=True)
        # remove Figure / Fig at start of caption
        .str.replace(
            r'^\s*(figure|fig)\s*\.?\s*\d*\s*[:.\-]\s*',
            '',
            regex=True,
            case=False
        )
        # remove sub-labels like "(a)", "(b)" at start of each sentence
        .str.replace(
            r'(^|[.!?]\s+)\([a-zA-Z]\)\s*',
            r'\1',
            regex=True
        )
    )

    matched_captions = df[
        (
            df["caption_clean"].str.contains(POSITIVE_PATTERN, na=False)

        )
        &
        ~(
            df["caption_clean"].str.contains(NEGATIVE_PATTERN, na=False)
        )
        ].copy()

    print(f"Found {len(matched_captions)} high-confidence quantum circuit rows")

    negative_df = df[~df["fig counter"].isin(matched_captions['fig counter'])]
    negative_df["caption_clean"] = (
        negative_df["caption_clean"]
        .fillna("")
        .str.replace(r"\b[a-zA-Z]\b", "", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )

    return matched_captions, negative_df


def get_important_bigrams_list(positive_df,negative_df):
    """
    Get the list of important bigrams in the filtered quantum circuit dataset.

    Parameters
    ----------
    positive_df : pd.DataFrame
        The DataFrame containing the positive samples.
    negative_df : pd.DataFrame
        The DataFrame containing the negative samples.

    Returns
    -------
    tfidf_df : pd.DataFrame
        The DataFrame containing the list of important bigrams.
    """
    vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words="english")
    tfidf_positive = vectorizer.fit_transform(positive_df["caption_clean"])
    tfidf_negative = vectorizer.transform(negative_df["caption_clean"])

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_positive.sum(axis=0).A1

    tfidf_df = pd.DataFrame({
        "bigram": feature_names,
        "tfidf_score": scores
    }).sort_values("tfidf_score", ascending=False)

    print("\n\n----List of important bigrams in filtered quantum circuit dataset----")
    print(tfidf_df.head(20))

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_negative.sum(axis=0).A1

    tfidf_negative_df = pd.DataFrame({
        "bigram": feature_names,
        "tfidf_score": scores
    }).sort_values("tfidf_score", ascending=False)
    print("\n\n----List of important bigrams in remaining non-quantum circuit dataset----")
    print(tfidf_negative_df.head(20))


def get_important_words_list(positive_df,negative_df):
    """
    Get the list of important words in the filtered quantum circuit dataset.

    Parameters
    ----------
    positive_df : pd.DataFrame
        The DataFrame containing the positive samples.
    negative_df : pd.DataFrame
        The DataFrame containing the negative samples.

    Returns
    -------
    tfidf_df : pd.DataFrame
        The DataFrame containing the list of important words.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_positive = vectorizer.fit_transform(positive_df["caption_clean"])
    tfidf_negative = vectorizer.transform(negative_df["caption_clean"])

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_positive.sum(axis=0).A1

    tfidf_df = pd.DataFrame({
        "word": feature_names,
        "tfidf_score": scores
    }).sort_values("tfidf_score", ascending=False)

    print("----List of important words in filtered quantum circuit dataset----")
    print(tfidf_df.head(20))


    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_negative.sum(axis=0).A1

    tfidf_negative_df = pd.DataFrame({
        "word": feature_names,
        "tfidf_score": scores
    }).sort_values("tfidf_score", ascending=False)
    print("\n\n----List of important words in remaining non-quantum circuit dataset----")
    print(tfidf_negative_df.head(20))


def preprocessing(df):
    """
    Preprocesses the caption text by removing hyphens and cleaning up whitespace.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the caption text.

    Returns
    -------
    df : pd.DataFrame
        The preprocessed DataFrame.
    """
    df['caption text'] = df['caption text'].str.replace(r'-\s+', '', regex=True)
    df["caption text"] = (
        df["caption text"]
        .fillna("")
        .str.replace(r"\s*\n\s*", " ", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )

    df['descriptions'] = df['descriptions'].str.replace(r'-\s+', '', regex=True)
    df["descriptions"] = (
        df["descriptions"]
        .fillna("")
        .str.replace(r"\s*\n\s*", " ", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )

    df['joined_descriptions'] = df['descriptions'].apply(
    lambda lst: " ".join(lst) if isinstance(lst, list) else lst
    )
    return df



if __name__ == "__main__":
    nltk.download('stopwords')
    df = pd.read_json("./captions_for_regex.json")
    df = preprocessing(df)
    positive_df, negative_df = process_captions(df)
    get_important_words_list(positive_df,negative_df)
    get_important_bigrams_list(positive_df,negative_df)