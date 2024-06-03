import pandas as pd
import os


def load_penn_treebank(data_dir):
    """
    Load and preprocess the Penn Treebank dataset.

    Args:
    - data_dir (str): Path to the directory containing the Penn Treebank files.

    Returns:
    - sentences (list of list of str): List of sentences, where each sentence is a list of words.
    - tags (list of list of str): List of tag sequences corresponding to the sentences.
    """
    sentences = []
    tags = []

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        sentence = []
        tag_sequence = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_sequence)
                    sentence = []
                    tag_sequence = []
            else:
                word, tag = line.split()
                sentence.append(word)
                tag_sequence.append(tag)
        if sentence:
            sentences.append(sentence)
            tags.append(tag_sequence)

    return sentences, tags


def load_conll2003(data_dir):
    """
    Load and preprocess the CoNLL 2003 dataset.

    Args:
    - data_dir (str): Path to the directory containing the CoNLL 2003 files.

    Returns:
    - sentences (list of list of str): List of sentences, where each sentence is a list of words.
    - tags (list of list of str): List of tag sequences corresponding to the sentences.
    """
    sentences = []
    tags = []

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        sentence = []
        tag_sequence = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_sequence)
                    sentence = []
                    tag_sequence = []
            else:
                word, tag = line.split()
                sentence.append(word)
                tag_sequence.append(tag)
        if sentence:
            sentences.append(sentence)
            tags.append(tag_sequence)

    return sentences, tags


def create_mappings(sentences, tags):
    """
    Create mappings from words/tags to indices and vice versa.

    Args:
    - sentences (list of list of str): List of sentences.
    - tags (list of list of str): List of tag sequences.

    Returns:
    - word_to_idx (dict): Mapping from words to indices.
    - tag_to_idx (dict): Mapping from tags to indices.
    """
    word_to_idx = {}
    tag_to_idx = {}

    for sentence, tag_sequence in zip(sentences, tags):
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
        for tag in tag_sequence:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

    return word_to_idx, tag_to_idx, idx_to_word, idx_to_tag


def convert_data(sentences, tags, word_to_idx, tag_to_idx):
    """
    Convert words and tags to their corresponding indices.

    Args:
    - sentences (list of list of str): List of sentences.
    - tags (list of list of str): List of tag sequences.
    - word_to_idx (dict): Mapping from words to indices.
    - tag_to_idx (dict): Mapping from tags to indices.

    Returns:
    - sentence_indices (list of list of int): List of sentences as indices.
    - tag_indices (list of list of int): List of tag sequences as indices.
    """
    sentence_indices = [[word_to_idx[word] for word in sentence] for sentence in sentences]
    tag_indices = [[tag_to_idx[tag] for tag in tag_sequence] for tag_sequence in tags]

    return sentence_indices, tag_indices
