import numpy as np
from models.hmm import HMM
from utils.data_loading import load_penn_treebank, create_mappings, convert_data
from utils.metrics import accuracy


def main():
    # Load the test data
    data_dir = '../data/raw/penn_treebank'
    sentences, tags = load_penn_treebank(data_dir)

    # Create mappings (assuming the same mappings as during training)
    word_to_idx, tag_to_idx, idx_to_word, idx_to_tag = create_mappings(sentences, tags)

    # Convert data to indices
    sentence_indices, tag_indices = convert_data(sentences, tags, word_to_idx, tag_to_idx)

    # Flatten the list of sentences and tags for HMM evaluation
    test_observations = [word for sentence in sentence_indices for word in sentence]
    true_states = [tag for tag_seq in tag_indices for tag in tag_seq]

    # Load the trained model parameters
    transition_probs = np.load('../models/transition_probs.npy')
    emission_probs = np.load('../models/emission_probs.npy')
    initial_probs = np.load('../models/initial_probs.npy')

    # Initialize the HMM with the trained parameters
    num_states = transition_probs.shape[0]
    num_observations = emission_probs.shape[1]
    hmm = HMM(num_states, num_observations)
    hmm.transition_probs = transition_probs
    hmm.emission_probs = emission_probs
    hmm.initial_probs = initial_probs

    # Decode the test observations
    predicted_states = hmm.decode(test_observations)

    # Calculate the accuracy
    acc = accuracy(true_states, predicted_states)
    print(f"Decoding accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
