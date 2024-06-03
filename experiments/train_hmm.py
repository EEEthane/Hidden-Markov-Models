import numpy as np
from models.hmm import HMM
from utils.data_loading import load_penn_treebank, create_mappings, convert_data


def main():
    # Load and preprocess the data
    data_dir = '../data/raw/penn_treebank'
    sentences, tags = load_penn_treebank(data_dir)

    # Create mappings
    word_to_idx, tag_to_idx, idx_to_word, idx_to_tag = create_mappings(sentences, tags)

    # Convert data to indices
    sentence_indices, tag_indices = convert_data(sentences, tags, word_to_idx, tag_to_idx)

    # Flatten the list of sentences and tags for HMM training
    observations = [word for sentence in sentence_indices for word in sentence]
    states = [tag for tag_seq in tag_indices for tag in tag_seq]

    # Define the number of states and observations
    num_states = len(tag_to_idx)
    num_observations = len(word_to_idx)

    # Initialize the HMM
    hmm = HMM(num_states, num_observations)

    # Train the HMM
    hmm.train(observations, max_iter=100)

    # Save the trained model parameters
    np.save('../models/transition_probs.npy', hmm.transition_probs)
    np.save('../models/emission_probs.npy', hmm.emission_probs)
    np.save('../models/initial_probs.npy', hmm.initial_probs)

    print("Training completed and model parameters saved.")


if __name__ == "__main__":
    main()
