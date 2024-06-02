import numpy as np
from models.hmm import HMM
from utils.data_preprocessing import preprocess_data
from utils.metrics import accuracy

def main():
    # Load the test data
    test_observations = preprocess_data('data/raw/test.txt')
    
    # Load the trained model parameters
    transition_probs = np.load('models/transition_probs.npy')
    emission_probs = np.load('models/emission_probs.npy')
    initial_probs = np.load('models/initial_probs.npy')
    
    # Initialize the HMM with the trained parameters
    num_states = transition_probs.shape[0]
    num_observations = emission_probs.shape[1]
    hmm = HMM(num_states, num_observations)
    hmm.transition_probs = transition_probs
    hmm.emission_probs = emission_probs
    hmm.initial_probs = initial_probs
    
    # Decode the test observations
    decoded_states = hmm.decode(test_observations)
    
    # Assume we have true states to compare with (for evaluation purposes)
    true_states = preprocess_data('data/raw/test_states.txt')
    
    # Calculate the accuracy
    acc = accuracy(true_states, decoded_states)
    print(f"Decoding accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
