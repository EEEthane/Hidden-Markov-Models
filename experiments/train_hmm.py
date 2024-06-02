import numpy as np
from models.hmm import HMM
from utils.data_preprocessing import preprocess_data

def main():
    # Load and preprocess the data
    observations = preprocess_data('data/raw/train.txt')
    
    # Define the number of states and observations
    num_states = 5  # Example number of states
    num_observations = len(set(observations))  # Number of unique observations
    
    # Initialize the HMM
    hmm = HMM(num_states, num_observations)
    
    # Train the HMM
    hmm.train(observations, max_iter=100)
    
    # Save the trained model parameters
    np.save('models/transition_probs.npy', hmm.transition_probs)
    np.save('models/emission_probs.npy', hmm.emission_probs)
    np.save('models/initial_probs.npy', hmm.initial_probs)
    
    print("Training completed and model parameters saved.")

if __name__ == "__main__":
    main()
