import unittest
import numpy as np
from models.hmm import HMM

class TestHMM(unittest.TestCase):

    def setUp(self):
        self.num_states = 2
        self.num_observations = 3
        self.hmm = HMM(self.num_states, self.num_observations)
        self.observations = [0, 1, 2, 1, 0]

    def test_initial_probabilities(self):
        initial_probs_sum = np.sum(self.hmm.initial_probs)
        self.assertAlmostEqual(initial_probs_sum, 1.0, places=5, msg="Initial probabilities do not sum to 1")

    def test_forward_algorithm(self):
        alpha = self.hmm._forward(self.observations)
        self.assertEqual(alpha.shape, (len(self.observations), self.num_states), "Forward algorithm output shape is incorrect")

    def test_backward_algorithm(self):
        beta = self.hmm._backward(self.observations)
        self.assertEqual(beta.shape, (len(self.observations), self.num_states), "Backward algorithm output shape is incorrect")

    def test_baum_welch_training(self):
        self.hmm.train(self.observations, max_iter=10)
        transition_probs_sum = np.sum(self.hmm.transition_probs, axis=1)
        emission_probs_sum = np.sum(self.hmm.emission_probs, axis=1)
        for i in range(self.num_states):
            self.assertAlmostEqual(transition_probs_sum[i], 1.0, places=5, msg="Transition probabilities for state {} do not sum to 1".format(i))
            self.assertAlmostEqual(emission_probs_sum[i], 1.0, places=5, msg="Emission probabilities for state {} do not sum to 1".format(i))

    def test_viterbi_decoding(self):
        self.hmm.train(self.observations, max_iter=10)
        states = self.hmm.decode(self.observations)
        self.assertEqual(len(states), len(self.observations), "Decoded states length does not match observations length")

if __name__ == '__main__':
    unittest.main()
