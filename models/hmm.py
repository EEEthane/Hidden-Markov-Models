import numpy as np

class HMM:
    def __init__(self, num_states, num_observations):
        self.num_states = num_states
        self.num_observations = num_observations
        self.transition_probs = np.full((num_states, num_states), 1/num_states)
        self.emission_probs = np.full((num_states, num_observations), 1/num_observations)
        self.initial_probs = np.full(num_states, 1/num_states)

    def _forward(self, observations):
        T = len(observations)
        alpha = np.zeros((T, self.num_states))
        alpha[0] = self.initial_probs * self.emission_probs[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.num_states):
                alpha[t, j] = self.emission_probs[j, observations[t]] * np.sum(alpha[t-1] * self.transition_probs[:, j])
        
        return alpha

    def _backward(self, observations):
        T = len(observations)
        beta = np.zeros((T, self.num_states))
        beta[T-1] = 1
        
        for t in range(T-2, -1, -1):
            for i in range(self.num_states):
                beta[t, i] = np.sum(beta[t+1] * self.transition_probs[i, :] * self.emission_probs[:, observations[t+1]])
        
        return beta

    def _baum_welch(self, observations, max_iter=100):
        T = len(observations)
        
        for _ in range(max_iter):
            alpha = self._forward(observations)
            beta = self._backward(observations)
            
            xi = np.zeros((T-1, self.num_states, self.num_states))
            for t in range(T-1):
                denominator = np.sum(alpha[t] * self.transition_probs * self.emission_probs[:, observations[t+1]].T * beta[t+1])
                for i in range(self.num_states):
                    numerator = alpha[t, i] * self.transition_probs[i, :] * self.emission_probs[:, observations[t+1]] * beta[t+1]
                    xi[t, i, :] = numerator / denominator
            
            gamma = np.sum(xi, axis=2)
            gamma = np.vstack((gamma, np.sum(xi[T-2, :, :], axis=0)))
            
            self.initial_probs = gamma[0]
            self.transition_probs = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0).reshape(-1, 1)
            for obs in range(self.num_observations):
                mask = observations == obs
                self.emission_probs[:, obs] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)
    
    def train(self, observations, max_iter=100):
        self._baum_welch(observations, max_iter)

    def decode(self, observations):
        T = len(observations)
        delta = np.zeros((T, self.num_states))
        psi = np.zeros((T, self.num_states), dtype=int)
        
        delta[0] = self.initial_probs * self.emission_probs[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.num_states):
                probabilities = delta[t-1] * self.transition_probs[:, j]
                psi[t, j] = np.argmax(probabilities)
                delta[t, j] = np.max(probabilities) * self.emission_probs[j, observations[t]]
        
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states

