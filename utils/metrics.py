def accuracy(true_states, predicted_states):
    """
    Calculate the accuracy of the predicted states compared to the true states.

    Args:
    - true_states (list of int): The true states.
    - predicted_states (list of int): The predicted states.

    Returns:
    - (float): The accuracy percentage.
    """
    correct = sum(t == p for t, p in zip(true_states, predicted_states))
    return (correct / len(true_states)) * 100

def precision(true_states, predicted_states, target_state):
    """
    Calculate the precision for a specific target state.

    Args:
    - true_states (list of int): The true states.
    - predicted_states (list of int): The predicted states.
    - target_state (int): The state for which precision is calculated.

    Returns:
    - (float): Precision for the target state.
    """
    true_positives = sum((t == p == target_state) for t, p in zip(true_states, predicted_states))
    predicted_positives = sum((p == target_state) for p in predicted_states)
    return true_positives / predicted_positives if predicted_positives != 0 else 0

def recall(true_states, predicted_states, target_state):
    """
    Calculate the recall for a specific target state.

    Args:
    - true_states (list of int): The true states.
    - predicted_states (list of int): The predicted states.
    - target_state (int): The state for which recall is calculated.

    Returns:
    - (float): Recall for the target state.
    """
    true_positives = sum((t == p == target_state) for t, p in zip(true_states, predicted_states))
    actual_positives = sum((t == target_state) for t in true_states)
    return true_positives / actual_positives if actual_positives != 0 else 0

def f1_score(true_states, predicted_states, target_state):
    """
    Calculate the F1 score for a specific target state.

    Args:
    - true_states (list of int): The true states.
    - predicted_states (list of int): The predicted states.
    - target_state (int): The state for which F1 score is calculated.

    Returns:
    - (float): F1 score for the target state.
    """
    prec = precision(true_states, predicted_states, target_state)
    rec = recall(true_states, predicted_states, target_state)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
