from typing import List, Tuple


def process_history(history: List[Tuple[str, float]]) -> str:
    """
    Process the history to make it suitable for the feedback model.

    Args:
        history: The history to be processed.

    Returns:
        The processed history.
    """

    history_str = ""
    for i in range(len(history)):
        history_string += f"""
    ### Edit Proposed
        {history[i][0]}
    ### Accuracy Change
        {history[i][1]}
    """

    return history_str
