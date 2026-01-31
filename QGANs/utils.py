import torch
from qiskit_machine_learning.utils import algorithm_globals

def set_seed(seed: int):
    """
    Set the random seed for reproducibility across torch and Qiskit.
    
    Parameters
    ----------
    seed : int
        The seed value to set.
    """
    torch.manual_seed(seed)
    algorithm_globals.random_seed = seed

    return algorithm_globals.random_seed