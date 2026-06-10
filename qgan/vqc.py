import pennylane as qml

def pqwgan_circuit(weights, noise, n_qubits=4, a_qubits=1, n_layers=3):
    """
    Construct a simple parameterized quantum circuit with layered rotations
    and nearest-neighbor entanglement.

    The input ``noise`` is first encoded into the quantum state using RY
    rotations. This is followed by multiple layers of trainable single-qubit
    rotations and CNOT-based entanglement. The circuit outputs measurement
    probabilities over all computational basis states.

    Parameters
    ----------
    weights : jnp.ndarray
        Trainable parameters for the variational layers. Expected shape is
        (n_layers, n_qubits, 3), where each triplet corresponds to RX, RY,
        and RZ rotation angles for a qubit.
    noise : jnp.ndarray
        Input features used to initialize the quantum state via RY rotations.
        Its length must be equal to ``n_qubits``.
    n_qubits : int, optional
        Number of qubits (wires) in the circuit. Default is 4.
    a_qubits : int, optional
        Number of ancillary qubits used for entanglement. Default is 1.
    n_layers : int, optional
        Number of variational layers applied in the circuit. Default is 3.

    Returns
    -------
    probs : ndarray
        Probability distribution over computational basis states. The output
        has shape (2**n_qubits,) and can be interpreted as, e.g., pixel
        intensities in generative tasks.

    Notes
    -----
    - Encoding is performed using ``qml.RY`` rotations.
    - Each layer consists of RX, RY, and RZ rotations per qubit.
    - Entanglement is introduced via a ring of CNOT gates.
    - Returns probabilities via ``qml.probs()`` rather than a statevector.
    """

    # 1. Latent Space Mapping (Encoding Noise)
    for i in range(n_qubits):
        qml.RY(noise[:, i], wires=i)
    
    # 2. Trainable Variational Layers
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RZ(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        
        # Entanglement (Crucial for spatial correlation in images)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
    # 3. Measurement 
    return qml.probs(wires=range(n_qubits))