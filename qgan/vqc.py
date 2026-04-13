import pennylane as qml


def simple_circuit(weights, noise, n_qubits=4, n_layers=3):
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
        qml.RY(noise[i], wires=i)
    
    # 2. Trainable Variational Layers
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        
        # Entanglement (Crucial for spatial correlation in images)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
    # 3. Measurement (Returns expectation values as pixel intensities)
    return qml.probs(wires=range(n_qubits))

def strongly_entangling_circuit(weights, noise, n_qubits=4):
    """
    Construct a parameterized quantum circuit with angle embedding and
    strongly entangling layers.

    The input ``noise`` is encoded into a quantum state using angle embedding.
    A sequence of trainable strongly entangling layers is then applied.
    The function returns the final quantum statevector.

    Parameters
    ----------
    weights : jnp.ndarray
        Trainable parameters for the strongly entangling layers. Expected shape
        is (n_layers, n_qubits, 3), where each parameter corresponds to a
        rotation angle.
    noise : jnp.ndarray
        Input features to encode into the quantum circuit. Its length must be
        equal to ``n_qubits``.
    n_qubits : int, optional
        Number of qubits (wires) used in the circuit. Default is 4.

    Returns
    -------
    state : ndarray
        Complex-valued statevector of shape (2**n_qubits,) representing the
        final quantum state of the circuit.

    Notes
    -----
    - Uses ``qml.AngleEmbedding`` for data encoding.
    - Uses ``qml.StronglyEntanglingLayers`` as the variational ansatz.
    - Returns the full statevector via ``qml.state()`` rather than measurement
      probabilities or expectation values.
    """
    # Encoding: Transform noise into quantum state
    # We use Angle Embedding for simplicity here
    qml.AngleEmbedding(noise, wires=range(n_qubits))
    
    # Variational Layers (The "Brain" of the generator)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Return probabilities (16 states for 16 pixels)
    # return qml.probs(wires=range(n_qubits))
    # return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2)), qml.expval(qml.Z(3))
    return qml.state()