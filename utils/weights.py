import logging
import hashlib
import numpy as np
import torch
import torch.nn as nn


def _get_numpy_dtype(dtype_name: str):
    """
    Convert dtype name string to NumPy dtype.
    """
    dtype_name = dtype_name.lower()

    if dtype_name == "float32":
        return np.float32

    if dtype_name == "float64":
        return np.float64

    raise ValueError(f"Unsupported weight dtype: {dtype_name}")


def weights_to_bytes(model: nn.Module, dtype_name: str) -> bytes:
    """
    Serialize all trainable model parameters into bytes.

    Args:
        model: PyTorch model
        dtype_name: "float32" or "float64"

    Returns:
        Serialized model weights as bytes.
    """
    target_dtype = _get_numpy_dtype(dtype_name)

    arrays = []

    for param in model.parameters():
        arr = param.detach().cpu().numpy().astype(target_dtype, copy=True).flatten()
        arrays.append(arr)

    if not arrays:
        raise ValueError("Model has no parameters to serialize.")

    flat = np.concatenate(arrays)
    data = flat.tobytes()

    logging.debug(
        f"Converted model weights to bytes | "
        f"elements={flat.size} | size={len(data)} bytes | dtype={dtype_name}"
    )

    return data


def bytes_to_weight_arrays(
    data: bytes,
    template_model: nn.Module,
    dtype_name: str,
) -> list[np.ndarray]:
    """
    Reconstruct model weight arrays from serialized bytes.

    Args:
        data: Serialized weight bytes
        template_model: Model used only for parameter shapes
        dtype_name: "float32" or "float64"

    Returns:
        List of NumPy arrays matching model parameter shapes.
    """
    source_dtype = _get_numpy_dtype(dtype_name)

    dtype_size = np.dtype(source_dtype).itemsize

    if len(data) % dtype_size != 0:
        raise ValueError(
            f"Invalid byte length {len(data)} for dtype {dtype_name} "
            f"with item size {dtype_size}."
        )

    flat = np.frombuffer(data, dtype=source_dtype).copy()

    shapes = [tuple(param.shape) for param in template_model.parameters()]
    expected_elements = sum(int(np.prod(shape)) for shape in shapes)

    if flat.size != expected_elements:
        raise ValueError(
            f"Weight size mismatch: received {flat.size} elements, "
            f"but template model expects {expected_elements} elements."
        )

    arrays = []
    idx = 0

    for shape in shapes:
        n = int(np.prod(shape))
        arr = flat[idx: idx + n].reshape(shape)
        arrays.append(arr)
        idx += n

    logging.debug(
        f"Reconstructed weight arrays | "
        f"total_elements={flat.size} | num_tensors={len(arrays)} | dtype={dtype_name}"
    )

    return arrays


def apply_weight_arrays(model: nn.Module, arrays: list[np.ndarray]):
    """
    Apply reconstructed NumPy weight arrays to a PyTorch model.
    """
    params = list(model.parameters())

    if len(params) != len(arrays):
        raise ValueError(
            f"Parameter count mismatch: model has {len(params)} tensors, "
            f"but received {len(arrays)} arrays."
        )

    with torch.no_grad():
        for param, arr in zip(params, arrays):
            arr_tensor = torch.from_numpy(arr).to(
                device=param.device,
                dtype=param.dtype,
            )

            if tuple(param.shape) != tuple(arr_tensor.shape):
                raise ValueError(
                    f"Shape mismatch: parameter shape {tuple(param.shape)} "
                    f"but received array shape {tuple(arr_tensor.shape)}."
                )

            param.data.copy_(arr_tensor)

    logging.debug("Applied weight arrays to model")


def model_to_weight_arrays(model: nn.Module) -> list[np.ndarray]:
    """
    Convert model parameters to a list of NumPy arrays.
    """
    arrays = [
        param.detach().cpu().numpy().copy()
        for param in model.parameters()
    ]

    logging.debug(f"Converted model to weight arrays | tensors={len(arrays)}")

    return arrays


def hash_bytes(data: bytes, hash_algorithm: str = "sha256") -> bytes:
    """
    Hash arbitrary bytes using SHA-256 or SHA-512.
    """
    hash_algorithm = hash_algorithm.lower()

    if hash_algorithm == "sha256":
        return hashlib.sha256(data).digest()

    if hash_algorithm == "sha512":
        return hashlib.sha512(data).digest()

    raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")


def hash_weights(model: nn.Module, dtype_name: str, hash_algorithm: str = "sha256") -> bytes:
    """
    Hash serialized model weights.

    This is useful before Dilithium signing:
        model weights -> bytes -> hash -> Dilithium signature
    """
    weight_bytes = weights_to_bytes(model, dtype_name=dtype_name)
    digest = hash_bytes(weight_bytes, hash_algorithm)

    logging.debug(
        f"Hashed model weights | algorithm={hash_algorithm} | digest_size={len(digest)} bytes"
    )

    return digest