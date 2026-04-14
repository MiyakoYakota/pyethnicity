import os
import numpy as np
import onnxruntime as ort

LABELS = ("asian", "black", "hispanic", "white")
MAXLEN = 10

_CHAR_MAP = {}
for i, c in enumerate("Eabcdefghijklmnopqrstuvwxyz U"):
    _CHAR_MAP[c] = i
_UNKNOWN_ID = 28

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_sessions = {}


def _get_session(model_name):
    if model_name not in _sessions:
        path = os.path.join(_MODEL_DIR, f"{model_name}.onnx")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found: {path}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _sessions[model_name] = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return _sessions[model_name]


def _encode_name(name, maxlen=MAXLEN):
    ids = []
    for c in name.lower()[:maxlen]:
        ids.append(_CHAR_MAP.get(c, _UNKNOWN_ID))
    ids += [0] * (maxlen - len(ids))
    return ids


def _encode_lastname_batch(lastnames):
    return np.array([_encode_name(n, MAXLEN) for n in lastnames], dtype=np.float32)


def _encode_fullname_batch(firstnames, lastnames):
    batch = []
    for fn, ln in zip(firstnames, lastnames):
        batch.append(_encode_name(fn, MAXLEN) + _encode_name(ln, MAXLEN))
    return np.array(batch, dtype=np.float32)


def predict_lastname(lastnames):
    if isinstance(lastnames, str):
        lastnames = [lastnames]

    session = _get_session("lastname_distill")
    input_name = session.get_inputs()[0].name
    encoded = _encode_lastname_batch(lastnames)
    probs = session.run(None, {input_name: encoded})[0]

    # normalize rows.should already sum to ~1 from softmax, but match R
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = probs / row_sums

    results = []
    for i, name in enumerate(lastnames):
        p = probs[i]
        results.append({
            "lastname": name,
            "prob_asian": float(p[0]),
            "prob_black": float(p[1]),
            "prob_hispanic": float(p[2]),
            "prob_white": float(p[3]),
            "race": LABELS[int(np.argmax(p))],
        })
    return results


def predict_fullname(firstnames, lastnames):
    if isinstance(firstnames, str):
        firstnames = [firstnames]
    if isinstance(lastnames, str):
        lastnames = [lastnames]
    if len(firstnames) != len(lastnames):
        raise ValueError("firstnames and lastnames must be the same length")

    session = _get_session("fullname_aligned_distill")
    input_name = session.get_inputs()[0].name
    encoded = _encode_fullname_batch(firstnames, lastnames)
    probs = session.run(None, {input_name: encoded})[0]

    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = probs / row_sums

    results = []
    for i in range(len(firstnames)):
        p = probs[i]
        results.append({
            "firstname": firstnames[i],
            "lastname": lastnames[i],
            "prob_asian": float(p[0]),
            "prob_black": float(p[1]),
            "prob_hispanic": float(p[2]),
            "prob_white": float(p[3]),
            "race": LABELS[int(np.argmax(p))],
        })
    return results


def predict_ethnicity(firstnames=None, lastnames=None, method="fullname"):
    if method == "fullname":
        if firstnames is None or lastnames is None:
            raise ValueError("Both firstnames and lastnames are required for method='fullname'")
        return predict_fullname(firstnames, lastnames)
    elif method == "lastname":
        if lastnames is None:
            raise ValueError("lastnames is required for method='lastname'")
        return predict_lastname(lastnames)
    else:
        raise ValueError(f"method must be 'fullname' or 'lastname', got '{method}'")
