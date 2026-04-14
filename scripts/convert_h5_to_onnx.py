#!/usr/bin/env python3
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def build_model(seq_len):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(29, 32, input_length=seq_len, name="embedding"),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
            name="bi_lstm_1",
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, dropout=0.2),
            name="bi_lstm_2",
        ),
        tf.keras.layers.Dense(4, activation="softmax", name="dense"),
    ])
    model.build(input_shape=(None, seq_len))
    return model


def load_weights_from_h5(model, h5_path):
    import h5py

    with h5py.File(h5_path, "r") as f:
        mw = f["model_weights"]

        # Embedding
        emb = np.array(mw["embedding_1/embedding_1/embeddings:0"])

        # BiLSTM 1 forward then backward
        fwd1_k = np.array(mw["bidirectional_4/bidirectional_4/forward_lstm_4/lstm_cell_13/kernel:0"])
        fwd1_rk = np.array(mw["bidirectional_4/bidirectional_4/forward_lstm_4/lstm_cell_13/recurrent_kernel:0"])
        fwd1_b = np.array(mw["bidirectional_4/bidirectional_4/forward_lstm_4/lstm_cell_13/bias:0"])
        bwd1_k = np.array(mw["bidirectional_4/bidirectional_4/backward_lstm_4/lstm_cell_14/kernel:0"])
        bwd1_rk = np.array(mw["bidirectional_4/bidirectional_4/backward_lstm_4/lstm_cell_14/recurrent_kernel:0"])
        bwd1_b = np.array(mw["bidirectional_4/bidirectional_4/backward_lstm_4/lstm_cell_14/bias:0"])

        # BiLSTM 2
        fwd2_k = np.array(mw["bidirectional_5/bidirectional_5/forward_lstm_5/lstm_cell_16/kernel:0"])
        fwd2_rk = np.array(mw["bidirectional_5/bidirectional_5/forward_lstm_5/lstm_cell_16/recurrent_kernel:0"])
        fwd2_b = np.array(mw["bidirectional_5/bidirectional_5/forward_lstm_5/lstm_cell_16/bias:0"])
        bwd2_k = np.array(mw["bidirectional_5/bidirectional_5/backward_lstm_5/lstm_cell_17/kernel:0"])
        bwd2_rk = np.array(mw["bidirectional_5/bidirectional_5/backward_lstm_5/lstm_cell_17/recurrent_kernel:0"])
        bwd2_b = np.array(mw["bidirectional_5/bidirectional_5/backward_lstm_5/lstm_cell_17/bias:0"])

        # Dense
        dense_k = np.array(mw["dense_1/dense_1/kernel:0"])
        dense_b = np.array(mw["dense_1/dense_1/bias:0"])

    model.layers[0].set_weights([emb])
    model.layers[1].set_weights([fwd1_k, fwd1_rk, fwd1_b, bwd1_k, bwd1_rk, bwd1_b])
    model.layers[2].set_weights([fwd2_k, fwd2_rk, fwd2_b, bwd2_k, bwd2_rk, bwd2_b])
    model.layers[3].set_weights([dense_k, dense_b])


def encode_name(name, maxlen=10):
    chars = list("Eabcdefghijklmnopqrstuvwxyz U")
    char_map = {c: i for i, c in enumerate(chars)}
    ids = []
    for c in name.lower()[:maxlen]:
        ids.append(char_map.get(c, 28))
    ids += [0] * (maxlen - len(ids))
    return ids


def convert(h5_path, onnx_path, seq_len):
    import tensorflow as tf

    model = build_model(seq_len)
    load_weights_from_h5(model, h5_path)

    # Sanity check
    test = np.array([encode_name("turing", seq_len)], dtype=np.float32)
    pred = model.predict(test, verbose=0)[0]
    labels = ["asian", "black", "hispanic", "white"]
    print(f"  Sanity check 'turing': {dict(zip(labels, [f'{p:.3f}' for p in pred]))}")

    # Export via saved_model for tf2onnx compatibility with Keras 3
    saved_dir = f"/tmp/pyethnicity_{os.path.basename(h5_path)}"
    model.export(saved_dir)

    print(f"Converting to ONNX...")
    ret = os.system(
        f"python3 -m tf2onnx.convert --saved-model {saved_dir} "
        f"--output {onnx_path} --opset 13 2>&1 | grep -E 'INFO|ERROR'"
    )
    if ret != 0:
        print("ERROR: tf2onnx conversion failed")
        sys.exit(1)

    # Verify ONNX output matches TF
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {input_name: test})[0][0]
    max_diff = np.max(np.abs(pred - onnx_pred))
    print(f"  ONNX verification - max diff: {max_diff:.8f}")
    print(f"  Saved {onnx_path} ({os.path.getsize(onnx_path) / 1024:.0f}KB)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_rethnicity_data_raw>")
        print(f"Example: {sys.argv[0]} ../rethnicity/data-raw")
        sys.exit(1)

    data_raw = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(__file__), "..", "pyethnicity", "models")
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ("lastname_distill", 10),
        ("fullname_aligned_distill", 20),
    ]

    for name, seq_len in models:
        h5_path = os.path.join(data_raw, f"{name}.h5")
        onnx_path = os.path.join(output_dir, f"{name}.onnx")

        if not os.path.isfile(h5_path):
            print(f"ERROR: {h5_path} not found")
            sys.exit(1)

        convert(h5_path, onnx_path, seq_len)

    print("Done. ONNX models saved to pyethnicity/models/")


if __name__ == "__main__":
    main()
