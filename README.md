# pyethnicity

A Python port of [rethnicity](https://github.com/fColumn/rethnicity) - predict ethnicity from names using BiLSTM neural networks.

Uses lightweight ONNX Runtime for inference (~20MB).

## Usage

```python
from pyethnicity import predict_ethnicity

# Full name prediction (recommended)
results = predict_ethnicity(firstnames="Alan", lastnames="Turing")
print(results)
# [{'firstname': 'Alan', 'lastname': 'Turing', 'prob_asian': 0.05, 'prob_black': 0.01,
#   'prob_hispanic': 0.02, 'prob_white': 0.92, 'race': 'white'}]

# Batch prediction
results = predict_ethnicity(
    firstnames=["Alan", "Yuki", "Carlos"],
    lastnames=["Turing", "Tanaka", "Garcia"],
)

# Last name only
results = predict_ethnicity(lastnames=["Garcia", "Chen", "Smith"], method="lastname")
```

## Model

The model uses character-level BiLSTM architecture:
- **Input**: Character sequences (a-z, space, pad, unknown) - 10 chars per name
- **Architecture**: Embedding(29, 32) -> BiLSTM(64) -> BiLSTM(64) -> Dense(4, softmax)
- **Output**: Probabilities for 4 categories: asian, black, hispanic, white
- **Training data**: 837,056 balanced name records from rethnicity

Two models are included:
- `lastname_distill.onnx` - last name only (10 char input)
- `fullname_aligned_distill.onnx` - first + last name (20 char input)

## Credits

- Original R package: [rethnicity](https://github.com/fangzhou-xie/rethnicity) by [Fangzhou Xie](https://github.com/fangzhou-xie) (NYU/Rutgers)
- Model architecture and training: rethnicity project
- Python port and ONNX conversion: pyethnicity