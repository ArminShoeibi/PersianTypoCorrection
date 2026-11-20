# PersianTypoCorrection
Persian typo correction using machine learning, character-level TF-IDF, and LinearSVC.



# Persian Typo Correction

A machine-learning project for detecting and correcting Persian (Farsi) spelling mistakes using character-level TF-IDF and LinearSVC.

## Overview

Persian Typo Correction is a supervised machine-learning project that learns to map incorrectly typed Persian words to their correct forms.

Unlike simple dictionary-based typo lookup, the model learns character-level patterns from training examples and can predict corrections for misspellings it has never encountered before.

For example:

| Incorrect Input | Correct Output |
|---|---|
| کشمیش | کشمش |
| کیشمش | کشمش |
| کیشمیش | کشمش |
| گشمش | کشمش |
| کیشمس | کشمش |

## How It Works

The project uses a scikit-learn Pipeline consisting of two main components:

1. **TfidfVectorizer** – Converts Persian text into numerical features using character n-grams.
2. **LinearSVC** – A Support Vector Machine classifier that predicts the correct spelling.

### Model Architecture

```text
Incorrect Persian Word
        |
        v
TfidfVectorizer
(Character n-grams: 2–5)
        |
        v
Numerical Feature Vectors
        |
        v
LinearSVC
        |
        v
Correct Persian Word
```

## Dataset

The training dataset is based on Persian typo examples stored in JSONL format.

Each record contains:

- `key`: The correct Persian word or phrase.
- `incorrectForms`: A collection of incorrect spelling variations.
- `published`: Determines whether the record is included in training.

Only published records are used.

Example:

```json
{
  "key": "کشمش",
  "incorrectForms": [
    "کشمیش",
    "کیشمش",
    "کیشمیش"
  ],
  "published": true
}
```

## Technology Stack

- Python
- Pandas
- scikit-learn
- TF-IDF
- LinearSVC
- ONNX (planned inference deployment)

## Model Configuration

The current model uses:

**TfidfVectorizer**

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    sublinear_tf=True
)
```

**LinearSVC**

```python
LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42
)
```

## Training

Install the dependencies:

```bash
pip install pandas scikit-learn
```

Load the JSONL dataset, prepare the training examples, and split the data into training and testing sets.

Train the model using:

```python
model.fit(
    train_df["incorrect"],
    train_df["correct"]
)
```

## Prediction

```python
def correct_word(word):
    return model.predict([word])[0]
```

Example:

```python
print(correct_word("گشمش"))

# کشمش
```

## Evaluation

The project uses a train/test split and classification metrics to evaluate model performance.

The goal is to measure how well the model generalizes to typo variations that were not included in the training examples.
