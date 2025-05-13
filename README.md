# Spam Detection using Machine Learning

This project builds a spam classifier using Natural Language Processing (NLP) and the Naive Bayes algorithm. The model classifies SMS messages as **spam** or **ham** (not spam).

## Features
- Text cleaning with NLTK
- TF-IDF vectorization
- Multinomial Naive Bayes classifier
- Accuracy ~98%
- Lightweight and fast

## Dataset
Dataset used: [SMS Spam Collection Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

Make sure to place `spam.csv` in the `dataset/` folder.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python spam_detector.py
```

## Project Output
- Prints model accuracy on test data
- Classifies new messages as spam or ham

## Future Scope
- Extend to multi-language spam detection
- Use deep learning (LSTM) for better context understanding
- Integrate with real-time messaging apps

## License
MIT License
