# 🎭 Sentiment Analysis - Emotion Classification

## 📋 Project Overview

This project performs sentiment analysis on text data to classify sentences into **6 emotion categories**:
- 😢 **Sadness**
- 😊 **Joy**
- ❤️ **Love**
- 😠 **Anger**
- 😨 **Fear**
- 😲 **Surprise**

## 🎯 Project Objective

Build and evaluate a Machine Learning model to determine which emotion category a user-written sentence belongs to.

## 📊 Dataset

- **Training:** 16,000 texts
- **Validation:** 2,000 texts
- **Test:** 2,000 texts

### Emotion Distribution (Training Set):
```
Sadness:  29.16% (4,666 texts)
Joy:      33.51% (5,362 texts)
Love:      8.15% (1,304 texts)
Anger:    13.49% (2,159 texts)
Fear:     12.11% (1,937 texts)
Surprise:  3.57% (572 texts)
```

## 🔬 Methods and Techniques

### 1. Text Preprocessing
- Lowercase transformation
- Number removal
- URL and email removal
- Punctuation removal
- Extra whitespace removal

### 2. Feature Extraction
- **TF-IDF Vectorization** (5,000 features)
- N-gram range: (1, 2) - unigrams and bigrams

### 3. Model Selection
The following models were tested and compared:

| Model | Validation Accuracy |
|-------|---------------------|
| Logistic Regression | 85.30% |
| Naive Bayes | 72.05% |
| **Linear SVM** | **89.30%** ✅ |

**Linear SVM** was selected as it achieved the best performance.

## 📈 Model Performance

### Test Set Results:
- **Overall Accuracy:** 88.35%

### Performance by Emotion:

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Sadness | 0.92 | 0.92 | 0.92 | 581 |
| Joy | 0.88 | 0.93 | 0.91 | 695 |
| Love | 0.77 | 0.74 | 0.75 | 159 |
| Anger | 0.90 | 0.86 | 0.88 | 275 |
| Fear | 0.88 | 0.84 | 0.86 | 224 |
| Surprise | 0.71 | 0.61 | 0.66 | 66 |

### Key Findings:

✅ **Strengths:**
- Sadness and Joy emotions are excellently recognized (92-93% recall)
- High overall accuracy (88.35%)
- Model performs in a balanced manner across most classes

⚠️ **Areas for Improvement:**
- Surprise emotion recognition is weaker (61% recall)
  - Reason: Limited samples in dataset (66 test samples)
- Love emotion is relatively harder to recognize
  - Reason: Overlaps with other emotions

## 🔍 Error Analysis

Total errors: 233 (11.65% of test set)

**Common error patterns:**
1. Fear and Anger are confused - both have negative tone
2. Confusion between Sadness and Joy - contextual ambiguity
3. Surprise often misclassified due to limited training data

## 🚀 Installation and Usage

### Install Required Packages:
```bash
pip install -r requirements.txt
```

### Run the Main Script:
```bash
python sentiment_analysis.py
```

### Launch Gradio Interface (optional):
```bash
python gradio_app.py
```

## 📁 File Structure

```
sentiment-analysis-project/
│
├── 📄 sentiment_analysis.py         # Main model and analysis code
├── 📄 gradio_app.py                 # Web interface (optional)
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Project documentation
├── 📄 ANALYSIS_REPORT.md           # Detailed results analysis
├── 📄 .gitignore                    # Git ignore file
│
├── 📊 Data Files (must be included)
│   ├── training.csv                 # Training dataset (16,000 rows)
│   ├── validation.csv               # Validation dataset (2,000 rows)
│   └── test.csv                     # Test dataset (2,000 rows)
│
└── 📈 Output Files (automatically generated)
    ├── eda_analysis.png             # EDA visualization
    └── confusion_matrix.png         # Confusion matrix plot
```

## 🛠️ Technologies

- **Python 3.8+**
- **pandas** - data manipulation
- **numpy** - numerical operations
- **scikit-learn** - machine learning models
- **matplotlib & seaborn** - visualization
- **gradio** - web interface (optional)

## 📊 Visualizations

The project generates the following plots:
1. **EDA Analysis** - Emotion distribution, text length analysis
2. **Confusion Matrix** - Visual representation of model errors

## 💡 Future Improvements

1. **Deep Learning Models:**
   - LSTM, GRU
   - Transformer models like BERT, RoBERTa

2. **Data Augmentation:**
   - Add more samples for Surprise
   - Back-translation technique

3. **Feature Engineering:**
   - Emoji analysis
   - Part-of-speech tagging
   - Sentiment lexicon features

4. **Model Ensemble:**
   - Combine multiple models
   - Voting classifier

## 📝 Test Examples

```python
test_sentences = {
    'sadness': "I feel so lonely and depressed today",
    'joy': "I am so happy and excited about this wonderful day",
    'love': "I love you so much, you make me feel complete",
    'anger': "I am so angry and frustrated with this situation",
    'fear': "I am scared and worried about what might happen",
    'surprise': "Wow, I cannot believe this is happening"
}
```

## 🎓 Key Learnings

1. **Text Preprocessing is crucial** - Proper cleaning improves text quality
2. **Class imbalance** problem - Some emotions are underrepresented
3. **Model Selection** - Linear SVM is optimal for this task
4. **Feature Engineering** - TF-IDF bigrams provide additional information
5. **Evaluation metrics** - Accuracy alone is not enough, precision/recall matter

## 🔧 Code Usage

### Basic Usage:
```python
from sentiment_analysis import SentimentAnalyzer

# Initialize and train model
analyzer = SentimentAnalyzer()
analyzer.load_data(
    train_path='training.csv',
    val_path='validation.csv',
    test_path='test.csv'
)
analyzer.prepare_data()
analyzer.train_models()

# Make prediction
text = "I am so excited about this opportunity!"
emotion, probabilities = analyzer.predict_emotion(text)

print(f"Predicted Emotion: {emotion}")
print(f"Confidence: {probabilities[emotion]:.2%}")
```

### Expected Output:
```
Predicted Emotion: joy
Confidence: 80.90%
```

## 📈 Model Pipeline

```
Text Input
    ↓
Preprocessing (lowercase, remove numbers, punctuation)
    ↓
TF-IDF Vectorization (5000 features, bigrams)
    ↓
Linear SVM Classifier
    ↓
Emotion Prediction + Probabilities
```

## 🎯 Performance Metrics Summary

```
Overall Test Accuracy:     88.35%
Training Time:             ~5 seconds
Inference Time:            <1ms per text
Model Size:                ~15MB
Feature Count:             5000

Best Performing Emotions:
─────────────────────────
1. Sadness:  92% F1-score ⭐⭐⭐⭐⭐
2. Joy:      91% F1-score ⭐⭐⭐⭐⭐
3. Anger:    88% F1-score ⭐⭐⭐⭐
4. Fear:     86% F1-score ⭐⭐⭐⭐
5. Love:     75% F1-score ⭐⭐⭐
6. Surprise: 66% F1-score ⭐⭐
```

## 🤝 Contributing

This project was developed for an AI internship program. Suggestions and improvements are welcome!

## 📄 License

This project is for educational purposes.

## 📧 Contact

For questions, please use the GitHub issues section.

---

**Project Status:** ✅ Complete  
**Code Quality:** Production-ready  
**Documentation:** Comprehensive  
**Test Coverage:** 88.35% accuracy on unseen data
