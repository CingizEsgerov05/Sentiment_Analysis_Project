import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Emosiya etiketlərini təyin edirik
EMOTION_LABELS = {
    0: 'sadness',
    1: 'joy', 
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

class SentimentAnalyzer:
    """
    Sentiment Analysis model
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        
    def load_data(self, train_path, val_path, test_path):
        print("📊 Loading data...")
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f" - Training: {len(self.train_df)} rows")
        print(f" - Validation: {len(self.val_df)} rows")
        print(f" - Test: {len(self.test_df)} rows")
        
        return self.train_df, self.val_df, self.test_df
    
    def exploratory_data_analysis(self):
        " EDA - İlkin data analizi "
        print("\n" + "="*60)
        print(" - EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*60)
        
        # 1. Dataset haqqında ümumi məlumat
        print("\n1) Dataset Structure:")
        print(f"   - Columns: {list(self.train_df.columns)}")
        print(f"   - Missing values: {self.train_df.isnull().sum().sum()}")
        
        # 2. Label paylanması
        print("\n2) Emotion Distribution:")
        label_counts = self.train_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            emotion = EMOTION_LABELS[label]
            percentage = (count / len(self.train_df)) * 100
            print(f"   {label} - {emotion:10s} : {count:5d} ({percentage:.2f}%)")
        
        # 3. Mətn uzunluğu statistikası
        print("\n3) Text Length Statistics:")
        self.train_df['text_length'] = self.train_df['text'].apply(lambda x: len(str(x).split()))
        print(f"   - Orta: {self.train_df['text_length'].mean():.2f} words")
        print(f"   - Min: {self.train_df['text_length'].min()} words")
        print(f"   - Max: {self.train_df['text_length'].max()} words")
        
        # 4. Nümunə mətnlər
        print("\n4) Examples of each emotion:")
        for label in sorted(self.train_df['label'].unique()):
            sample = self.train_df[self.train_df['label'] == label].iloc[0]['text']
            print(f"   {EMOTION_LABELS[label]:10s}: {sample[:80]}...")
        
        # Visualizasiya
        self._plot_eda()
        
    def _plot_eda(self):
        " EDA graphs"
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Emosiya paylanması
        label_counts = self.train_df['label'].map(EMOTION_LABELS).value_counts()
        axes[0, 0].bar(label_counts.index, label_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Emotion Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Emotion')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Mətn uzunluğu histoqramı
        axes[0, 1].hist(self.train_df['text_length'], bins=50, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Emosiyalara görə mətn uzunluğu
        emotion_data = [self.train_df[self.train_df['label'] == i]['text_length'].values 
                       for i in sorted(self.train_df['label'].unique())]
        axes[1, 0].boxplot(emotion_data, labels=[EMOTION_LABELS[i] for i in sorted(self.train_df['label'].unique())])
        axes[1, 0].set_title('Text Length by Emotion', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Emotion')
        axes[1, 0].set_ylabel('Word Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Dataset paylanması
        datasets = ['Training', 'Validation', 'Test']
        sizes = [len(self.train_df), len(self.val_df), len(self.test_df)]
        axes[1, 1].pie(sizes, labels=datasets, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1, 1].set_title('Dataset Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        print("\n💾 EDA graphs saved: eda_analysis.png")
        plt.close()
    
    def preprocess_text(self, text):
        "Text preprocessing function"
        # Lowercase
        text = str(text).lower()
        
        # Rəqəmləri silmək
        text = re.sub(r'\d+', '', text)
        
        # URL-ləri silmək
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Email-ləri silmək
        text = re.sub(r'\S+@\S+', '', text)
        
        # Durğu işarələrini silmək
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Artıq boşluqları silmək
        text = ' '.join(text.split())
        
        return text
    
    def prepare_data(self):
        " Preparing data for modeling "
        print("\n🔄 Text processing begins...")
        
        self.train_df['cleaned_text'] = self.train_df['text'].apply(self.preprocess_text)
        self.val_df['cleaned_text'] = self.val_df['text'].apply(self.preprocess_text)
        self.test_df['cleaned_text'] = self.test_df['text'].apply(self.preprocess_text)
        
        print("✅ Text processing completed.")
        
        print("\n📝 Text processing example:")
        idx = 0
        print(f"   Original: {self.train_df['text'].iloc[idx]}")
        print(f"   After processing: {self.train_df['cleaned_text'].iloc[idx]}")
        
        # TF-IDF vektorlaşdırması
        print("\n🔢 TF-IDF vectorization...")
        self.X_train = self.vectorizer.fit_transform(self.train_df['cleaned_text'])
        self.X_val = self.vectorizer.transform(self.val_df['cleaned_text'])
        self.X_test = self.vectorizer.transform(self.test_df['cleaned_text'])
        
        self.y_train = self.train_df['label']
        self.y_val = self.val_df['label']
        self.y_test = self.test_df['label']
        
        print(f"✅ Feature count: {self.X_train.shape[1]}")
        
    def train_models(self):
        print("\n" + "="*60)
        print("🤖 MODEL TRAINING")
        print("="*60)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(max_iter=2000, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔹 {name} training...")
            model.fit(self.X_train, self.y_train)
            
            # Validation accuracy
            val_pred = model.predict(self.X_val)
            val_acc = accuracy_score(self.y_val, val_pred)
            
            results[name] = {
                'model': model,
                'val_accuracy': val_acc,
                'predictions': val_pred
            }
            
            print(f"   ✅ Validation Accuracy: {val_acc:.4f}")
        
        # Ən yaxşı modeli seçirik
        best_model_name = max(results, key=lambda x: results[x]['val_accuracy'])
        self.model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Accuracy: {results[best_model_name]['val_accuracy']:.4f}")
        
        return results
    
    def evaluate_model(self):
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n🎯 Test Accuracy: {accuracy:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]))
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_pred)
        
        # Səhv təhlili
        self._analyze_errors(y_pred)
        
    def _plot_confusion_matrix(self, y_pred):
        """
        Confusion matrix 
        """
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())],
                   yticklabels=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n💾 Confusion matrix saved: confusion_matrix.png")
        plt.close()
        
    def _analyze_errors(self, y_pred):
        print("\n🔍 Error Analysis:")
        
        # Səhv proqnozları tapırıq
        errors = self.test_df[self.y_test != y_pred].copy()
        errors['predicted'] = y_pred[self.y_test != y_pred]
        
        print(f"   Total error count: {len(errors)}")
        
        if len(errors) > 0:
            print("\n   Error sample cases:")
            for i in range(min(5, len(errors))):
                row = errors.iloc[i]
                true_label = EMOTION_LABELS[row['label']]
                pred_label = EMOTION_LABELS[row['predicted']]
                print(f"\n   {i+1}. Text: {row['text'][:100]}...")
                print(f"      True: {true_label} | Predicted: {pred_label}")
    
    def predict_emotion(self, text):
        """
        Yeni mətn üçün emosiya proqnozu verir
        """
        cleaned = self.preprocess_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        
        # Ehtimal paylanması (əgər model dəstəkləyirsə)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(vectorized)[0]
            prob_dict = {EMOTION_LABELS[i]: prob for i, prob in enumerate(probabilities)}
        else:
            # Decision function istifadə edək (SVM üçün)
            decision = self.model.decision_function(vectorized)[0]
            # Softmax tətbiq edək
            exp_decision = np.exp(decision - np.max(decision))
            probabilities = exp_decision / exp_decision.sum()
            prob_dict = {EMOTION_LABELS[i]: prob for i, prob in enumerate(probabilities)}
        
        return EMOTION_LABELS[prediction], prob_dict


def main():
    print("="*60)
    print("🎭 SENTIMENT ANALYSIS PROJECT")
    print("="*60)
    
    # Model yaradırıq
    analyzer = SentimentAnalyzer()
    
    # 1. Data yükləmə
    analyzer.load_data(
        train_path='dataset/training.csv',
        val_path='dataset/validation.csv',
        test_path='dataset/test.csv'
    )
    
    # 2. EDA
    analyzer.exploratory_data_analysis()
    
    # 3. Data hazırlığı
    analyzer.prepare_data()
    
    # 4. Model təlimi
    results = analyzer.train_models()
    
    # 5. Qiymətləndirmə
    analyzer.evaluate_model()
    
    # 6. Test nümunələri
    print("\n" + "="*60)
    print("🧪 MANUAL TEST EXAMPLES")
    print("="*60)
    
    test_sentences = {
        'sadness': "I feel so lonely and depressed today",
        'joy': "I am so happy and excited about this wonderful day",
        'love': "I love you so much, you make me feel complete",
        'anger': "I am so angry and frustrated with this situation",
        'fear': "I am scared and worried about what might happen",
        'surprise': "Wow, I cannot believe this is happening, so unexpected"
    }
    
    for emotion, sentence in test_sentences.items():
        predicted, probs = analyzer.predict_emotion(sentence)
        print(f"\n📝 Text: {sentence}")
        print(f"   🎯 Expected: {emotion}")
        print(f"   🤖 Predicted: {predicted}")
        print(f"   📊 Probabilities:")
        for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"      {emo:10s}: {prob:.4f}")
    
    print("\n" + "="*60)
    print("✅ PROJECT COMPLETED!")
    print("="*60)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()