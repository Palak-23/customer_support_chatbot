"""
Intent Classifier Training Module
Trains a multi-label classifier for customer support intents
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os


class IntentClassifierTrainer:
    """
    Trains and evaluates multi-label intent classifier
    """
    
    def __init__(self, data_path='data/intents.csv'):
        self.data_path = data_path
        self.vectorizer = None
        self.classifier = None
        self.mlb = None
        self.intent_labels = ['billing', 'technical', 'account', 'complaints']
        
    def load_data(self):
        """Load and preprocess training data"""
        print("Loading training data...")
        df = pd.read_csv(self.data_path)
        
        # Handle multi-label intents (separated by |)
        df['intent_list'] = df['intent'].apply(
            lambda x: [i.strip() for i in x.split('|')]
        )
        
        print(f"Loaded {len(df)} training examples")
        print(f"Intent distribution:")
        
        # Count each intent
        all_intents = []
        for intents in df['intent_list']:
            all_intents.extend(intents)
        
        intent_counts = pd.Series(all_intents).value_counts()
        print(intent_counts)
        
        return df
    
    def prepare_features(self, df):
        """Convert text to TF-IDF features"""
        print("\nPreparing features...")
        
        # Initialize MultiLabelBinarizer for multi-label encoding
        self.mlb = MultiLabelBinarizer(classes=self.intent_labels)
        y = self.mlb.fit_transform(df['intent_list'])
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english'
        )
        
        X = self.vectorizer.fit_transform(df['text'])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label matrix shape: {y.shape}")
        
        return X, y
    
    def train_model(self, X, y):
        """Train multi-label classifier"""
        print("\nTraining classifier...")
        
        # Check if we have any positive labels
        if y.sum() == 0:
            print("⚠️  WARNING: No labels found in training data!")
            return None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"Training set: {X_train.shape[0]} examples")
        print(f"Test set: {X_test.shape[0]} examples")
        
        # Train OneVsRest Logistic Regression
        self.classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.classifier.predict(X_test)
        
        # Calculate accuracy
        from sklearn.metrics import hamming_loss, jaccard_score
        accuracy = 1 - hamming_loss(y_test, y_pred)
        jaccard = jaccard_score(y_test, y_pred, average='samples', zero_division=0)
        
        print(f"\nHamming Accuracy: {accuracy:.3f}")
        print(f"Jaccard Score: {jaccard:.3f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.intent_labels,
            zero_division=0
        ))
        
        return X_train, X_test, y_train, y_test
    
    def save_model(self, model_dir='models'):
        """Save trained model and vectorizer"""
        print(f"\nSaving model to {model_dir}/...")
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save components
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer.pkl')
        joblib.dump(self.classifier, f'{model_dir}/intent_classifier.pkl')
        joblib.dump(self.mlb, f'{model_dir}/mlb.pkl')
        
        print("✓ Vectorizer saved")
        print("✓ Classifier saved")
        print("✓ Label binarizer saved")
        print("\nModel training complete!")
    
    def test_predictions(self, test_queries):
        """Test model on sample queries"""
        print("\n" + "=" * 60)
        print("TESTING PREDICTIONS")
        print("=" * 60)
        
        for query in test_queries:
            # Vectorize
            X = self.vectorizer.transform([query])
            
            # Get probabilities
            y_pred_proba = self.classifier.predict_proba(X)
            
            # Get confidence scores
            intent_scores = {
                intent: float(proba)
                for intent, proba in zip(self.intent_labels, y_pred_proba[0])
            }
            
            # Get top intents (threshold 0.3 for demo)
            predicted_intents = [
                intent for intent, score in intent_scores.items() 
                if score >= 0.30
            ]
            
            # If no intents above threshold, take top 1
            if not predicted_intents:
                top_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
                predicted_intents = [top_intent]
            
            print(f"\nQuery: {query}")
            print(f"Predicted Intents: {predicted_intents}")
            print(f"Confidence Scores:")
            for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(score * 30)
                print(f"  {intent:12s} {score:.3f} {bar}")
            print("-" * 60)
    
    def run_full_training(self):
        """Execute complete training pipeline"""
        print("=" * 60)
        print("INTENT CLASSIFIER TRAINING")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train model
        self.train_model(X, y)
        
        # Save model
        self.save_model()
        
        # Test on sample queries
        test_queries = [
            "I want to check my bill",
            "My app is crashing",
            "How do I change my email?",
            "I'm not satisfied with your service",
            "I can't login and need to pay my bill",
            "Product X not working and I want refund",
            "Update my account and check billing"
        ]
        
        self.test_predictions(test_queries)


# Main execution
if __name__ == "__main__":
    trainer = IntentClassifierTrainer(data_path='data/intents.csv')
    trainer.run_full_training()