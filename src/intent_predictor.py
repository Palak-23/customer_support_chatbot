"""
Intent Predictor Module
Loads trained model and predicts intents for new queries
"""

import joblib
import numpy as np
from typing import List, Dict, Tuple
from entity_extractor import EntityExtractor


class IntentPredictor:
    """
    Predicts intents and extracts entities from customer queries
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.vectorizer = None
        self.classifier = None
        self.mlb = None
        self.entity_extractor = EntityExtractor()
        self.confidence_threshold = 0.30  # Lowered threshold for better predictions
        
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.vectorizer = joblib.load(f'{self.model_dir}/vectorizer.pkl')
            self.classifier = joblib.load(f'{self.model_dir}/intent_classifier.pkl')
            self.mlb = joblib.load(f'{self.model_dir}/mlb.pkl')
            print("✓ Models loaded successfully")
        except FileNotFoundError as e:
            print(f"Error: Model files not found. Please train the model first.")
            print(f"Run: python src/train_intent.py")
            raise e
    
    def predict(self, query: str) -> Dict:
        """
        Predict intent(s) and extract entities from query
        
        Args:
            query: User's input text
            
        Returns:
            Dictionary containing intents, confidence scores, and entities
        """
        # Vectorize query
        X = self.vectorizer.transform([query])
        
        # Get probability predictions
        y_pred_proba = self.classifier.predict_proba(X)[0]
        
        # Get confidence scores for all intents
        intent_scores = {
            intent: float(proba)
            for intent, proba in zip(self.mlb.classes_, y_pred_proba)
        }
        
        # Filter intents by confidence threshold
        high_confidence_intents = [
            intent for intent, score in intent_scores.items()
            if score >= self.confidence_threshold
        ]
        
        # If no high confidence intents, take the top one
        if not high_confidence_intents:
            top_intent = max(intent_scores.items(), key=lambda x: x[1])
            high_confidence_intents = [top_intent[0]]
        
        # Extract entities
        entities = self.entity_extractor.extract_all(query)
        
        # Calculate overall confidence (average of top intents)
        avg_confidence = np.mean([
            intent_scores[intent] for intent in high_confidence_intents
        ]) if high_confidence_intents else 0.0
        
        return {
            'query': query,
            'intents': high_confidence_intents,
            'all_intents': high_confidence_intents,  # Same for now
            'confidence_scores': intent_scores,
            'overall_confidence': float(avg_confidence),
            'entities': entities,
            'entity_summary': self.entity_extractor.get_entity_summary(query)
        }
    
    def predict_batch(self, queries: List[str]) -> List[Dict]:
        """
        Predict intents for multiple queries
        
        Args:
            queries: List of query strings
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(query) for query in queries]
    
    def get_top_intent(self, query: str) -> Tuple[str, float]:
        """
        Get the single most likely intent
        
        Args:
            query: User's input text
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        result = self.predict(query)
        top_intent = max(
            result['confidence_scores'].items(),
            key=lambda x: x[1]
        )
        return top_intent
    
    def is_ambiguous(self, query: str, threshold: float = 0.3) -> bool:
        """
        Check if query is ambiguous (multiple intents with similar confidence)
        
        Args:
            query: User's input text
            threshold: Maximum difference between top 2 intents to be considered ambiguous
            
        Returns:
            True if query is ambiguous
        """
        result = self.predict(query)
        scores = sorted(result['confidence_scores'].values(), reverse=True)
        
        if len(scores) < 2:
            return False
        
        # If top 2 scores are very close, it's ambiguous
        return (scores[0] - scores[1]) < threshold
    
    def format_prediction(self, result: Dict) -> str:
        """
        Format prediction result as readable string
        
        Args:
            result: Prediction dictionary from predict()
            
        Returns:
            Formatted string
        """
        lines = [
            f"Query: {result['query']}",
            f"Detected Intents: {', '.join(result['intents'])}",
            f"Overall Confidence: {result['overall_confidence']:.2%}",
            "",
            "Confidence Breakdown:"
        ]
        
        # Sort intents by confidence
        sorted_scores = sorted(
            result['confidence_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for intent, score in sorted_scores:
            bar = "█" * int(score * 20)
            lines.append(f"  {intent:12s} {score:.2%} {bar}")
        
        if result['entity_summary'] != "No entities found":
            lines.append("")
            lines.append(f"Entities: {result['entity_summary']}")
        
        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("INTENT PREDICTOR TEST")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = IntentPredictor(model_dir='models')
        
        # Test queries
        test_queries = [
            "I want to check my bill",
            "My product X is not working",
            "How do I change my email address?",
            "I'm very disappointed with your service",
            "I can't login to see my bill for account 1234567890",
            "Product Alpha-500 charged $99.99 but not delivered, order #XYZ789",
            "Cancel subscription and refund $49.99 to account 9876543210"
        ]
        
        for query in test_queries:
            print("\n" + "=" * 60)
            result = predictor.predict(query)
            print(predictor.format_prediction(result))
            
            # Check if ambiguous
            if predictor.is_ambiguous(query):
                print("\n⚠️  This query is AMBIGUOUS - may need clarification")
        
        print("\n" + "=" * 60)
        print("Test complete!")
        
    except FileNotFoundError:
        print("\n❌ Models not found!")
        print("Please run: python src/train_intent.py")