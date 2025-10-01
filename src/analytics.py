"""
Analytics Module
Tracks performance metrics and logs interactions
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, List


class Analytics:
    """
    Handles analytics tracking and logging
    """
    
    def __init__(self, log_dir='analytics'):
        self.log_dir = log_dir
        self.queries_log_file = f'{log_dir}/queries_log.csv'
        self.failed_queries_file = f'{log_dir}/failed_queries.csv'
        
        # Create directory if not exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files if they don't exist
        self._initialize_logs()
    
    def _initialize_logs(self):
        """Initialize log files with headers"""
        if not os.path.exists(self.queries_log_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'query', 'intents', 'confidence', 
                'similarity', 'response_time', 'feedback'
            ])
            df.to_csv(self.queries_log_file, index=False)
        
        if not os.path.exists(self.failed_queries_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'query', 'intents', 'confidence', 
                'similarity', 'reason'
            ])
            df.to_csv(self.failed_queries_file, index=False)
    
    def log_query(self, query: str, intents: List[str], confidence: float, 
                  similarity: float, response_time: float, feedback: str = None):
        """Log a query to CSV"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intents': '|'.join(intents) if intents else '',
            'confidence': confidence,
            'similarity': similarity,
            'response_time': response_time,
            'feedback': feedback or ''
        }
        
        df = pd.DataFrame([data])
        df.to_csv(self.queries_log_file, mode='a', header=False, index=False)
    
    def log_failed_query(self, query: str, intents: List[str], 
                        confidence: float, similarity: float, reason: str):
        """Log a failed query for improvement"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intents': '|'.join(intents) if intents else '',
            'confidence': confidence,
            'similarity': similarity,
            'reason': reason
        }
        
        df = pd.DataFrame([data])
        df.to_csv(self.failed_queries_file, mode='a', header=False, index=False)
    
    def update_feedback(self, query_index: int, feedback: str):
        """Update feedback for a specific query"""
        try:
            df = pd.read_csv(self.queries_log_file)
            if query_index < len(df):
                df.loc[query_index, 'feedback'] = feedback
                df.to_csv(self.queries_log_file, index=False)
        except Exception as e:
            print(f"Error updating feedback: {e}")
    
    def get_statistics(self) -> Dict:
        """Get overall statistics"""
        try:
            df = pd.read_csv(self.queries_log_file)
            
            if len(df) == 0:
                return self._empty_stats()
            
            # Calculate statistics
            stats = {
                'total_queries': len(df),
                'avg_confidence': df['confidence'].mean(),
                'avg_similarity': df['similarity'].mean(),
                'avg_response_time': df['response_time'].mean(),
                'satisfaction_rate': self._calculate_satisfaction_rate(df),
                'intent_distribution': self._get_intent_distribution(df),
                'failed_queries_count': self._count_failed_queries(df)
            }
            
            return stats
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return self._empty_stats()
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics"""
        return {
            'total_queries': 0,
            'avg_confidence': 0.0,
            'avg_similarity': 0.0,
            'avg_response_time': 0.0,
            'satisfaction_rate': 0.0,
            'intent_distribution': {},
            'failed_queries_count': 0
        }
    
    def _calculate_satisfaction_rate(self, df: pd.DataFrame) -> float:
        """Calculate satisfaction rate from feedback"""
        feedback_df = df[df['feedback'].notna() & (df['feedback'] != '')]
        if len(feedback_df) == 0:
            return 0.0
        
        positive = len(feedback_df[feedback_df['feedback'] == 'positive'])
        return (positive / len(feedback_df)) * 100
    
    def _get_intent_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution of intents"""
        all_intents = []
        for intents_str in df['intents']:
            if pd.notna(intents_str) and intents_str:
                all_intents.extend(intents_str.split('|'))
        
        if not all_intents:
            return {}
        
        intent_series = pd.Series(all_intents)
        return intent_series.value_counts().to_dict()
    
    def _count_failed_queries(self, df: pd.DataFrame) -> int:
        """Count queries with low confidence or similarity"""
        failed = df[(df['confidence'] < 0.35) | (df['similarity'] < 0.60)]
        return len(failed)
    
    def get_failed_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent failed queries"""
        try:
            if os.path.exists(self.failed_queries_file):
                df = pd.read_csv(self.failed_queries_file)
                df = df.tail(limit)
                return df.to_dict('records')
            return []
        except Exception as e:
            print(f"Error getting failed queries: {e}")
            return []


# Example usage
if __name__ == "__main__":
    analytics = Analytics()
    
    # Test logging
    analytics.log_query(
        query="How do I reset my password?",
        intents=['technical'],
        confidence=0.85,
        similarity=0.95,
        response_time=0.5
    )
    
    # Get statistics
    stats = analytics.get_statistics()
    print("Analytics Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")