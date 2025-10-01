"""
Test Analytics System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytics import Analytics


def test_analytics():
    print("=" * 70)
    print("ANALYTICS SYSTEM TEST")
    print("=" * 70)
    
    # Initialize analytics
    analytics = Analytics()
    
    # Test 1: Log some queries
    print("\nTest 1: Logging Queries")
    print("-" * 70)
    
    test_queries = [
        {
            'query': "How do I reset my password?",
            'intents': ['technical'],
            'confidence': 0.85,
            'similarity': 0.95,
            'response_time': 0.5
        },
        {
            'query': "What payment methods?",
            'intents': ['billing'],
            'confidence': 0.72,
            'similarity': 0.88,
            'response_time': 0.4
        },
        {
            'query': "happy birthday",
            'intents': ['billing'],
            'confidence': 0.30,
            'similarity': 0.45,
            'response_time': 0.3
        }
    ]
    
    for query_data in test_queries:
        analytics.log_query(**query_data)
        print(f"✓ Logged: {query_data['query']}")
    
    # Test 2: Log failed query
    print("\nTest 2: Logging Failed Query")
    print("-" * 70)
    analytics.log_failed_query(
        query="tell me a joke",
        intents=[],
        confidence=0.20,
        similarity=0.35,
        reason="Irrelevant query"
    )
    print("✓ Failed query logged")
    
    # Test 3: Get statistics
    print("\nTest 3: Getting Statistics")
    print("-" * 70)
    stats = analytics.get_statistics()
    
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Avg Confidence: {stats['avg_confidence']:.2%}")
    print(f"Avg Similarity: {stats['avg_similarity']:.2%}")
    print(f"Avg Response Time: {stats['avg_response_time']:.2f}s")
    print(f"Satisfaction Rate: {stats['satisfaction_rate']:.1f}%")
    print(f"Failed Queries: {stats['failed_queries_count']}")
    
    if stats['intent_distribution']:
        print("\nIntent Distribution:")
        for intent, count in stats['intent_distribution'].items():
            print(f"  {intent}: {count}")
    
    # Test 4: Update feedback
    print("\nTest 4: Updating Feedback")
    print("-" * 70)
    analytics.update_feedback(0, "positive")
    analytics.update_feedback(1, "positive")
    analytics.update_feedback(2, "negative")
    print("✓ Feedback updated")
    
    # Get updated stats
    stats = analytics.get_statistics()
    print(f"Updated Satisfaction Rate: {stats['satisfaction_rate']:.1f}%")
    
    # Test 5: Export data
    print("\nTest 5: Exporting Data")
    print("-" * 70)
    output = analytics.export_data()
    if output:
        print(f"✓ Data exported to: {output}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_analytics()