"""
Test RAG System
Comprehensive testing for knowledge base retrieval
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knowledge_base import KnowledgeBase


def test_basic_search():
    """Test basic semantic search"""
    print("\n" + "=" * 70)
    print("TEST 1: BASIC SEMANTIC SEARCH")
    print("=" * 70)
    
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    test_cases = [
        {
            'query': "How do I reset my password?",
            'expected_keyword': 'password',
            'min_similarity': 0.7
        },
        {
            'query': "What payment options are available?",
            'expected_keyword': 'payment',
            'min_similarity': 0.6
        },
        {
            'query': "My app is not working",
            'expected_keyword': 'working',
            'min_similarity': 0.5
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['query']}")
        print("-" * 70)
        
        results = kb.search(test['query'], top_k=1)
        
        if results:
            result = results[0]
            print(f"Found: {result['question']}")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Answer: {result['answer'][:100]}...")
            
            # Check similarity threshold
            if result['similarity'] >= test['min_similarity']:
                print(f"✓ PASSED - Good similarity: {result['similarity']:.3f}")
                passed += 1
            else:
                print(f"✗ FAILED - Low similarity: {result['similarity']:.3f}")
                failed += 1
        else:
            print("✗ FAILED - No results found")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Basic Search Results: {passed} passed, {failed} failed")
    return passed, failed


def test_intent_filtering():
    """Test search with intent filtering"""
    print("\n" + "=" * 70)
    print("TEST 2: INTENT-BASED FILTERING")
    print("=" * 70)
    
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    test_cases = [
        {
            'query': "I need help with billing",
            'intent': 'billing',
            'expected_intent': 'billing'
        },
        {
            'query': "Technical issue with login",
            'intent': 'technical',
            'expected_intent': 'technical'
        },
        {
            'query': "Change my profile settings",
            'intent': 'account',
            'expected_intent': 'account'
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['query']}")
        print(f"Intent Filter: {test['intent']}")
        print("-" * 70)
        
        results = kb.search(test['query'], top_k=1, intent_filter=test['intent'])
        
        if results:
            result = results[0]
            print(f"Found: {result['question']}")
            print(f"Intent: {result['intent']}")
            print(f"Similarity: {result['similarity']:.3f}")
            
            if result['intent'] == test['expected_intent']:
                print(f"✓ PASSED - Correct intent: {result['intent']}")
                passed += 1
            else:
                print(f"✗ FAILED - Wrong intent: {result['intent']} (expected {test['expected_intent']})")
                failed += 1
        else:
            print("✗ FAILED - No results found")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Intent Filtering Results: {passed} passed, {failed} failed")
    return passed, failed


def test_contextual_answers():
    """Test contextual answer generation"""
    print("\n" + "=" * 70)
    print("TEST 3: CONTEXTUAL ANSWER GENERATION")
    print("=" * 70)
    
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    test_cases = [
        {
            'query': "How do I check my bill for account 1234567890?",
            'intents': ['billing'],
            'entities': {'account_numbers': ['1234567890'], 'product_names': [], 'order_numbers': [], 'dates': [], 'amounts': []}
        },
        {
            'query': "Product X is not working",
            'intents': ['technical'],
            'entities': {'account_numbers': [], 'product_names': ['X'], 'order_numbers': [], 'dates': [], 'amounts': []}
        },
        {
            'query': "Cancel my subscription",
            'intents': ['billing'],
            'entities': {'account_numbers': [], 'product_names': [], 'order_numbers': [], 'dates': [], 'amounts': []}
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['query']}")
        print(f"Intents: {test['intents']}")
        print(f"Entities: {[k for k, v in test['entities'].items() if v]}")
        print("-" * 70)
        
        result = kb.get_contextual_answer(
            test['query'], 
            test['intents'], 
            test['entities']
        )
        
        if result['found']:
            print(f"✓ Answer Found")
            print(f"  Similarity: {result['similarity']:.3f}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Answer Preview: {result['answer'][:150]}...")
            passed += 1
        else:
            print(f"✗ No suitable answer found")
            print(f"  Fallback: {result['answer'][:100]}...")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Contextual Answers Results: {passed} passed, {failed} failed")
    return passed, failed


def test_similarity_threshold():
    """Test similarity threshold behavior"""
    print("\n" + "=" * 70)
    print("TEST 4: SIMILARITY THRESHOLD")
    print("=" * 70)
    
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    # Test with queries that should have varying similarity
    test_cases = [
        {
            'query': "How do I reset my password?",  # Should match well
            'should_find': True
        },
        {
            'query': "Tell me about quantum physics",  # Irrelevant query
            'should_find': False
        },
        {
            'query': "What are the payment methods?",  # Should match well
            'should_find': True
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['query']}")
        print(f"Expected to find answer: {test['should_find']}")
        print("-" * 70)
        
        result = kb.get_best_answer(test['query'], threshold=0.5)
        
        if result['found'] == test['should_find']:
            print(f"✓ PASSED - Correct behavior")
            print(f"  Found: {result['found']}")
            print(f"  Similarity: {result['similarity']:.3f}")
            passed += 1
        else:
            print(f"✗ FAILED - Unexpected behavior")
            print(f"  Expected found={test['should_find']}, got found={result['found']}")
            failed += 1
        
        if result['found']:
            print(f"  Answer: {result['answer'][:100]}...")
    
    print("\n" + "=" * 70)
    print(f"Threshold Test Results: {passed} passed, {failed} failed")
    return passed, failed


def test_multi_result_ranking():
    """Test if results are properly ranked by similarity"""
    print("\n" + "=" * 70)
    print("TEST 5: RESULT RANKING")
    print("=" * 70)
    
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    query = "How do I update my payment information?"
    print(f"\nQuery: {query}")
    print("-" * 70)
    
    results = kb.search(query, top_k=5)
    
    print(f"Found {len(results)} results:\n")
    
    passed = True
    prev_similarity = 1.0
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result['similarity']:.3f} | {result['question']}")
        
        # Check if results are in descending order
        if result['similarity'] > prev_similarity:
            print(f"   ✗ FAILED - Results not properly ranked!")
            passed = False
        
        prev_similarity = result['similarity']
    
    if passed:
        print("\n✓ PASSED - Results properly ranked by similarity")
        return 1, 0
    else:
        print("\n✗ FAILED - Results not in correct order")
        return 0, 1


def test_knowledge_base_stats():
    """Test knowledge base statistics"""
    print("\n" + "=" * 70)
    print("TEST 6: KNOWLEDGE BASE STATISTICS")
    print("=" * 70)
    
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    stats = kb.get_statistics()
    
    print("\nKnowledge Base Stats:")
    print(f"  Total FAQs: {stats['total_faqs']}")
    print(f"  Index Size: {stats['index_size']}")
    print(f"  Embedding Dimension: {stats['embedding_dimension']}")
    
    print("\nCategories:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count}")
    
    print("\nIntents:")
    for intent, count in stats['intents'].items():
        print(f"  {intent}: {count}")
    
    # Validation
    if stats['total_faqs'] > 0 and stats['index_size'] > 0:
        print("\n✓ PASSED - Knowledge base properly loaded")
        return 1, 0
    else:
        print("\n✗ FAILED - Knowledge base not properly loaded")
        return 0, 1


def run_all_tests():
    """Run all RAG system tests"""
    print("\n" + "🚀" * 35)
    print(" " * 20 + "PHASE 3 RAG TESTING SUITE")
    print("🚀" * 35)
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Basic Search
    passed, failed = test_basic_search()
    total_passed += passed
    total_failed += failed
    
    # Test 2: Intent Filtering
    passed, failed = test_intent_filtering()
    total_passed += passed
    total_failed += failed
    
    # Test 3: Contextual Answers
    passed, failed = test_contextual_answers()
    total_passed += passed
    total_failed += failed
    
    # Test 4: Similarity Threshold
    passed, failed = test_similarity_threshold()
    total_passed += passed
    total_failed += failed
    
    # Test 5: Result Ranking
    passed, failed = test_multi_result_ranking()
    total_passed += passed
    total_failed += failed
    
    # Test 6: Statistics
    passed, failed = test_knowledge_base_stats()
    total_passed += passed
    total_failed += failed
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"✓ Total Passed: {total_passed}")
    print(f"✗ Total Failed: {total_failed}")
    
    if total_failed == 0:
        print(f"Success Rate: 100%")
        print("\n🎉 ALL TESTS PASSED! Phase 3 is complete!")
        print("✅ Ready to move to Phase 4: Streamlit Frontend")
    else:
        print(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()