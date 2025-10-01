"""
Test Conversation Management
Tests multi-turn conversations, context tracking, and clarifications
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation_manager import ConversationManager


def test_follow_up_detection():
    """Test detection of follow-up questions"""
    print("\n" + "=" * 70)
    print("TEST 1: FOLLOW-UP QUESTION DETECTION")
    print("=" * 70)
    
    manager = ConversationManager()
    
    test_cases = [
        {
            'query': "How do I reset my password?",
            'expected': False,
            'description': "Initial question"
        },
        {
            'query': "What if I don't receive the email?",
            'expected': True,
            'description': "Follow-up with 'what if'"
        },
        {
            'query': "How long does that take?",
            'expected': True,
            'description': "Follow-up with 'that'"
        },
        {
            'query': "What about mobile app?",
            'expected': True,
            'description': "Follow-up with 'what about'"
        },
        {
            'query': "How do I cancel my subscription?",
            'expected': False,
            'description': "New independent question"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['query']}")
        print(f"Description: {test['description']}")
        print("-" * 70)
        
        is_followup = manager.is_follow_up_question(test['query'])
        
        if is_followup == test['expected']:
            print(f"✓ PASSED - Correctly detected as {'follow-up' if is_followup else 'new question'}")
            passed += 1
        else:
            print(f"✗ FAILED - Expected {test['expected']}, got {is_followup}")
            failed += 1
        
        # Simulate adding to history
        if i < len(test_cases):
            manager.add_message('user', test['query'], {'intents': ['technical']})
            manager.add_message('assistant', "Sample response", {})
    
    print("\n" + "=" * 70)
    print(f"Follow-up Detection Results: {passed} passed, {failed} failed")
    return passed, failed


def test_clarification_needed():
    """Test clarification question generation"""
    print("\n" + "=" * 70)
    print("TEST 2: CLARIFICATION QUESTIONS")
    print("=" * 70)
    
    manager = ConversationManager()
    
    test_cases = [
        {
            'intents': ['billing'],
            'confidence': 0.85,
            'should_clarify': False,
            'description': "High confidence, single intent"
        },
        {
            'intents': ['billing', 'technical'],
            'confidence': 0.20,
            'should_clarify': True,
            'description': "Low confidence"
        },
        {
            'intents': ['billing', 'technical', 'account'],
            'confidence': 0.45,
            'should_clarify': True,
            'description': "Multiple intents"
        },
        {
            'intents': ['technical', 'billing'],
            'confidence': 0.32,
            'should_clarify': True,
            'description': "Medium confidence, ambiguous"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}")
        print(f"Description: {test['description']}")
        print(f"Intents: {test['intents']}, Confidence: {test['confidence']:.2f}")
        print("-" * 70)
        
        clarification = manager.should_ask_clarification(
            test['intents'],
            test['confidence']
        )
        
        needs_clarification = clarification is not None
        
        if needs_clarification == test['should_clarify']:
            print(f"✓ PASSED - Correct behavior")
            if clarification:
                print(f"  Clarification: {clarification}")
            passed += 1
        else:
            print(f"✗ FAILED - Expected clarification={test['should_clarify']}, got {needs_clarification}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Clarification Results: {passed} passed, {failed} failed")
    return passed, failed


def test_context_tracking():
    """Test context tracking across conversation"""
    print("\n" + "=" * 70)
    print("TEST 3: CONTEXT TRACKING")
    print("=" * 70)
    
    manager = ConversationManager()
    
    # Simulate conversation
    print("\nSimulating conversation...")
    print("-" * 70)
    
    # Message 1
    manager.add_message('user', "Check my bill for account 123456", {
        'intents': ['billing'],
        'entities': {'account_numbers': ['123456'], 'product_names': []}
    })
    manager.add_message('assistant', "Response 1", {})
    
    context1 = manager.get_context()
    print(f"\nAfter message 1:")
    print(f"  Context: {context1}")
    
    # Message 2
    manager.add_message('user', "What about product X?", {
        'intents': ['billing'],
        'entities': {'account_numbers': [], 'product_names': ['X']}
    })
    manager.add_message('assistant', "Response 2", {})
    
    context2 = manager.get_context()
    print(f"\nAfter message 2:")
    print(f"  Context: {context2}")
    
    # Verify context
    passed = 0
    failed = 0
    
    if context1.get('account') == '123456':
        print("✓ PASSED - Account number tracked")
        passed += 1
    else:
        print("✗ FAILED - Account number not tracked")
        failed += 1
    
    if context2.get('product') == 'X':
        print("✓ PASSED - Product name tracked")
        passed += 1
    else:
        print("✗ FAILED - Product name not tracked")
        failed += 1
    
    if context2.get('last_intent') == 'billing':
        print("✓ PASSED - Last intent tracked")
        passed += 1
    else:
        print("✗ FAILED - Last intent not tracked")
        failed += 1
    
    print("\n" + "=" * 70)
    print(f"Context Tracking Results: {passed} passed, {failed} failed")
    return passed, failed


def test_query_enhancement():
    """Test query enhancement with context"""
    print("\n" + "=" * 70)
    print("TEST 4: QUERY ENHANCEMENT")
    print("=" * 70)
    
    manager = ConversationManager()
    
    # Set up context
    manager.add_message('user', "How do I reset my password?", {
        'intents': ['technical'],
        'entities': {'product_names': ['MobileApp']}
    })
    manager.add_message('assistant', "Response", {})
    
    test_queries = [
        {
            'query': "What if that doesn't work?",
            'should_enhance': True
        },
        {
            'query': "How long does it take?",
            'should_enhance': True
        },
        {
            'query': "How do I cancel my subscription?",
            'should_enhance': False
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_queries, 1):
        print(f"\nTest Case {i}: {test['query']}")
        print("-" * 70)
        
        enhanced = manager.enhance_query_with_context(test['query'])
        is_enhanced = enhanced != test['query']
        
        if is_enhanced == test['should_enhance']:
            print(f"✓ PASSED")
            if is_enhanced:
                print(f"  Enhanced to: {enhanced}")
            passed += 1
        else:
            print(f"✗ FAILED - Expected enhancement={test['should_enhance']}, got {is_enhanced}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Query Enhancement Results: {passed} passed, {failed} failed")
    return passed, failed


def test_fallback_responses():
    """Test fallback response generation"""
    print("\n" + "=" * 70)
    print("TEST 5: FALLBACK RESPONSES")
    print("=" * 70)
    
    manager = ConversationManager()
    
    test_cases = [
        {
            'intents': ['billing'],
            'similarity': 0.3,
            'description': "Low similarity with billing intent"
        },
        {
            'intents': ['technical'],
            'similarity': 0.25,
            'description': "Low similarity with technical intent"
        },
        {
            'intents': [],
            'similarity': 0.2,
            'description': "No intent detected"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['description']}")
        print(f"Intents: {test['intents']}, Similarity: {test['similarity']}")
        print("-" * 70)
        
        fallback = manager.get_fallback_response(
            test['intents'],
            test['similarity']
        )
        
        if fallback:
            print(f"✓ PASSED - Fallback generated")
            print(f"  Response: {fallback[:100]}...")
            passed += 1
        else:
            print(f"✗ FAILED - No fallback generated")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Fallback Response Results: {passed} passed, {failed} failed")
    return passed, failed


def run_all_tests():
    """Run all conversation management tests"""
    print("\n" + "🔄" * 35)
    print(" " * 15 + "CONVERSATION MANAGEMENT TESTS")
    print("🔄" * 35)
    
    total_passed = 0
    total_failed = 0
    
    # Test 1
    passed, failed = test_follow_up_detection()
    total_passed += passed
    total_failed += failed
    
    # Test 2
    passed, failed = test_clarification_needed()
    total_passed += passed
    total_failed += failed
    
    # Test 3
    passed, failed = test_context_tracking()
    total_passed += passed
    total_failed += failed
    
    # Test 4
    passed, failed = test_query_enhancement()
    total_passed += passed
    total_failed += failed
    
    # Test 5
    passed, failed = test_fallback_responses()
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
        print("\n🎉 ALL TESTS PASSED! Conversation Management Complete!")
        print("✅ Ready for final submission!")
    else:
        print(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()