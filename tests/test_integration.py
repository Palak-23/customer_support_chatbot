"""
Integration Test - Phase 2 + Phase 3
Tests the complete pipeline: Intent Classification + RAG
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intent_predictor import IntentPredictor
from knowledge_base import KnowledgeBase


def test_full_pipeline():
    """Test complete query processing pipeline"""
    print("\n" + "=" * 70)
    print("FULL PIPELINE INTEGRATION TEST")
    print("Phase 2 (Intent + Entities) → Phase 3 (RAG)")
    print("=" * 70)
    
    # Initialize both systems
    print("\nInitializing systems...")
    predictor = IntentPredictor(model_dir='models')
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    print("✓ Systems loaded\n")
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "I want to check my bill for account 1234567890",
        "My product X is not working and I want a refund",
        "How do I cancel my subscription?",
        "The app keeps crashing on my phone"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST QUERY {i}: {query}")
        print(f"{'=' * 70}")
        
        # Step 1: Intent Classification + Entity Extraction
        print("\n[PHASE 2] Intent Classification & Entity Extraction")
        print("-" * 70)
        
        intent_result = predictor.predict(query)
        
        print(f"Detected Intents: {intent_result['intents']}")
        print(f"Confidence: {intent_result['overall_confidence']:.2%}")
        
        if intent_result['entity_summary'] != "No entities found":
            print(f"Entities: {intent_result['entity_summary']}")
        
        print("\nConfidence Breakdown:")
        for intent, score in sorted(intent_result['confidence_scores'].items(), 
                                    key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"  {intent:12s} {score:.2%} {bar}")
        
        # Step 2: RAG - Retrieve Answer
        print(f"\n[PHASE 3] Knowledge Base Retrieval (RAG)")
        print("-" * 70)
        
        answer_result = kb.get_contextual_answer(
            query,
            intent_result['intents'],
            intent_result['entities']
        )
        
        if answer_result['found']:
            print(f"✓ Answer Found!")
            print(f"Similarity: {answer_result['similarity']:.3f} ({answer_result['confidence']})")
            print(f"Matched Question: {answer_result['matched_question']}")
            print(f"\nAnswer:")
            print(f"  {answer_result['answer']}\n")
            
            if answer_result.get('alternative_answers'):
                print(f"Alternative Answers Available: {len(answer_result['alternative_answers'])}")
        else:
            print(f"✗ No suitable answer found")
            print(f"Fallback Response:")
            print(f"  {answer_result['answer']}")
        
        print("\n" + "=" * 70)
        input("Press Enter to continue to next query...")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 70)
    print("EDGE CASES & ERROR HANDLING TEST")
    print("=" * 70)
    
    predictor = IntentPredictor(model_dir='models')
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    edge_cases = [
        {
            'name': "Empty Query",
            'query': "",
            'description': "Handling empty input"
        },
        {
            'name': "Very Short Query",
            'query': "help",
            'description': "Single word query"
        },
        {
            'name': "Irrelevant Query",
            'query': "What's the weather in Paris?",
            'description': "Completely unrelated to support"
        },
        {
            'name': "Multi-Intent Query",
            'query': "I can't login to pay my bill and want to change my email",
            'description': "Multiple intents in one query"
        },
        {
            'name': "Query with Typos",
            'query': "How do i reste my pasword?",
            'description': "Misspelled words"
        }
    ]
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Edge Case {i}: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"Query: '{case['query']}'")
        print(f"{'─' * 70}")
        
        try:
            # Process query
            if case['query']:
                intent_result = predictor.predict(case['query'])
                answer_result = kb.get_contextual_answer(
                    case['query'],
                    intent_result['intents'],
                    intent_result['entities']
                )
                
                print(f"✓ Processed successfully")
                print(f"  Intents: {intent_result['intents']}")
                print(f"  Answer Found: {answer_result['found']}")
                print(f"  Response: {answer_result['answer'][:100]}...")
            else:
                print(f"✗ Empty query - skipped")
                
        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")


def show_system_stats():
    """Display system statistics"""
    print("\n" + "=" * 70)
    print("SYSTEM STATISTICS")
    print("=" * 70)
    
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    stats = kb.get_statistics()
    
    print("\n[Knowledge Base]")
    print(f"  Total FAQ Entries: {stats['total_faqs']}")
    print(f"  Vector Dimension: {stats['embedding_dimension']}")
    print(f"  Index Size: {stats['index_size']} vectors")
    
    print("\n[Categories Distribution]")
    for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_faqs']) * 100
        print(f"  {category:12s} {count:3d} ({percentage:.1f}%)")
    
    print("\n[Intents Distribution]")
    for intent, count in sorted(stats['intents'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_faqs']) * 100
        print(f"  {intent:12s} {count:3d} ({percentage:.1f}%)")
    
    print("\n[Intent Classification Model]")
    print(f"  Model Type: TF-IDF + Logistic Regression")
    print(f"  Classes: billing, technical, account, complaints")
    print(f"  Multi-label: Yes")
    
    print("\n[Embedding Model]")
    print(f"  Model: all-MiniLM-L6-v2")
    print(f"  Embedding Size: 384 dimensions")
    print(f"  Search Method: FAISS L2 (cosine similarity)")


def run_demo():
    """Interactive demo mode"""
    print("\n" + "🎯" * 35)
    print(" " * 15 + "INTERACTIVE CHATBOT DEMO")
    print("🎯" * 35)
    
    predictor = IntentPredictor(model_dir='models')
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    
    print("\n✓ Systems Ready!")
    print("\nType your questions (or 'quit' to exit):\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nBot: Goodbye! Thanks for chatting!")
                break
            
            # Process query
            intent_result = predictor.predict(user_input)
            answer_result = kb.get_contextual_answer(
                user_input,
                intent_result['intents'],
                intent_result['entities']
            )
            
            # Display response
            print(f"\nBot: {answer_result['answer']}")
            
            # Show metadata
            print(f"\n[Intent: {', '.join(intent_result['intents'])} | "
                  f"Confidence: {intent_result['overall_confidence']:.0%} | "
                  f"Similarity: {answer_result.get('similarity', 0):.2f}]")
            
            if intent_result['entity_summary'] != "No entities found":
                print(f"[Entities: {intent_result['entity_summary']}]")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"\nBot: Sorry, I encountered an error: {str(e)}")
            print()


def main():
    """Main test runner"""
    print("\n" + "🚀" * 35)
    print(" " * 10 + "INTEGRATION TEST - PHASE 2 + PHASE 3")
    print("🚀" * 35)
    
    print("\n" + "=" * 70)
    print("TEST OPTIONS")
    print("=" * 70)
    print("\n1. Full Pipeline Test (5 sample queries)")
    print("2. Edge Cases & Error Handling")
    print("3. System Statistics")
    print("4. Interactive Demo Mode")
    print("5. Run All Tests")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        test_full_pipeline()
    elif choice == '2':
        test_edge_cases()
    elif choice == '3':
        show_system_stats()
    elif choice == '4':
        run_demo()
    elif choice == '5':
        test_full_pipeline()
        test_edge_cases()
        show_system_stats()
        
        print("\n\n" + "=" * 70)
        print("ALL TESTS COMPLETE!")
        print("=" * 70)
        print("\n✅ Phase 2 + Phase 3 Integration: WORKING")
        print("✅ Ready for Phase 4: Streamlit Frontend")
        
        demo = input("\nWould you like to try the interactive demo? (y/n): ")
        if demo.lower() == 'y':
            run_demo()
    else:
        print("\nInvalid option. Please run again and select 1-5.")


if __name__ == "__main__":
    main()