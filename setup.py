"""
Setup Script - Trains models if they don't exist
Run this before starting the app for the first time
"""

import os
import sys

def setup():
    print("=" * 70)
    print("SETTING UP CUSTOMER SUPPORT CHATBOT")
    print("=" * 70)
    
    # Check if models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ Created models directory")
    
    # Check if intent classifier exists
    if not os.path.exists('models/intent_classifier.pkl'):
        print("\n[1/2] Training intent classifier...")
        print("-" * 70)
        try:
            sys.path.insert(0, 'src')
            from train_intent import IntentClassifierTrainer
            trainer = IntentClassifierTrainer(data_path='data/intents.csv')
            trainer.run_full_training()
            print("✓ Intent classifier trained successfully")
        except Exception as e:
            print(f"✗ Error training intent classifier: {e}")
            return False
    else:
        print("✓ Intent classifier already exists")
    
    # Check if FAISS index exists
    if not os.path.exists('models/faiss_index/faiss.index'):
        print("\n[2/2] Building FAISS knowledge base...")
        print("-" * 70)
        try:
            sys.path.insert(0, 'src')
            from knowledge_base import KnowledgeBase
            kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
            kb.build_index(save_path='models/faiss_index')
            print("✓ FAISS index built successfully")
        except Exception as e:
            print(f"✗ Error building FAISS index: {e}")
            return False
    else:
        print("✓ FAISS index already exists")
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\nYou can now run the app:")
    print("  streamlit run app.py")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)