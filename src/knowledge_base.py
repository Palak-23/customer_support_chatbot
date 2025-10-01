"""
Knowledge Base RAG System
Uses FAISS for semantic search to retrieve relevant answers
"""

import pandas as pd
import numpy as np
import faiss
import joblib
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple


class KnowledgeBase:
    """
    RAG-based knowledge base for customer support
    Uses sentence embeddings + FAISS for fast semantic search
    """
    
    def __init__(self, faq_path='data/faq_knowledge_base.csv', 
                 model_name='all-MiniLM-L6-v2'):
        """
        Initialize knowledge base
        
        Args:
            faq_path: Path to FAQ CSV file
            model_name: Sentence transformer model name
        """
        self.faq_path = faq_path
        self.model_name = model_name
        self.faq_data = None
        self.index = None
        self.embeddings = None
        self.embedding_model = None
        
        # Load model and data
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"✓ Model '{model_name}' loaded")
        
        self.load_faq_data()
    
    def load_faq_data(self):
        """Load FAQ data from CSV"""
        print(f"\nLoading FAQ data from {self.faq_path}...")
        self.faq_data = pd.read_csv(self.faq_path)
        print(f"✓ Loaded {len(self.faq_data)} FAQ entries")
        
        # Display categories
        print("\nCategories:")
        for category, count in self.faq_data['category'].value_counts().items():
            print(f"  {category}: {count}")
    
    def build_index(self, save_path='models/faiss_index'):
        """
        Build FAISS index from FAQ questions
        
        Args:
            save_path: Directory to save index and embeddings
        """
        print("\nBuilding FAISS index...")
        
        # Generate embeddings for all questions
        questions = self.faq_data['question'].tolist()
        print(f"Encoding {len(questions)} questions...")
        
        self.embeddings = self.embedding_model.encode(
            questions,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"✓ Embeddings shape: {self.embeddings.shape}")
        
        # Build FAISS index (using L2 for normalized vectors = cosine similarity)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✓ FAISS index built with {self.index.ntotal} vectors")
        
        # Save index and embeddings
        os.makedirs(save_path, exist_ok=True)
        
        faiss.write_index(self.index, f'{save_path}/faiss.index')
        np.save(f'{save_path}/embeddings.npy', self.embeddings)
        self.faq_data.to_csv(f'{save_path}/faq_data.csv', index=False)
        
        print(f"✓ Index saved to {save_path}/")
    
    def load_index(self, index_path='models/faiss_index'):
        """
        Load pre-built FAISS index
        
        Args:
            index_path: Directory containing saved index
        """
        print(f"\nLoading FAISS index from {index_path}...")
        
        try:
            self.index = faiss.read_index(f'{index_path}/faiss.index')
            self.embeddings = np.load(f'{index_path}/embeddings.npy')
            self.faq_data = pd.read_csv(f'{index_path}/faq_data.csv')
            
            print(f"✓ Index loaded: {self.index.ntotal} vectors")
            print(f"✓ FAQ data loaded: {len(self.faq_data)} entries")
            
        except FileNotFoundError:
            print("❌ Index not found. Building new index...")
            self.build_index(save_path=index_path)
    
    def search(self, query: str, top_k: int = 3, 
               intent_filter: str = None) -> List[Dict]:
        """
        Search for relevant answers using semantic similarity
        
        Args:
            query: User's question
            top_k: Number of top results to return
            intent_filter: Filter by intent (billing, technical, etc.)
            
        Returns:
            List of dictionaries containing matched FAQs with scores
        """
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
        
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k * 2  # Get more results for filtering
        )
        
        # Convert distances to similarity scores (0 to 1)
        # Since vectors are normalized, L2 distance relates to cosine similarity
        similarities = 1 - (distances[0] / 4)  # Scale to 0-1 range
        
        # Prepare results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.faq_data):  # Safety check
                faq = self.faq_data.iloc[idx]
                
                # Apply intent filter if specified
                if intent_filter and faq['intent'] != intent_filter:
                    continue
                
                results.append({
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'category': faq['category'],
                    'intent': faq['intent'],
                    'similarity': float(similarity),
                    'confidence': 'high' if similarity > 0.7 else 'medium' if similarity > 0.5 else 'low'
                })
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_best_answer(self, query: str, intent: str = None, 
                       threshold: float = 0.5) -> Dict:
        """
        Get single best answer for a query
        
        Args:
            query: User's question
            intent: Detected intent (optional, helps filtering)
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with best answer or fallback
        """
        results = self.search(query, top_k=1, intent_filter=intent)
        
        if results and results[0]['similarity'] >= threshold:
            return {
                'found': True,
                'answer': results[0]['answer'],
                'question': results[0]['question'],
                'similarity': results[0]['similarity'],
                'confidence': results[0]['confidence'],
                'source': 'knowledge_base'
            }
        else:
            return {
                'found': False,
                'answer': "I'm sorry, I couldn't find a relevant answer to your question. Would you like to speak with a human agent?",
                'similarity': 0.0,
                'confidence': 'none',
                'source': 'fallback'
            }
    
    def get_contextual_answer(self, query: str, intents: List[str], 
                             entities: Dict) -> Dict:
        """
        Get answer with context from intent and entities
        
        Args:
            query: User's question
            intents: List of detected intents
            entities: Extracted entities
            
        Returns:
            Enhanced answer with context
        """
        # Use primary intent for filtering
        primary_intent = intents[0] if intents else None
        
        # Search with intent filter
        results = self.search(query, top_k=3, intent_filter=primary_intent)
        
        if not results:
            # Try without filter
            results = self.search(query, top_k=3)
        
        if results and results[0]['similarity'] >= 0.5:
            best_match = results[0]
            
            # Add context from entities
            answer = best_match['answer']
            
            # Personalize with entity information
            if entities.get('account_numbers'):
                account = entities['account_numbers'][0]
                answer += f"\n\nFor account {account}, you can access this in your account dashboard."
            
            if entities.get('product_names'):
                product = entities['product_names'][0]
                answer += f"\n\nFor {product}, please ensure you're using the latest version."
            
            return {
                'found': True,
                'answer': answer,
                'matched_question': best_match['question'],
                'similarity': best_match['similarity'],
                'confidence': best_match['confidence'],
                'intent_used': primary_intent,
                'alternative_answers': results[1:] if len(results) > 1 else []
            }
        else:
            return {
                'found': False,
                'answer': self._get_fallback_response(intents),
                'confidence': 'none',
                'intent_used': primary_intent
            }
    
    def _get_fallback_response(self, intents: List[str]) -> str:
        """Generate intent-specific fallback responses"""
        if not intents:
            return "I'm not sure I understand. Could you please rephrase your question?"
        
        intent = intents[0]
        
        fallbacks = {
            'billing': "I couldn't find specific billing information for your query. Please contact our billing department at billing@company.com or call 1-800-BILLING.",
            'technical': "I couldn't find a solution for this technical issue. Please contact our technical support team at support@company.com or visit our troubleshooting guide.",
            'account': "I couldn't find information about this account feature. Please check your Account Settings or contact support@company.com for assistance.",
            'complaints': "I'm sorry you're experiencing this issue. A customer service representative will contact you within 24 hours. You can also call 1-800-SUPPORT for immediate assistance."
        }
        
        return fallbacks.get(intent, 
            "I couldn't find a relevant answer. Would you like to speak with a human agent?")
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            'total_faqs': len(self.faq_data),
            'categories': self.faq_data['category'].value_counts().to_dict(),
            'intents': self.faq_data['intent'].value_counts().to_dict(),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }


# Main execution for building index
if __name__ == "__main__":
    print("=" * 70)
    print("KNOWLEDGE BASE SETUP")
    print("=" * 70)
    
    # Initialize knowledge base
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    
    # Build and save index
    kb.build_index(save_path='models/faiss_index')
    
    # Test searches
    print("\n" + "=" * 70)
    print("TESTING SEARCH")
    print("=" * 70)
    
    test_queries = [
        "How do I reset my password?",
        "What payment methods do you accept?",
        "My app keeps crashing",
        "I want to cancel my subscription",
        "How do I contact support?"
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Query: {query}")
        print(f"{'─' * 70}")
        
        results = kb.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Similarity: {result['similarity']:.3f} ({result['confidence']})")
            print(f"  Question: {result['question']}")
            print(f"  Answer: {result['answer'][:100]}...")
            print(f"  Category: {result['category']}")
    
    # Show statistics
    print("\n" + "=" * 70)
    print("KNOWLEDGE BASE STATISTICS")
    print("=" * 70)
    stats = kb.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ Knowledge base setup complete!")