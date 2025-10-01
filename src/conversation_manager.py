"""
Conversation Manager - FINAL VERSION
Handles multi-turn conversations, context tracking, and clarifying questions
"""

from typing import List, Dict, Optional
from datetime import datetime


class ConversationManager:
    """
    Manages conversation context and multi-turn dialogue
    """
    
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.conversation_history = []
        self.current_context = {}
        
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Keep only last N messages
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        # Update context
        if role == 'user':
            self._update_context(content, metadata)
    
    def _update_context(self, content: str, metadata: Dict):
        """Update current context from user message"""
        if metadata:
            # Track intents
            if 'intents' in metadata:
                self.current_context['last_intent'] = metadata['intents'][0] if metadata['intents'] else None
                self.current_context['all_intents'] = metadata.get('intents', [])
            
            # Track entities
            if 'entities' in metadata:
                entities = metadata['entities']
                if entities.get('account_numbers'):
                    self.current_context['account'] = entities['account_numbers'][0]
                if entities.get('product_names'):
                    self.current_context['product'] = entities['product_names'][0]
                if entities.get('order_numbers'):
                    self.current_context['order'] = entities['order_numbers'][0]
    
    def get_context(self) -> Dict:
        """Get current conversation context"""
        return self.current_context.copy()
    
    def get_recent_messages(self, count: int = 3) -> List[Dict]:
        """Get recent messages"""
        return self.conversation_history[-count*2:] if self.conversation_history else []
    
    def is_follow_up_question(self, query: str) -> bool:
        """Detect if this is a follow-up question"""
        follow_up_indicators = [
            'what about', 'how about', 'and', 'also',
            'what if', 'but', 'however', 'still',
            'that', 'this', 'it', 'them', 'those'
        ]
        
        query_lower = query.lower()
        
        # Short queries are often follow-ups
        if len(query.split()) <= 5:
            for indicator in follow_up_indicators:
                if indicator in query_lower:
                    return True
        
        # Check if query references previous context
        if self.current_context:
            if 'that' in query_lower or 'this' in query_lower:
                return True
        
        return False
    
    def enhance_query_with_context(self, query: str) -> str:
        """Enhance query with context from previous conversation"""
        if not self.is_follow_up_question(query):
            return query
        
        # Get last user message for context
        recent_user_msgs = [m for m in self.conversation_history if m['role'] == 'user']
        if len(recent_user_msgs) < 1:
            return query
        
        # Get the previous query (before adding current one)
        last_query = recent_user_msgs[-1]['content']
        
        # Smart enhancement
        query_lower = query.lower()
        
        # Combine with previous context
        if any(phrase in query_lower for phrase in ['what if', 'what about', 'how about']):
            return f"{last_query} {query}"
        
        if any(word in query_lower for word in ['that', 'this', 'it']):
            return f"{last_query} {query}"
        
        if len(query.split()) <= 4:
            return f"{last_query} {query}"
        
        return query
    
    def is_completely_irrelevant(self, query: str, intents: List[str], confidence: float, similarity: float) -> bool:
        """Check if query is completely irrelevant to customer support"""
        query_lower = query.lower().strip()
        
        # Very short queries that aren't questions
        if len(query_lower) < 15 and '?' not in query:
            common_greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 
                              'good night', 'thanks', 'thank you', 'ok', 'okay', 
                              'bye', 'goodbye', 'happy', 'sad', 'wow', 'nice']
            if query_lower in common_greetings or any(query_lower.startswith(g) for g in common_greetings):
                return True
        
        # Off-topic keywords
        off_topic_keywords = [
            'birthday', 'weather', 'joke', 'game', 'recipe', 'news', 
            'sports', 'movie', 'music', 'song', 'poem', 'story'
        ]
        
        if any(keyword in query_lower for keyword in off_topic_keywords):
            return True
        
        # Low confidence AND low similarity = probably irrelevant
        if confidence < 0.35 and similarity < 0.60:
            return True
        
        # Very low similarity regardless of confidence
        if similarity < 0.50:
            return True
        
        return False
    
    def should_ask_clarification(self, query: str, intents: List[str], confidence: float, similarity: float = 0) -> Optional[str]:
        """Determine if clarification is needed"""
        # Very low confidence AND low similarity
        if confidence < 0.30 and similarity < 0.50:
            return "I'm not sure I understand your question. Could you please rephrase it? I can help you with billing, technical issues, account management, or complaints."
        
        # Low confidence
        if confidence < 0.25:
            return "I'm not sure I understand. Could you please provide more details about what you need help with?"
        
        # Multiple intents
        if len(intents) > 2 and confidence > 0.30:
            intent_list = ", ".join(intents[:-1]) + f" or {intents[-1]}"
            return f"I see you might be asking about {intent_list}. Which one would you like help with?"
        
        # Ambiguous with medium confidence
        if 0.25 <= confidence < 0.35 and len(intents) >= 2:
            clarifications = {
                'billing': "Are you asking about billing, payments, or subscription?",
                'technical': "Is this a technical issue with the app or website?",
                'account': "Do you need help with your account settings or profile?",
                'complaints': "Would you like to file a complaint or speak with a supervisor?"
            }
            
            primary_intent = intents[0]
            return clarifications.get(primary_intent, 
                "Could you provide more details about what you need help with?")
        
        return None
    
    def get_fallback_response(self, query: str, intents: List[str], similarity: float) -> str:
        """Generate appropriate fallback response"""
        # Check if completely irrelevant first
        if self.is_completely_irrelevant(query, intents, 0.30, similarity):
            return (
                "I don't understand that question. I'm a customer support chatbot that can help with:\n\n"
                "• Billing and payment questions\n"
                "• Technical issues and troubleshooting\n"
                "• Account management\n"
                "• Complaints and feedback\n\n"
                "Please ask a question related to customer support."
            )
        
        # Very low similarity
        if similarity < 0.60:
            if not intents:
                return (
                    "I couldn't understand your question. Could you please rephrase it?\n\n"
                    "I can help with:\n"
                    "• Billing questions\n"
                    "• Technical support\n"
                    "• Account management\n"
                    "• Filing complaints"
                )
            
            # Have intent but weak match
            intent = intents[0]
            fallbacks = {
                'billing': (
                    "I understand you're asking about billing, but I need more details. Try asking:\n"
                    "• How do I check my bill?\n"
                    "• What payment methods do you accept?\n"
                    "• Can I get a refund?\n"
                    "• How do I cancel my subscription?"
                ),
                'technical': (
                    "I understand you need technical help, but could you be more specific? Try:\n"
                    "• My app is crashing\n"
                    "• How do I reset my password?\n"
                    "• I can't log in\n"
                    "• How do I update the software?"
                ),
                'account': (
                    "I understand you're asking about your account, but I need more information. Try:\n"
                    "• How do I update my email?\n"
                    "• Can I change my username?\n"
                    "• How do I delete my account?\n"
                    "• How do I update my profile?"
                ),
                'complaints': (
                    "I understand you have a concern. To help you better, please be more specific:\n"
                    "• I received a damaged product\n"
                    "• The service quality is poor\n"
                    "• I want to speak to a manager\n"
                    "• My order hasn't arrived"
                )
            }
            
            return fallbacks.get(intent, 
                "I'm not sure I understand. Could you rephrase your question more clearly?")
        
        return "I couldn't find a relevant answer. Could you rephrase your question?"
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation yet"
        
        user_messages = [m for m in self.conversation_history if m['role'] == 'user']
        
        summary = f"Conversation with {len(user_messages)} queries"
        
        if self.current_context.get('all_intents'):
            intents = set(self.current_context['all_intents'])
            summary += f"\nTopics discussed: {', '.join(intents)}"
        
        return summary
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_history = []
        self.current_context = {}