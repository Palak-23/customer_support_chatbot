"""
Customer Support Chatbot - FINAL VERSION
Complete Streamlit Application with all features
"""

import streamlit as st
import sys
import os
import shutil
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from intent_predictor import IntentPredictor
from knowledge_base import KnowledgeBase
from conversation_manager import ConversationManager
from analytics import Analytics

# Page configuration
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Minimal CSS
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .user-msg {
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    .bot-msg {
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load models once and cache them"""
    predictor = IntentPredictor(model_dir='models')
    kb = KnowledgeBase(faq_path='data/faq_knowledge_base.csv')
    kb.load_index(index_path='models/faiss_index')
    return predictor, kb


def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(max_history=5)
    
    if 'analytics' not in st.session_state:
        st.session_state.analytics = Analytics()
    
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = {}


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.title("🤖 Customer Support Chatbot")
    st.markdown("Ask questions about billing, technical issues, account management, or complaints")
    st.markdown("---")
    
    # Load models
    try:
        with st.spinner("Loading AI models..."):
            predictor, kb = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure you've run: python src/train_intent.py and python src/knowledge_base.py")
        st.stop()
    
    # Sidebar with Analytics
    with st.sidebar:
        st.header("📊 Analytics Dashboard")
        
        # Get analytics stats
        stats = st.session_state.analytics.get_statistics()
        
        if stats['total_queries'] == 0:
            st.info("No queries yet. Start chatting to see analytics!")
        else:
            # Key Metrics
            st.subheader("Performance Metrics")
            st.metric("Total Queries", stats['total_queries'])
            st.metric("Avg Intent Confidence", f"{stats['avg_confidence']:.1%}")
            st.metric("Avg Answer Match", f"{stats['avg_similarity']:.1%}")
            st.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
            
            # User Satisfaction
            st.subheader("User Satisfaction")
            st.metric("Satisfaction Rate", f"{stats['satisfaction_rate']:.1f}%")
            
            # Intent Distribution
            if stats['intent_distribution']:
                st.subheader("Intent Distribution")
                for intent, count in stats['intent_distribution'].items():
                    st.text(f"{intent}: {count}")
            
            # Failed Queries
            st.subheader("Quality Metrics")
            st.metric("Failed Queries", stats['failed_queries_count'])
        
        st.markdown("---")
        st.markdown("### Sample Questions")
        st.markdown("""
        - How do I reset my password?
        - What payment methods do you accept?
        - How do I cancel my subscription?
        - My app keeps crashing
        - How do I update my email?
        """)
        
        st.markdown("---")
        
        # Clear All button
        if st.button("Clear All", use_container_width=True):
            # Clear session
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.conversation_manager = ConversationManager(max_history=5)
            st.session_state.feedback_given = {}
            
            # Delete CSV files to start fresh
            if os.path.exists('analytics'):
                shutil.rmtree('analytics')
            
            # Reinitialize analytics
            st.session_state.analytics = Analytics()
            
            st.rerun()
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="user-msg">
                    <strong>You:</strong><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            # Bot message
            st.markdown(f"""
                <div class="bot-msg">
                    <strong>Bot:</strong><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
            
            # Show metadata
            metadata = message.get('metadata', {})
            if metadata:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    intents = metadata.get('intents', [])
                    if intents:
                        intent_str = ', '.join([i.upper() for i in intents])
                        st.markdown(f"**Intent:** {intent_str}")
                
                with col2:
                    confidence = metadata.get('confidence', 0)
                    if confidence > 0:
                        st.markdown(f"**Confidence:** {confidence:.0%}")
                
                with col3:
                    similarity = metadata.get('similarity', 0)
                    if similarity > 0:
                        st.markdown(f"**Match:** {similarity:.0%}")
                
                # Show entities if any
                entities = metadata.get('entities', '')
                if entities and entities != "No entities found":
                    st.caption(f"📋 {entities}")
            
            # Feedback buttons
            feedback_key = f"feedback_{idx}"
            if feedback_key not in st.session_state.feedback_given:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍 Helpful", key=f"pos_{idx}"):
                        st.session_state.feedback_given[feedback_key] = "positive"
                        # Update analytics
                        query_idx = idx // 2  # Each query has 2 messages
                        st.session_state.analytics.update_feedback(query_idx, "positive")
                        st.rerun()
                
                with col2:
                    if st.button("👎 Not Helpful", key=f"neg_{idx}"):
                        st.session_state.feedback_given[feedback_key] = "negative"
                        # Update analytics and log as failed
                        query_idx = idx // 2
                        st.session_state.analytics.update_feedback(query_idx, "negative")
                        # Log as failed query
                        if idx > 0:
                            user_msg = st.session_state.messages[idx-1]
                            st.session_state.analytics.log_failed_query(
                                user_msg['content'],
                                metadata.get('intents', []),
                                metadata.get('confidence', 0),
                                metadata.get('similarity', 0),
                                "User marked as not helpful"
                            )
                        st.rerun()
            else:
                # Show feedback was given
                feedback = st.session_state.feedback_given[feedback_key]
                if feedback == "positive":
                    st.caption("✓ Marked as helpful")
                else:
                    st.caption("✓ Feedback recorded")
            
            st.markdown("---")
    
    # Welcome message
    if len(st.session_state.messages) == 0:
        st.info("👋 Welcome! Type your question below to get started.")
    
    # Chat input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Add user message to display
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        st.session_state.query_count += 1
        
        # Get conversation manager
        conv_manager = st.session_state.conversation_manager
        
        # Process query
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            # Check if follow-up
            is_followup = conv_manager.is_follow_up_question(user_input)
            
            # Enhance query if follow-up
            if is_followup:
                enhanced_query = conv_manager.enhance_query_with_context(user_input)
            else:
                enhanced_query = user_input
            
            # Get intent (use original query)
            intent_result = predictor.predict(user_input)
            
            # Search knowledge base (use enhanced query)
            answer_result = kb.get_contextual_answer(
                enhanced_query,
                intent_result['intents'],
                intent_result['entities']
            )
            
            similarity = answer_result.get('similarity', 0)
            
            # Decide response based on relevance
            is_irrelevant = conv_manager.is_completely_irrelevant(
                user_input,
                intent_result['intents'],
                intent_result['overall_confidence'],
                similarity
            )
            
            if is_irrelevant:
                # Completely off-topic
                response_text = conv_manager.get_fallback_response(
                    user_input,
                    intent_result['intents'],
                    similarity
                )
            elif similarity < 0.70 and intent_result['overall_confidence'] < 0.35:
                # Ambiguous - try clarification
                clarification = conv_manager.should_ask_clarification(
                    user_input,
                    intent_result['intents'],
                    intent_result['overall_confidence'],
                    similarity
                )
                if clarification:
                    response_text = clarification
                else:
                    response_text = conv_manager.get_fallback_response(
                        user_input,
                        intent_result['intents'],
                        similarity
                    )
            elif similarity < 0.65:
                # Weak match - suggest better questions
                response_text = conv_manager.get_fallback_response(
                    user_input,
                    intent_result['intents'],
                    similarity
                )
            else:
                # Good answer found
                response_text = answer_result['answer']
            
            response_time = time.time() - start_time
            
            # Log to analytics
            st.session_state.analytics.log_query(
                query=user_input,
                intents=intent_result['intents'],
                confidence=intent_result['overall_confidence'],
                similarity=similarity,
                response_time=response_time
            )
            
            # Log failed query if low quality
            if is_irrelevant or similarity < 0.60 or intent_result['overall_confidence'] < 0.35:
                reason = []
                if is_irrelevant:
                    reason.append("Irrelevant query")
                if similarity < 0.60:
                    reason.append(f"Low similarity ({similarity:.2f})")
                if intent_result['overall_confidence'] < 0.35:
                    reason.append(f"Low confidence ({intent_result['overall_confidence']:.2f})")
                
                st.session_state.analytics.log_failed_query(
                    query=user_input,
                    intents=intent_result['intents'],
                    confidence=intent_result['overall_confidence'],
                    similarity=similarity,
                    reason="; ".join(reason)
                )
            
            # Add to conversation history
            conv_manager.add_message('user', user_input, {
                'intents': intent_result['intents'],
                'entities': intent_result['entities']
            })
            conv_manager.add_message('assistant', response_text, {})
        
        # Add bot response to display
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response_text,
            'metadata': {
                'intents': intent_result['intents'],
                'confidence': intent_result['overall_confidence'],
                'similarity': similarity,
                'entities': intent_result['entity_summary'],
                'response_time': response_time,
                'is_followup': is_followup
            }
        })
        
        st.rerun()


if __name__ == "__main__":
    main()