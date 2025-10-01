# Customer Support Chatbot

An intelligent AI-powered customer support chatbot built using Intent Classification, RAG (Retrieval-Augmented Generation), and conversation management techniques.                                                                                                         Deployed Link -- https://palak-23-customer-support-chatbot-app-ryawox.streamlit.app/

## Features

- **Multi-label Intent Classification**: Detects billing, technical, account, and complaint queries
- **Entity Extraction**: Identifies account numbers, product names, order numbers, dates, and amounts
- **Semantic Search**: FAISS-powered knowledge base with 70+ FAQ entries
- **Context-Aware Conversations**: Tracks multi-turn conversations and understands follow-up questions
- **Smart Fallbacks**: Handles ambiguous and irrelevant queries gracefully
- **Real-time Analytics**: User satisfaction tracking, performance metrics, and continuous learning

## Technologies Used

### Machine Learning
- **scikit-learn**: Intent classification (TF-IDF + Logistic Regression)
- **Sentence Transformers**: Text embeddings (all-MiniLM-L6-v2)
- **FAISS**: Fast similarity search for semantic matching

### NLP
- Multi-label classification (OneVsRest)
- Named entity recognition (Regex-based)
- Context tracking and conversation management

### Framework
- **Streamlit**: Web interface
- **Pandas**: Data processing
- **NumPy**: Numerical operations

## Project Structure

```
customer-support-chatbot/
├── app.py                          # Main Streamlit application
├── setup.py                        # Automatic model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   ├── intents.csv                 # Intent training data (90 examples)
│   └── faq_knowledge_base.csv      # FAQ database (70+ entries)               
├── src/
│   ├── intent_predictor.py         # Intent classification
│   ├── entity_extractor.py         # Entity extraction
│   ├── knowledge_base.py           # RAG implementation
│   ├── train_intent.py             # Model training
│   ├── conversation_manager.py     # Context tracking
│   └── analytics.py                # Performance tracking
├── tests/
│   ├── quick_tests.py              # Intent classification tests
│   ├── test_rag.py                 # Knowledge base tests
│   ├── test_conversation.py        # Conversation tests
│   ├── test_analytics.py           # Analytics tests
│   └── test_integration.py         # Full pipeline tests
└── analytics/                      # Created automatically
    ├── queries_log.csv             # All interactions logged
    └── failed_queries.csv          # Failed queries tracked
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/customer-support-chatbot.git
cd customer-support-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train models**
```bash
python setup.py
```

This will:
- Train the intent classifier (~30 seconds)
- Build the FAISS knowledge base (~1 minute)
- Save models to `models/` directory

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Sample Queries

**Billing Questions:**
- "How do I check my bill?"
- "What payment methods do you accept?"
- "Can I get a refund?"

**Technical Support:**
- "How do I reset my password?"
- "My app keeps crashing"
- "I can't log in"

**Account Management:**
- "How do I update my email?"
- "Can I change my username?"
- "How do I delete my account?"

**Complaints:**
- "I received a damaged product"
- "I want to speak to a manager"
- "My order hasn't arrived"

### Multi-Turn Conversations

The bot understands context:
```
You: "How do I reset my password?"
Bot: [Provides password reset instructions]

You: "What if I don't get an email?"
Bot: [Understands you're asking about password reset email]
```

## Features in Detail

### Part 1: Intent Classification

- **Model**: TF-IDF + Logistic Regression (OneVsRest)
- **Intents**: billing, technical, account, complaints
- **Multi-label support**: Detects multiple intents per query
- **Entity extraction**: Accounts, products, orders, dates, amounts

### Part 2: Knowledge Base (RAG)

- **Embeddings**: Sentence Transformers (384 dimensions)
- **Search**: FAISS IndexFlatL2 for fast similarity search
- **Database**: 70+ FAQ entries
- **Accuracy**: 85-99% similarity for exact matches

### Part 3: Conversation Management

- **Context tracking**: Remembers last 5 message exchanges
- **Follow-up detection**: Understands "what if", "that", "this"
- **Query enhancement**: Combines follow-ups with previous context
- **Clarifying questions**: Asks for details when ambiguous
- **Smart fallbacks**: Intent-specific helpful responses

### Part 4: Analytics & Improvement

- **User satisfaction**: Thumbs up/down feedback after each response
- **Performance metrics**: Confidence, similarity, response time
- **Intent distribution**: Track which topics are most common
- **Failed queries**: Log low-quality responses for improvement
- **CSV logging**: All interactions saved for model retraining

## Testing

Run individual test suites:

```bash
# Intent classification
python tests/quick_tests.py

# Knowledge base
python tests/test_rag.py

# Conversation management
python tests/test_conversation.py

# Analytics
python tests/test_analytics.py

# Full integration
python tests/test_integration.py
```

## Performance Metrics

### Intent Classification
- Training examples: 90
- Hamming accuracy: ~72%
- Confidence scores: 30-50% (normal for small dataset)

### Knowledge Base
- FAQ entries: 70+
- Embedding dimension: 384
- Search speed: <1ms per query
- Similarity range: 0-100%

### System Performance
- Average response time: 0.3-0.5 seconds
- Model loading: ~3-5 seconds (one-time)
- Memory usage: ~200MB

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your repository
4. Deploy

Models will be trained automatically on first run using `setup.py`.

### Local Network

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Access at `http://YOUR_IP:8501`

## Continuous Improvement

The system logs all interactions for improvement:

1. **Failed queries** are tracked in `analytics/failed_queries.csv`
2. **Add failed queries** to `data/intents.csv` for retraining
3. **Retrain model**: `python src/train_intent.py`
4. **Add missing FAQs** to `data/faq_knowledge_base.csv`
5. **Rebuild index**: `python src/knowledge_base.py`

## Known Limitations

1. **Small training dataset**: 90 examples (can be expanded)
2. **Limited FAQ coverage**: 70 entries (easily extensible)
3. **Regex entities**: Works for structured data only
4. **Session-based memory**: Context resets on refresh

## Future Enhancements

- Integrate LLM (GPT/Claude) for better response generation
- Voice input/output support
- Multi-language support
- Sentiment analysis
- Live agent handoff
- Integration with ticketing systems

## Requirements

See `requirements.txt` for full list. Key dependencies:

- streamlit
- scikit-learn
- sentence-transformers
- faiss-cpu
- pandas
- numpy

## License

This project was created for educational purposes as part of an AI assignment.

## Authors

Built as part of an AI Customer Support Chatbot assignment.

## Acknowledgments

- Sentence Transformers for text embeddings
- FAISS for efficient similarity search
- Streamlit for the web framework
- scikit-learn for machine learning tools

## Support

For issues or questions, please check:
- Test suites in `tests/` directory
- Analytics logs in `analytics/` directory
- Code documentation in source files

---

**Note**: On first run, the app will automatically train models using `setup.py`. This takes 1-2 minutes and only happens once.
