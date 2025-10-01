#!/bin/bash

# Customer Support Chatbot Launcher

echo "🤖 Customer Support Chatbot"
echo "============================"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    echo ""
    exit 1
fi

# Check if models exist
if [ ! -d "models" ]; then
    echo "❌ Models directory not found!"
    echo "Please train models first:"
    echo "  python src/train_intent.py"
    echo "  python src/knowledge_base.py"
    exit 1
fi

# Launch Streamlit
echo "🚀 Launching Streamlit app..."
echo ""
streamlit run app.py