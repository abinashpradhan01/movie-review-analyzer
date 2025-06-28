import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .score-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading to improve performance
@st.cache_resource
def load_sentiment_model():
    """Load the pre-trained sentiment analysis model"""
    try:
        # Load the IMDB dataset word index
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        # Load the pre-trained model
        model = load_model('best_simple_rnn_imdb.keras')
        
        return model, word_index, reverse_word_index
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Helper Functions
def decode_review(encoded_review, reverse_word_index):
    """Decode encoded review back to text"""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text, word_index):
    """Preprocess user input text for model prediction"""
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review, model, word_index):
    """Predict sentiment of the review"""
    try:
        preprocessed_input = preprocess_text(review, word_index)
        prediction = model.predict(preprocessed_input, verbose=0)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        return sentiment, float(prediction[0][0])
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def create_confidence_gauge(score):
    """Create a confidence gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_sentiment_bar(score):
    """Create a sentiment probability bar chart"""
    sentiments = ['Negative', 'Positive']
    probabilities = [1 - score, score]
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = px.bar(
        x=sentiments, 
        y=probabilities, 
        color=sentiments,
        color_discrete_map={'Negative': '#FF6B6B', 'Positive': '#4ECDC4'},
        title="Sentiment Probability"
    )
    fig.update_layout(
        showlegend=False,
        yaxis_title="Probability",
        xaxis_title="Sentiment",
        height=400
    )
    return fig

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This app uses a pre-trained RNN model to analyze the sentiment of movie reviews. 
    Simply enter a movie review and get instant sentiment analysis!
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, word_index, reverse_word_index = load_sentiment_model()
    
    if model is None:
        st.error("Failed to load the model. Please ensure 'best_simple_rnn_imdb.keras' is in the same directory.")
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Movie Review")
        
        # Text input
        review_text = st.text_area(
            "Type your movie review here:",
            height=150,
            placeholder="Example: This movie was fantastic! The acting was great and the plot was thrilling."
        )
        
        # Predict button
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if review_text.strip():
                with st.spinner("Analyzing sentiment..."):
                    sentiment, score = predict_sentiment(review_text, model, word_index)
                    
                    if sentiment is not None:
                        # Store results in session state
                        st.session_state.last_sentiment = sentiment
                        st.session_state.last_score = score
                        st.session_state.last_review = review_text
            else:
                st.warning("Please enter a movie review to analyze.")
    
    with col2:
        st.subheader("📊 Model Information")
        
        # Model summary in expandable section
        with st.expander("View Model Architecture"):
            if model is not None:
                # Create a string representation of model summary
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
        
        # Dataset info
        st.info("""
        **Model Details:**
        - Dataset: IMDB Movie Reviews
        - Architecture: Simple RNN with ReLU
        - Vocabulary: 10,000 most common words
        - Sequence Length: 500 tokens
        """)
    
    # Display results if available
    if hasattr(st.session_state, 'last_sentiment'):
        st.markdown("---")
        st.subheader("🎯 Analysis Results")
        
        # Create three columns for results
        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
        
        with result_col1:
            st.markdown("**Original Review:**")
            st.write(f"_{st.session_state.last_review}_")
            
            # Sentiment result
            sentiment_class = "sentiment-positive" if st.session_state.last_sentiment == "Positive" else "sentiment-negative"
            st.markdown(f'<div class="{sentiment_class}">Sentiment: {st.session_state.last_sentiment}</div>', unsafe_allow_html=True)
            
            # Score
            st.metric("Prediction Score", f"{st.session_state.last_score:.4f}")
        
        with result_col2:
            # Confidence gauge
            gauge_fig = create_confidence_gauge(st.session_state.last_score)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with result_col3:
            # Sentiment bar chart
            bar_fig = create_sentiment_bar(st.session_state.last_score)
            st.plotly_chart(bar_fig, use_container_width=True)
        
        # Interpretation
        st.markdown("---")
        st.subheader("💡 Interpretation")
        
        confidence_level = "High" if abs(st.session_state.last_score - 0.5) > 0.3 else "Medium" if abs(st.session_state.last_score - 0.5) > 0.1 else "Low"
        
        st.write(f"""
        **Confidence Level:** {confidence_level}
        
        **Explanation:** 
        - Scores closer to 0 indicate negative sentiment
        - Scores closer to 1 indicate positive sentiment
        - Your review scored {st.session_state.last_score:.4f}, suggesting a **{st.session_state.last_sentiment.lower()}** sentiment
        """)
    
    # Example reviews section
    st.markdown("---")
    st.subheader("💡 Try These Example Reviews")
    
    examples = [
        "This movie was absolutely amazing! The cinematography was breathtaking and the story was compelling.",
        "Terrible movie. Poor acting, weak plot, and completely boring. Waste of time.",
        "The movie was okay. Not great, not terrible. Some parts were interesting.",
        "Outstanding performance by the lead actor. One of the best films I've seen this year!",
        "Disappointing sequel. The original was much better. This felt rushed and poorly written."
    ]
    
    example_cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.example_review = example
                # Auto-analyze the example
                sentiment, score = predict_sentiment(example, model, word_index)
                if sentiment is not None:
                    st.session_state.last_sentiment = sentiment
                    st.session_state.last_score = score
                    st.session_state.last_review = example
                    st.experimental_rerun()
    
    # Display selected example
    if hasattr(st.session_state, 'example_review'):
        st.text_area("Selected Example:", value=st.session_state.example_review, height=100, disabled=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-style: italic; margin-top: 2rem;'>
        I have no shame in accepting that the code for this Streamlit app is written by Claude 🤖
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()