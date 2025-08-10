import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import time
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .sidebar-info {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the sentiment analysis model with caching for better performance"""
    try:
        classifier = pipeline(
            'text-classification', 
            model='bert-base-uncased-sentiment-model',
            return_all_scores=True
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_text(text):
    """Basic text preprocessing"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags for cleaner analysis (optional)
    # text = re.sub(r'@\w+|#\w+', '', text)
    
    return text

def create_confidence_chart(results):
    """Create a horizontal bar chart for sentiment confidence scores"""
    
    # Extract labels and scores
    labels = [result['label'].title() for result in results]
    scores = [result['score'] for result in results]
    
    # Create color mapping for sentiments
    color_map = {
        'Joy': '#fbbf24',
        'Sadness': '#3b82f6', 
        'Anger': '#ef4444',
        'Fear': '#8b5cf6',
        'Surprise': '#f97316',
        'Love': '#ec4899'
    }
    
    colors = [color_map.get(label, '#6b7280') for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            y=labels,
            x=scores,
            orientation='h',
            marker_color=colors,
            text=[f'{score:.2%}' for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Confidence Scores",
        xaxis_title="Confidence Score",
        yaxis_title="Emotion",
        height=400,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 BERT Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280;">Fine-tuned BERT model for multi-class sentiment classification of Twitter tweets</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Information")
        st.markdown("**Model:** BERT Base Uncased")
        st.markdown("**Training:** 10 epochs on Twitter sentiment data")
        st.markdown("**Classes:** Joy, Sadness, Anger, Fear, Surprise, Love")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model settings
        st.markdown("### ⚙️ Settings")
        show_all_scores = st.checkbox("Show all sentiment scores", value=True)
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
        
        # About section
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            This application uses a fine-tuned BERT model to classify the emotional sentiment 
            of text into six categories: Joy, Sadness, Anger, Fear, Surprise, and Love.
            
            **How to use:**
            1. Enter your text in the text area
            2. Click 'Analyze Sentiment'
            3. View the results and confidence scores
            """)
    
    # Load model
    with st.spinner("Loading model..."):
        classifier = load_model()
    
    if classifier is None:
        st.error("Failed to load the sentiment analysis model. Please check if the model files exist.")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Enter Text for Analysis</h2>', unsafe_allow_html=True)
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Single text", "Batch analysis"])
        
        if input_method == "Single text":
            # Single text analysis
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste your text here... (e.g., 'I am so excited about this new project!')"
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
            with col_btn2:
                clear_button = st.button("🗑️ Clear")
            
            if clear_button:
                st.experimental_rerun()
            
            if analyze_button and text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    # Preprocess text
                    processed_text = preprocess_text(text_input)
                    
                    # Get prediction
                    start_time = time.time()
                    results = classifier(processed_text)
                    prediction_time = time.time() - start_time
                    
                    # Display results
                    st.markdown('<h2 class="sub-header">📈 Analysis Results</h2>', unsafe_allow_html=True)
                    
                    # Get top prediction
                    top_prediction = max(results[0], key=lambda x: x['score'])
                    
                    # Metrics row
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Predicted Emotion", top_prediction['label'].title())
                    with col_m2:
                        st.metric("Confidence", f"{top_prediction['score']:.2%}")
                    with col_m3:
                        st.metric("Processing Time", f"{prediction_time:.3f}s")
                    with col_m4:
                        st.metric("Text Length", f"{len(processed_text)} chars")
                    
                    # Confidence indicator
                    if top_prediction['score'] >= confidence_threshold:
                        st.success(f"✅ High confidence prediction: **{top_prediction['label'].title()}**")
                    else:
                        st.warning(f"⚠️ Low confidence prediction: **{top_prediction['label'].title()}**")
                    
                    # Show all scores if enabled
                    if show_all_scores:
                        st.markdown("### Detailed Sentiment Scores")
                        
                        # Create and display chart
                        fig = create_confidence_chart(results[0])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed scores table
                        df_results = pd.DataFrame(results[0])
                        df_results['score'] = df_results['score'].apply(lambda x: f"{x:.4f}")
                        df_results['percentage'] = df_results['score'].astype(float).apply(lambda x: f"{x:.2%}")
                        df_results['label'] = df_results['label'].str.title()
                        df_results = df_results.sort_values('score', ascending=False)
                        
                        st.dataframe(
                            df_results[['label', 'percentage']].rename(columns={'label': 'Emotion', 'percentage': 'Confidence'}),
                            use_container_width=True
                        )
            
            elif analyze_button and not text_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
        
        else:
            # Batch analysis
            st.markdown("### 📊 Batch Analysis")
            uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("The CSV file must contain a 'text' column.")
                    else:
                        st.success(f"Loaded {len(df)} texts for analysis.")
                        
                        if st.button("🚀 Analyze All Texts"):
                            progress_bar = st.progress(0)
                            results_list = []
                            
                            for i, text in enumerate(df['text'].fillna('')):
                                if text.strip():
                                    processed_text = preprocess_text(text)
                                    result = classifier(processed_text)
                                    top_pred = max(result[0], key=lambda x: x['score'])
                                    results_list.append({
                                        'text': text[:100] + '...' if len(text) > 100 else text,
                                        'predicted_emotion': top_pred['label'].title(),
                                        'confidence': top_pred['score']
                                    })
                                else:
                                    results_list.append({
                                        'text': text,
                                        'predicted_emotion': 'N/A',
                                        'confidence': 0.0
                                    })
                                
                                progress_bar.progress((i + 1) / len(df))
                            
                            # Display results
                            results_df = pd.DataFrame(results_list)
                            st.markdown("### 📋 Batch Analysis Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            st.markdown("### 📊 Summary Statistics")
                            emotion_counts = results_df['predicted_emotion'].value_counts()
                            
                            fig_pie = px.pie(
                                values=emotion_counts.values,
                                names=emotion_counts.index,
                                title="Distribution of Predicted Emotions"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Quick Examples</h2>', unsafe_allow_html=True)
        
        example_texts = {
            "😊 Joy": "I just got promoted at work! This is the best day ever!",
            "😢 Sadness": "I'm feeling really down today. Everything seems to be going wrong.",
            "😠 Anger": "I can't believe they cancelled my flight again! This is so frustrating!",
            "😨 Fear": "I'm really nervous about the job interview tomorrow.",
            "😲 Surprise": "Wow! I never expected to win the lottery!",
            "❤️ Love": "I love spending time with my family on weekends."
        }
        
        st.markdown("Click on any example to try it:")
        
        for emotion, example_text in example_texts.items():
            if st.button(f"{emotion}", key=f"example_{emotion}"):
                # This would ideally populate the text area, but Streamlit doesn't support this directly
                # Instead, we'll show the example and let users copy it
                st.code(example_text)
                st.info("👆 Copy this text and paste it in the text area above")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6b7280;">Made with ❤️ using Streamlit and Transformers | '
        f'Last updated: {datetime.now().strftime("%Y-%m-%d")}</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()