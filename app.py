import streamlit as st
import requests
import os 
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Text Normalization Tool",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Helper function to normalize comments in parallel
def normalize_comments_batch(comments: List[str], api_endpoint: str) -> List[Dict[str, Any]]:
    results = []
    with ThreadPoolExecutor(max_workers=len(comments)) as executor:
        future_to_comment = {
            executor.submit(normalize_comment, comment, api_endpoint, idx+1): comment
            for idx, comment in enumerate(comments)
        }

        for future in as_completed(future_to_comment):
            results.append(future.result())
    return results 


def normalize_comment(comment, api_endpoint, comment_id):
    payload = {
        "comment_id": comment_id,
        "text": comment.strip()
    }
    try:
        response = requests.post(
            f"{api_endpoint}/normalize",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            result['original_text'] = comment
            return result
        else:
            return {'error': response.text, 'original_text': comment}
    except Exception as e:
        return {'error': str(e), 'original_text': comment}

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .text-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }
    .copy-button {
        background: #28a745;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📝 Text Normalization Tool</h1>
    <p>Professional text normalization for insurance underwriting using AWS Bedrock Nova LLM</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API endpoint configuration
    api_endpoint = st.text_input(
        "API Endpoint",
        value=os.environ.get("API_ENDPOINT", "http://localhost:8000"),
        help="Backend API server endpoint"
    )
    
    # Test mode toggle
    test_mode = st.checkbox("Test Mode", value=True, help="Use test data for demonstration")
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0
    st.metric("Total Processed", st.session_state.total_processed)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Comments to Normalize (one per line)")
    
    # Text input
    if test_mode:
        # Pre-filled test cases
        test_cases = [
            "Loan-to-value high. Need bring down to 80.5%. Risk too big.",
            "this policy need review for 15% increase",
            "claim amount is between 7.5 ~ 8 thousand dollars",
            "customer satisfaction rate below 75%",
            "premium adjustment required for high risk cases"
        ]
        
        selected_test = st.selectbox(
            "Select a test case:",
            test_cases,
            index=0
        )
        
        user_text = st.text_area(
            "Text to normalize:",
            value=selected_test,
            height=150,
            placeholder="Enter your text here..."
        )
    else:
        user_text = st.text_area(
            "Enter one comment per line:",
            height=150,
            placeholder="Comment 1\nComment 2\nComment 3"
        )

with col2:
    st.header("🚀 Actions")

    
    # Normalize all comments button
    if st.button("🔄 Normalize All Comments", type="primary", use_container_width=True):
        comments = [line.strip() for line in user_text.split('\n') if line.strip()]
        if comments:
            with st.spinner("Processing all comments in parallel..."):
                results = normalize_comments_batch(comments, api_endpoint)
                st.session_state.results = results
                st.session_state.total_processed = len(results)
                st.success("✅ All comments normalized successfully!")
        else:
            st.warning("⚠️ Please enter some comments to normalize.")
    
    # Normalize button
    if st.button("🔄 Normalize Text", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Processing with AWS Bedrock Nova LLM..."):
                try:
                    # Prepare request
                    payload = {
                        "comment_id": st.session_state.total_processed + 1,
                        "text": user_text.strip()
                    }
                    
                    # Make API call
                    response = requests.post(
                        f"{api_endpoint}/normalize",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store result in session state
                        if 'results' not in st.session_state:
                            st.session_state.results = []
                        
                        st.session_state.results.append(result)
                        st.session_state.total_processed += 1
                        
                        st.success("✅ Text normalized successfully!")
                        
                    else:
                        st.error(f"❌ Error: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to normalize.")

    # Clear results button
    if st.button("🗑️ Clear All Results", use_container_width=True):
        st.session_state.results = []
        st.session_state.total_processed = 0
        st.success("All results cleared!")

# Display results
if 'results' in st.session_state and st.session_state.results:
    st.header("📊 Results")
    
    for i, result in enumerate(st.session_state.results):
        print(f"Result: {result}")
        with st.container():
            st.markdown("""
                <style>
                    .result-card {
                        color: black !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card">
                <h4>Comment #{result.get('comment_id','')} - Processed in {result.get('processing_time',0):.3f}s</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📄 Original Text:**")
                st.markdown("""
                <style>
                    .text-box {
                        color: black !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="text-box">
                    {result.get('original_text','')}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy original text
                if st.button(f"📋 Copy Original", key=f"copy_orig_{i}"):
                    st.write("Original text copied to clipboard!")
                    st.code(result.get('original_text',''))
            
            with col2:
                st.markdown("**✨ Normalized Text:**")
                st.markdown(f"""
                <div class="text-box">
                    {result.get('normalized_text', result.get('error', 'Error'))}
                </div>
                """, unsafe_allow_html=True)
                
                # Copy normalized text
                if st.button(f"📋 Copy Normalized", key=f"copy_norm_{i}"):
                    st.write("Normalized text copied to clipboard!")
                    st.code(result.get('normalized_text',''))
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Powered by AWS Bedrock Nova LLM</p>
</div>
""", unsafe_allow_html=True)

# Debug information (expandable)
with st.expander("🔧 Debug Information"):
    st.json({
        "API Endpoint": api_endpoint,
        "Test Mode": test_mode,
        "Total Processed": st.session_state.total_processed,
        "Results Count": len(st.session_state.results) if 'results' in st.session_state else 0
    })

# Simple Streamlit app - no complex classes
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Powered by AWS Bedrock Nova LLM</p>
</div>
""", unsafe_allow_html=True)

# Debug information (expandable)
with st.expander("🔧 Debug Information"):
    st.json({
        "API Endpoint": api_endpoint,
        "Test Mode": test_mode,
        "Total Processed": st.session_state.total_processed,
        "Results Count": len(st.session_state.results) if 'results' in st.session_state else 0
    })


