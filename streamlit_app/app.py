import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
from pathlib import Path


@st.cache(allow_output_mutation=True)
def load_model():
    # Resolve model path relative to this file so it works regardless of working dir
    model_path = Path(__file__).resolve().parent.parent / "models" / "classifier.pkl"
    return joblib.load(model_path)

# Set page configuration
st.set_page_config(
    page_title="Model Prediction App",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    h1, h2, h3 {
        color: #1f449c;
    }
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .divider {
        border-bottom: 1px solid #e1e4e8;
    }
    .dataframe-container {
        max-height: 200px;
        overflow-y: auto;
    }
    .prediction-container {
        max-width: 300px;
    }
    .bar-container {
        max-width: 400px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔮 Predictive Model Application")
st.markdown("This application helps you classify your data into categories (P1, P2, P3, P4) using a pre-trained model.")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Create two columns for the main content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Data")
    
    # Upload CSV file with more descriptive instructions
    uploaded_file = st.file_uploader(
        "Upload your input CSV file (format must match training data)",
        type=["csv"],
        help="Make sure your CSV contains all the features required by the model"
    )
    
    # Add a sample data option
    st.markdown("#### Need a sample file?")
    if st.button("Download Sample Template"):
        try:
            sample_path = Path(__file__).resolve().parent / "sample.csv"
            df = pd.read_csv(sample_path)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sample_template.csv">Click here to download</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception:
            st.warning("Sample file not available. Please check the file path.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Add information about model
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Model Information")
    st.markdown("""
    - **Model Type**: Categorical Classifier
    - **Prediction Classes**: P1, P2, P3, P4
    - **File**: classifier.pkl
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Load and display the uploaded data
        try:
            with st.spinner("Processing your data..."):
                input_df = pd.read_csv(uploaded_file)
                
                # Show complete data with scrollbar
                st.markdown("### Uploaded Data")
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(input_df, height=200)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Process for prediction
                df = input_df.copy()
                
                # Load model and predict
                try:
                    model = load_model()
                    prediction = model.predict(df).ravel()
                    a_dict = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
                    prediction = np.array([a_dict[i] for i in prediction])
                    
                    # Create a new row for predictions and chart
                    pred_col1, pred_col2 = st.columns([1, 1.5])
                    
                    with pred_col1:
                        # Show predictions in a compact table
                        st.markdown("### Prediction Results")
                        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                        result_df = pd.DataFrame(prediction, columns=['Prediction'])
                        st.dataframe(result_df, height=200)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with pred_col2:
                        # Count distribution of predictions with narrower bars
                        value_counts = pd.Series(prediction).value_counts().reset_index()
                        value_counts.columns = ['Category', 'Count']
                        
                        st.markdown("### Distribution")
                        st.markdown('<div class="bar-container">', unsafe_allow_html=True)
                        st.bar_chart(value_counts.set_index('Category'), width=200, height=200)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Create output file with predictions
                    output = input_df.copy()
                    output["Prediction"] = prediction
                    
                    # Provide download option
                    st.markdown('<div class="success-message">', unsafe_allow_html=True)
                    st.markdown("**🎉 Prediction completed successfully!**")
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=output.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error in prediction: {e}")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to see prediction results.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# No footer