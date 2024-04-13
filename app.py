import streamlit as st
import base64
import pandas as pd
from io import BytesIO
from src.plot import plot

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# set_background('image/background.png')

st.title('Human Brain Activity Classification')

st.info('This app classifies human brain activity into different categories.')

uploaded_file = st.file_uploader("Upload EEG data", type=["parquet"])

if uploaded_file is not None:
    try:
        data = BytesIO(uploaded_file.getvalue())
        st.success("File is readable.")

        df = pd.read_parquet(data)
        plot_obj = None
        col1, col2, col3 = st.columns([1, 1, 10])
        with col1:
            if st.button("Create Plot", help="Click to create a plot", key="create_plot"):
                plot_obj = plot(df)
                if plot_obj is not None:
                    fig, ax = plot_obj
        with col2:
            if st.button("Predict", help="Click to make predictions", key="predict"):
                # Code for prediction
                st.write("Prediction Performed!")
        with col3:
            if st.button("Export", help="Export the results", key="export"):
                # Code for prediction
                st.write("Export Successfully!")

        if plot_obj is not None:
            st.pyplot(fig)
            st.write("Plot Created!")

    except Exception as e:
        st.error(f"Error: {e}")
