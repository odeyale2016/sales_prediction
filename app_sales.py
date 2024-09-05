import streamlit as st
import joblib
import numpy as np
# Load the trained model
sales_model = joblib.load('sales_model.pkl')




def main():
    # Streamlit app title
    st.title('Sales prediction Model')
    html_temp = """
    <div style="background-color:green; padding:10px">
    <h4 style="color:white; text-align:center;">Sales Prediction means predicting how much of product people will buy based on factors such as amount you spend to advertise your product, the segment of people you advertise for, or the platform you used for the advertisement. </h4>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)        
    # Input Text for Tv Sales
    tv_input = st.number_input('TV Sales', min_value=0.0, max_value=760.0, value=230.0)
    # Input Text for Radio Sales
    radio_input = st.number_input('Radio Sales', min_value=0.0, max_value=150.0, value=23.0)
    # Input Text for Tv Sales
    news_input = st.number_input('NewsPaper Sales', min_value=0.0, max_value=600.0, value=20.0)


    # Predict button

    if st.button('Predict'):
        features = np.array([[tv_input, radio_input, news_input]])
        prediction = sales_model.predict(features)
    

        st.success(f"Predicted Sales: {prediction[0]}")   
if __name__ == '__main__':
    main()