import streamlit as st

def app():
    st.markdown('''<h1><center>Kidney Health Knowledge Centre</center></h1>''', unsafe_allow_html=True)
    
    # Paragraph 1: Kidney Disease Detection
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("./images/1.png", caption="Kidney Disease Detection", width=200)
    with col2:
        st.markdown('''
            The kidney health detection system is an AI-powered platform designed to identify kidney diseases at various stages using medical test data. By leveraging machine learning algorithms, the system analyzes essential health metrics such as creatinine levels, blood urea nitrogen (BUN), glomerular filtration rate (GFR), and proteinuria levels. It accurately classifies conditions such as chronic kidney disease (CKD), acute kidney injury (AKI), and nephrotic syndrome, ensuring early diagnosis and timely intervention for patients.
        ''')

    # Paragraph 2: Medical Recommendations
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('''
            Beyond detection, the system provides personalized medical recommendations based on a patient's specific kidney health status. These recommendations include dietary guidelines, hydration plans, medication suggestions, and lifestyle modifications to improve kidney function. Patients can also generate and download a detailed health report in PDF format, making it easy to share with healthcare professionals for further evaluation and treatment planning.
        ''')
    with col2:
        st.image("./images/2.png", caption="Medical Recommendations", width=200)

    # Paragraph 3: Kidney Health Chatbot
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("./images/3.png", caption="Kidney Health Chatbot", width=200)
    with col2:
        st.markdown('''
            A unique feature of the system is the Kidney Health Chatbot, an AI-driven assistant designed to address any kidney-related concerns. Using an extensive medical knowledge base and natural language processing (NLP), the chatbot provides answers to questions about symptoms, medication dosages, potential complications, and treatment options. Whether a patient seeks advice on managing CKD or understanding lab test results, the chatbot is available 24/7 for support and guidance.
        ''')

    # Paragraph 4: Trend Visualization
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('''
            The system includes an interactive data visualization module named Trend, which offers insights into global kidney health statistics. Through interactive charts, maps, and graphs, Trend presents data on the prevalence of kidney diseases across different regions, demographic risk factors, and historical trends. This feature is invaluable for researchers, healthcare professionals, and policymakers, enabling them to make data-driven decisions and develop effective strategies for kidney disease prevention and management.
        ''')
    with col2:
        st.image("./images/4.png", caption="Trend Visualization", width=200)

    # Paragraph 5: Streamlit Integration
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("./images/5.png", caption="Streamlit Integration", width=200)
    with col2:
        st.markdown('''
            Built using Streamlit, this kidney health detection system offers a smooth and interactive user experience. Streamlit's framework allows for seamless integration of machine learning models, data visualizations, and interactive features, making the system powerful yet easy to use. The platform is accessible via web browsers, ensuring compatibility across various devices. By combining AI-driven analysis, personalized recommendations, and interactive components like the chatbot and Trend module, this system plays a crucial role in enhancing kidney health awareness and disease management.
        ''')

# Run the app
if __name__ == "__main__":
    app()
