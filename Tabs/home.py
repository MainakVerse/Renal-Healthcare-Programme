import streamlit as st
import PIL

def app():
    st.title('Integrated Renal Health Care Program')
    st.image('./images/renal.png')

    
    st.markdown(
    """<p style="font-size:20px;">
            
**Kidney disease** is a chronic (long-lasting) health condition that affects how well your kidneys filter waste and excess fluids from your blood.  
There isn’t a complete cure yet for kidney disease, but maintaining a healthy lifestyle, eating a balanced diet, staying hydrated, and managing underlying conditions like diabetes and high blood pressure can help slow its progression.  
This **Web app** will help you predict whether a person has kidney disease or is at risk of developing it in the future by analyzing several health parameters using the **Random Forest Classifier**.
        </p>
    """, unsafe_allow_html=True)
    