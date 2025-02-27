import streamlit as st
from web_functions import predict
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import io
import os
import csv
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Load API Key from Streamlit secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


if not GEMINI_API_KEY:
    raise ValueError("Gemini API key is missing! Add it to Streamlit secrets.")

genai.configure(api_key=GEMINI_API_KEY)

def app(df, X, y):
    """This function creates the Streamlit app with tabs."""
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 24px;
        color: #0000cc; /* Neon cyan text color */
        
    }
</style>

""", unsafe_allow_html=True)
    # Create two tabs
    tab1, tab2, tab3 = st.tabs(["Diagnosis 🩺", "Medication 💊", "Data Source 🛢️"])

    # First Tab: Prediction Page
    with tab1:
        st.title("Diagnosis Page")
        st.write("The aim is to detect the different types of diabetes and the risk of onset from the clinical test data. This makes the detection process extremely fast and feature-rich augmenting treatment experience and ease of access for both patient and physician")

        # Take input of features from the user
        st.subheader("Select Values:")
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()))
            Gender = st.slider("Gender", int(df["Gender"].min()), int(df["Gender"].max()))
            Ethnicity = st.slider("Ethnicity", int(df["Ethnicity"].min()), int(df["Ethnicity"].max()))
            SocioeconomicStatus = st.slider("Socioeconomic Status", int(df["SocioeconomicStatus"].min()), int(df["SocioeconomicStatus"].max()))
            EducationLevel = st.slider("Education Level", int(df["EducationLevel"].min()), int(df["EducationLevel"].max()))
            BMI = st.slider("BMI", float(df["BMI"].min()), float(df["BMI"].max()))
            Smoking = st.slider("Smoking", int(df["Smoking"].min()), int(df["Smoking"].max()))
            AlcoholConsumption = st.slider("Alcohol Consumption", int(df["AlcoholConsumption"].min()), int(df["AlcoholConsumption"].max()))
            PhysicalActivity = st.slider("Physical Activity", int(df["PhysicalActivity"].min()), int(df["PhysicalActivity"].max()))
            DietQuality = st.slider("Diet Quality", int(df["DietQuality"].min()), int(df["DietQuality"].max()))
            SleepQuality = st.slider("Sleep Quality", int(df["SleepQuality"].min()), int(df["SleepQuality"].max()))
            FamilyHistoryKidneyDisease = st.slider("Family History Kidney Disease", int(df["FamilyHistoryKidneyDisease"].min()), int(df["FamilyHistoryKidneyDisease"].max()))
            FamilyHistoryHypertension = st.slider("Family History Hypertension", int(df["FamilyHistoryHypertension"].min()), int(df["FamilyHistoryHypertension"].max()))
            FamilyHistoryDiabetes = st.slider("Family History Diabetes", int(df["FamilyHistoryDiabetes"].min()), int(df["FamilyHistoryDiabetes"].max()))
            PreviousAcuteKidneyInjury = st.slider("Previous Acute Kidney Injury", int(df["PreviousAcuteKidneyInjury"].min()), int(df["PreviousAcuteKidneyInjury"].max()))
            UrinaryTractInfections = st.slider("Urinary Tract Infections", int(df["UrinaryTractInfections"].min()), int(df["UrinaryTractInfections"].max()))
            SystolicBP = st.slider("Systolic BP", int(df["SystolicBP"].min()), int(df["SystolicBP"].max()))

        with col2:
            DiastolicBP = st.slider("Diastolic BP", int(df["DiastolicBP"].min()), int(df["DiastolicBP"].max()))
            FastingBloodSugar = st.slider("Fasting Blood Sugar", float(df["FastingBloodSugar"].min()), float(df["FastingBloodSugar"].max()))
            HbA1c = st.slider("HbA1c", float(df["HbA1c"].min()), float(df["HbA1c"].max()))
            SerumCreatinine = st.slider("Serum Creatinine", float(df["SerumCreatinine"].min()), float(df["SerumCreatinine"].max()))
            BUNLevels = st.slider("BUN Levels", float(df["BUNLevels"].min()), float(df["BUNLevels"].max()))
            GFR = st.slider("GFR", float(df["GFR"].min()), float(df["GFR"].max()))
            ProteinInUrine = st.slider("Protein in Urine", float(df["ProteinInUrine"].min()), float(df["ProteinInUrine"].max()))
            ACR = st.slider("ACR", float(df["ACR"].min()), float(df["ACR"].max()))
            SerumElectrolytesSodium = st.slider("Serum Electrolytes Sodium", float(df["SerumElectrolytesSodium"].min()), float(df["SerumElectrolytesSodium"].max()))
            SerumElectrolytesPotassium = st.slider("Serum Electrolytes Potassium", float(df["SerumElectrolytesPotassium"].min()), float(df["SerumElectrolytesPotassium"].max()))
            SerumElectrolytesCalcium = st.slider("Serum Electrolytes Calcium", float(df["SerumElectrolytesCalcium"].min()), float(df["SerumElectrolytesCalcium"].max()))
            SerumElectrolytesPhosphorus = st.slider("Serum Electrolytes Phosphorus", float(df["SerumElectrolytesPhosphorus"].min()), float(df["SerumElectrolytesPhosphorus"].max()))
            HemoglobinLevels = st.slider("Hemoglobin Levels", float(df["HemoglobinLevels"].min()), float(df["HemoglobinLevels"].max()))
            CholesterolTotal = st.slider("Cholesterol Total", float(df["CholesterolTotal"].min()), float(df["CholesterolTotal"].max()))
            CholesterolLDL = st.slider("Cholesterol LDL", float(df["CholesterolLDL"].min()), float(df["CholesterolLDL"].max()))
            CholesterolHDL = st.slider("Cholesterol HDL", float(df["CholesterolHDL"].min()), float(df["CholesterolHDL"].max()))
            CholesterolTriglycerides = st.slider("Cholesterol Triglycerides", float(df["CholesterolTriglycerides"].min()), float(df["CholesterolTriglycerides"].max()))

        with col3:
            ACEInhibitors = st.slider("ACE Inhibitors", int(df["ACEInhibitors"].min()), int(df["ACEInhibitors"].max()))
            Diuretics = st.slider("Diuretics", int(df["Diuretics"].min()), int(df["Diuretics"].max()))
            NSAIDsUse = st.slider("NSAIDs Use", int(df["NSAIDsUse"].min()), int(df["NSAIDsUse"].max()))
            Statins = st.slider("Statins", int(df["Statins"].min()), int(df["Statins"].max()))
            AntidiabeticMedications = st.slider("Antidiabetic Medications", int(df["AntidiabeticMedications"].min()), int(df["AntidiabeticMedications"].max()))
            Edema = st.slider("Edema", int(df["Edema"].min()), int(df["Edema"].max()))
            FatigueLevels = st.slider("Fatigue Levels", int(df["FatigueLevels"].min()), int(df["FatigueLevels"].max()))
            NauseaVomiting = st.slider("Nausea Vomiting", int(df["NauseaVomiting"].min()), int(df["NauseaVomiting"].max()))
            MuscleCramps = st.slider("Muscle Cramps", int(df["MuscleCramps"].min()), int(df["MuscleCramps"].max()))
            Itching = st.slider("Itching", int(df["Itching"].min()), int(df["Itching"].max()))
            QualityOfLifeScore = st.slider("Quality of Life Score", int(df["QualityOfLifeScore"].min()), int(df["QualityOfLifeScore"].max()))
            HeavyMetalsExposure = st.slider("Heavy Metals Exposure", int(df["HeavyMetalsExposure"].min()), int(df["HeavyMetalsExposure"].max()))
            OccupationalExposureChemicals = st.slider("Occupational Exposure to Chemicals", int(df["OccupationalExposureChemicals"].min()), int(df["OccupationalExposureChemicals"].max()))
            WaterQuality = st.slider("Water Quality", int(df["WaterQuality"].min()), int(df["WaterQuality"].max()))
            MedicalCheckupsFrequency = st.slider("Medical Checkups Frequency", int(df["MedicalCheckupsFrequency"].min()), int(df["MedicalCheckupsFrequency"].max()))
            MedicationAdherence = st.slider("Medication Adherence", int(df["MedicationAdherence"].min()), int(df["MedicationAdherence"].max()))
            HealthLiteracy = st.slider("Health Literacy", int(df["HealthLiteracy"].min()), int(df["HealthLiteracy"].max()))


                # Create a list to store all the features
        features = [Age,Gender,Ethnicity,SocioeconomicStatus,EducationLevel,BMI,Smoking,AlcoholConsumption,PhysicalActivity,DietQuality,SleepQuality,FamilyHistoryKidneyDisease,FamilyHistoryHypertension,FamilyHistoryDiabetes,PreviousAcuteKidneyInjury,UrinaryTractInfections,SystolicBP,DiastolicBP,FastingBloodSugar,HbA1c,SerumCreatinine,BUNLevels,GFR,ProteinInUrine,ACR,SerumElectrolytesSodium,SerumElectrolytesPotassium,SerumElectrolytesCalcium,SerumElectrolytesPhosphorus,HemoglobinLevels,CholesterolTotal,CholesterolLDL,CholesterolHDL,CholesterolTriglycerides,ACEInhibitors,Diuretics,NSAIDsUse,Statins,AntidiabeticMedications,Edema,FatigueLevels,NauseaVomiting,MuscleCramps,Itching,QualityOfLifeScore,HeavyMetalsExposure,OccupationalExposureChemicals,WaterQuality,MedicalCheckupsFrequency,MedicationAdherence,HealthLiteracy]

        # Create a DataFrame to store slider values
        slider_values = {
            "Feature": ["Age","Gender","Ethnicity","SocioeconomicStatus","EducationLevel","BMI","Smoking","AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality","FamilyHistoryKidneyDisease","FamilyHistoryHypertension","FamilyHistoryDiabetes","PreviousAcuteKidneyInjury","UrinaryTractInfections","SystolicBP","DiastolicBP","FastingBloodSugar","HbA1c","SerumCreatinine","BUNLevels","GFR","ProteinInUrine","ACR","SerumElectrolytesSodium","SerumElectrolytesPotassium","SerumElectrolytesCalcium","SerumElectrolytesPhosphorus","HemoglobinLevels","CholesterolTotal","CholesterolLDL","CholesterolHDL","CholesterolTriglycerides","ACEInhibitors","Diuretics","NSAIDsUse","Statins","AntidiabeticMedications","Edema","FatigueLevels","NauseaVomiting","MuscleCramps","Itching","QualityOfLifeScore","HeavyMetalsExposure","OccupationalExposureChemicals","WaterQuality","MedicalCheckupsFrequency","MedicationAdherence","HealthLiteracy"],
            "Value": [Age,Gender,Ethnicity,SocioeconomicStatus,EducationLevel,BMI,Smoking,AlcoholConsumption,PhysicalActivity,DietQuality,SleepQuality,FamilyHistoryKidneyDisease,FamilyHistoryHypertension,FamilyHistoryDiabetes,PreviousAcuteKidneyInjury,UrinaryTractInfections,SystolicBP,DiastolicBP,FastingBloodSugar,HbA1c,SerumCreatinine,BUNLevels,GFR,ProteinInUrine,ACR,SerumElectrolytesSodium,SerumElectrolytesPotassium,SerumElectrolytesCalcium,SerumElectrolytesPhosphorus,HemoglobinLevels,CholesterolTotal,CholesterolLDL,CholesterolHDL,CholesterolTriglycerides,ACEInhibitors,Diuretics,NSAIDsUse,Statins,AntidiabeticMedications,Edema,FatigueLevels,NauseaVomiting,MuscleCramps,Itching,QualityOfLifeScore,HeavyMetalsExposure,OccupationalExposureChemicals,WaterQuality,MedicalCheckupsFrequency,MedicationAdherence,HealthLiteracy]
        }
        slider_df = pd.DataFrame(slider_values)

        # Create a button to predict
        if st.button("Predict"):
            # Get prediction and model score
            prediction, score = predict(X, y, features)
            score = score  # Correction factor
           
            # Store prediction result
            prediction_result = ""
            
            # Print the output according to the prediction
            if prediction == 1:
                prediction_result = "The person has kidney problems"
                st.error(prediction_result)
            
            else:
                prediction_result = "The person has low risk of kidney problems"
                st.success(prediction_result)

            # Print the score of the model
            model_accuracy = f"The model used is trusted by doctors and has an accuracy of {round((score * 100), 2)}%"
            st.sidebar.write(model_accuracy)

            # Store these in session state for PDF generation
            st.session_state['prediction_result'] = prediction_result
            st.session_state['model_accuracy'] = model_accuracy

        # Display the slider values in a table
        st.subheader("Selected Values:")
        st.table(slider_df)

        # Download section
        st.subheader("Download Test Report")
        user_name = st.text_input("Enter your name (required for download):")

        if user_name:
            col1, col2 = st.columns(2)

            # PDF Download Button
            with col1:
                try:
                    # Generate PDF
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Add title
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(200, 10, txt="Renal Risk Assessment Report", ln=True, align='C')
                    pdf.ln(10)

                    # Add user name and timestamp
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"User Name: {user_name}", ln=True)
                    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                    pdf.ln(10)

                    # Add prediction result if available
                    if 'prediction_result' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Prediction Result:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('prediction_result', ''), ln=True)
                        pdf.ln(5)

                    # Add model accuracy if available
                    if 'model_accuracy' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Model Accuracy:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('model_accuracy', ''), ln=True)
                        pdf.ln(10)

                    # Add the measurements table
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, txt="Measurements:", ln=True)
                    pdf.set_font("Arial", size=12)
                    
                    # Create the data table
                    for index, row in slider_df.iterrows():
                        pdf.cell(100, 10, txt=f"{row['Feature']}:", ln=False)
                        pdf.cell(100, 10, txt=f"{str(row['Value'])}", ln=True)

                    # Create a temporary file path
                    temp_file = f"temp_{user_name}_report.pdf"
                    
                    # Save PDF to a temporary file
                    pdf.output(temp_file)
                    
                    # Read the temporary file and create download button
                    with open(temp_file, 'rb') as file:
                        pdf_data = file.read()
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"{user_name}_kidney_report.pdf",
                            mime="application/pdf",
                        )
                    
                    # Import os and remove the temporary file
                    
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

                except Exception as e:
                    pass
                try:
                    # Generate PDF
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Add title
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(200, 10, txt="Renal Risk Assessment Report", ln=True, align='C')
                    pdf.ln(10)

                    # Add user name and timestamp
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"User Name: {user_name}", ln=True)
                    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                    pdf.ln(10)

                    # Add prediction result if available
                    if 'prediction_result' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Prediction Result:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('prediction_result', ''), ln=True)
                        pdf.ln(5)

                    # Add model accuracy if available
                    if 'model_accuracy' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Model Accuracy:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('model_accuracy', ''), ln=True)
                        pdf.ln(10)

                    # Add the measurements table
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, txt="Measurements:", ln=True)
                    pdf.set_font("Arial", size=12)
                    
                    # Create the data table
                    for index, row in slider_df.iterrows():
                        pdf.cell(100, 10, txt=f"{row['Feature']}:", ln=False)
                        pdf.cell(100, 10, txt=f"{str(row['Value'])}", ln=True)

                    # Save to bytes
                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    pdf_bytes = pdf_output.getvalue()
                    
                    # Create download button
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{user_name}_kidney_report.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.success("Your report is generated")

            # CSV Download Button
            with col2:
                try:
                    # Convert DataFrame to CSV
                    csv_buffer = io.StringIO()
                    slider_df.to_csv(csv_buffer, index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV Data",
                        data=csv_buffer.getvalue(),
                        file_name=f"{user_name}_kidney_data.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")
        else:
            st.info("Please enter your name to enable downloads.")


    with tab2:
        
            def get_gemini_medication_recommendation(disease_type, patient_data):
                prompt = f"""
                You are a medical expert. Based on the following disease diagnosis, suggest the appropriate medications, their dosage, and additional lifestyle recommendations:
                
                **Disease Type**: {disease_type}
                
                **Patient Data**:
                {patient_data}
                
                Provide a clear and structured recommendation including:
                - Medication name
                - Recommended dosage
                - Special precautions
                - Any additional lifestyle suggestions
                """
                
                model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Using Gemini Pro for text-based generation
                response = model.generate_content(prompt)
                
                return response.text

            # Streamlit UI
            st.title("Medication Recommendations")
            st.markdown(
                """
                    <p style="font-size:25px">
                        Upload your patient data to get medication recommendations.
                    </p>
                """, unsafe_allow_html=True
            )

            # File uploader for CSV files
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    df_original = pd.read_csv(uploaded_file)

                    # Display original data
                    st.subheader("Original Data:")
                    st.dataframe(df_original)

                    if df_original.shape[1] < 2:
                        st.error("CSV file must have at least two columns: parameters and values")
                        st.stop()

                    # Convert the uploaded data into a structured format
                    df_processed = pd.DataFrame([
                        {param: value for param, value in zip(df_original.iloc[:, 0], df_original.iloc[:, 1])}
                    ])

                    # Display transformed data
                    st.subheader("Transformed Data:")
                    st.dataframe(df_processed)

                    # Required columns check
                    required_columns = [
                        "Age","Gender","Ethnicity","SocioeconomicStatus","EducationLevel","BMI","Smoking","AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality","FamilyHistoryKidneyDisease","FamilyHistoryHypertension","FamilyHistoryDiabetes","PreviousAcuteKidneyInjury","UrinaryTractInfections","SystolicBP","DiastolicBP","FastingBloodSugar","HbA1c","SerumCreatinine","BUNLevels","GFR","ProteinInUrine","ACR","SerumElectrolytesSodium","SerumElectrolytesPotassium","SerumElectrolytesCalcium","SerumElectrolytesPhosphorus","HemoglobinLevels","CholesterolTotal","CholesterolLDL","CholesterolHDL","CholesterolTriglycerides","ACEInhibitors","Diuretics","NSAIDsUse","Statins","AntidiabeticMedications","Edema","FatigueLevels","NauseaVomiting","MuscleCramps","Itching","QualityOfLifeScore","HeavyMetalsExposure","OccupationalExposureChemicals","WaterQuality","MedicalCheckupsFrequency","MedicationAdherence","HealthLiteracy"
                    ]
                    
                    missing_columns = [col for col in required_columns if col not in df_processed.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        st.write("Your CSV should have these parameters in the first column:")
                        st.write(required_columns)
                        st.stop()

                    try:
                        # Convert all columns to numeric
                        for col in required_columns:
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                        
                        

                        # Extract features for prediction
                        features = [
                            float(df_processed.iloc[0]['Age']),
                            int(df_processed.iloc[0]['Gender']),
                            str(df_processed.iloc[0]['Ethnicity']),
                            str(df_processed.iloc[0]['SocioeconomicStatus']),
                            str(df_processed.iloc[0]['EducationLevel']),
                            float(df_processed.iloc[0]['BMI']),
                            int(df_processed.iloc[0]['Smoking']),
                            int(df_processed.iloc[0]['AlcoholConsumption']),
                            int(df_processed.iloc[0]['PhysicalActivity']),
                            int(df_processed.iloc[0]['DietQuality']),
                            int(df_processed.iloc[0]['SleepQuality']),
                            int(df_processed.iloc[0]['FamilyHistoryKidneyDisease']),
                            int(df_processed.iloc[0]['FamilyHistoryHypertension']),
                            int(df_processed.iloc[0]['FamilyHistoryDiabetes']),
                            int(df_processed.iloc[0]['PreviousAcuteKidneyInjury']),
                            int(df_processed.iloc[0]['UrinaryTractInfections']),
                            float(df_processed.iloc[0]['SystolicBP']),
                            float(df_processed.iloc[0]['DiastolicBP']),
                            float(df_processed.iloc[0]['FastingBloodSugar']),
                            float(df_processed.iloc[0]['HbA1c']),
                            float(df_processed.iloc[0]['SerumCreatinine']),
                            float(df_processed.iloc[0]['BUNLevels']),
                            float(df_processed.iloc[0]['GFR']),
                            int(df_processed.iloc[0]['ProteinInUrine']),
                            float(df_processed.iloc[0]['ACR']),
                            float(df_processed.iloc[0]['SerumElectrolytesSodium']),
                            float(df_processed.iloc[0]['SerumElectrolytesPotassium']),
                            float(df_processed.iloc[0]['SerumElectrolytesCalcium']),
                            float(df_processed.iloc[0]['SerumElectrolytesPhosphorus']),
                            float(df_processed.iloc[0]['HemoglobinLevels']),
                            float(df_processed.iloc[0]['CholesterolTotal']),
                            float(df_processed.iloc[0]['CholesterolLDL']),
                            float(df_processed.iloc[0]['CholesterolHDL']),
                            float(df_processed.iloc[0]['CholesterolTriglycerides']),
                            int(df_processed.iloc[0]['ACEInhibitors']),
                            int(df_processed.iloc[0]['Diuretics']),
                            int(df_processed.iloc[0]['NSAIDsUse']),
                            int(df_processed.iloc[0]['Statins']),
                            int(df_processed.iloc[0]['AntidiabeticMedications']),
                            int(df_processed.iloc[0]['Edema']),
                            int(df_processed.iloc[0]['FatigueLevels']),
                            int(df_processed.iloc[0]['NauseaVomiting']),
                            int(df_processed.iloc[0]['MuscleCramps']),
                            int(df_processed.iloc[0]['Itching']),
                            float(df_processed.iloc[0]['QualityOfLifeScore']),
                            int(df_processed.iloc[0]['HeavyMetalsExposure']),
                            int(df_processed.iloc[0]['OccupationalExposureChemicals']),
                            int(df_processed.iloc[0]['WaterQuality']),
                            int(df_processed.iloc[0]['MedicalCheckupsFrequency']),
                            int(df_processed.iloc[0]['MedicationAdherence']),
                            int(df_processed.iloc[0]['HealthLiteracy'])

                                                    ]

                        # Make prediction
                        prediction, confidence = predict(X, y, features)

                        # Disease mapping
                        disease_type = ""
                        if prediction == 1:
                            disease_type = "High risk of kidney ailments"
                        
                        else:
                            disease_type = "Low risk of kidney ailment."

                        st.subheader("Patient Recommendation:")
                        
                        if disease_type != "No kidney problem detected":
                            st.warning(disease_type)
                            patient_data = df_processed.iloc[0].to_dict()
                            
                            # Call Gemini to generate medication recommendations
                            medication_info = get_gemini_medication_recommendation(disease_type, patient_data)

                            st.info("AI Recommended Medication:")
                            st.write(medication_info)
                        else:
                            st.success("No kidney problem detected")
                            st.info("Maintain a healthy lifestyle.")
                        confidence = confidence*100
                        st.write(f"Prediction confidence: {confidence:.2f}%")

                    except Exception as e:
                        st.error(f"Error processing the data: {str(e)}")
                        st.write("Please ensure all values are numeric and properly formatted.")

                except Exception as e:
                    st.error(f"Error reading the file: {str(e)}")


                    

                       
    # Second Tab: Data Source Page
    with tab3:
        st.title("Data Info Page")
        st.subheader("View Data")

        # Create an expansion option to check the data
        with st.expander("View data"):
            st.dataframe(df)

        # Create a section for columns description
        st.subheader("Columns Description:")

             # Create multiple checkboxes in a row
        col_name, summary, col_data = st.columns(3)

        # Show name of all columns
        with col_name:
            if st.checkbox("Column Names"):
                st.dataframe(df.columns)

        # Show datatype of all columns
        with summary:
            if st.checkbox("View Summary"):
                st.dataframe(df.describe())

        # Show data for each column
        with col_data:
            if st.checkbox("Columns Data"):
                col = st.selectbox("Column Name", list(df.columns))
                st.dataframe(df[col])

        # Add the link to the dataset
        st.link_button("View Data Set", "https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis")
        
