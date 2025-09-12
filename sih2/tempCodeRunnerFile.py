import streamlit as st
import re
import random
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
# Suppress deprecation warnings for scikit-learn
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------ Global Dictionaries & Data Loading (Cached for Performance) ------------------

# Use st.cache_data to cache the DataFrame loading, so it only runs once per session.
@st.cache_data
def load_data():
    """Loads and preprocesses the training and testing data."""
    try:
        training = pd.read_csv('Data/Training.csv')
        testing = pd.read_csv('Data/Testing.csv')

        # Clean duplicate column names
        training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
        testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
        training = training.loc[:, ~training.columns.duplicated()]
        testing = testing.loc[:, ~testing.columns.duplicated()]

        return training, testing
    except FileNotFoundError:
        st.error("Error: CSV files not found. Please make sure 'Data/Training.csv' and 'Data/Testing.csv' are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

# Use st.cache_data to cache the dictionary loading.
@st.cache_data
def load_dictionaries():
    """Loads all supplementary data dictionaries."""
    severity_dict = {}
    description_list = {}
    precaution_dict = {}

    # Load symptom descriptions
    try:
        with open('MasterData/symptom_Description.csv') as csv_file:
            for row in csv.reader(csv_file):
                if len(row) > 1:
                    description_list[row[0]] = row[1]
    except FileNotFoundError:
        st.warning("Warning: symptom_Description.csv not found.")

    # Load symptom severity
    try:
        with open('MasterData/symptom_severity.csv') as csv_file:
            for row in csv.reader(csv_file):
                if len(row) > 1:
                    try:
                        severity_dict[row[0]] = int(row[1])
                    except ValueError:
                        pass
    except FileNotFoundError:
        st.warning("Warning: symptom_severity.csv not found.")
        
    # Load symptom precautions
    try:
        with open('MasterData/symptom_precaution.csv') as csv_file:
            for row in csv.reader(csv_file):
                if len(row) > 4:
                    precaution_dict[row[0]] = [row[1], row[2], row[3], row[4]]
    except FileNotFoundError:
        st.warning("Warning: symptom_precaution.csv not found.")
    
    return severity_dict, description_list, precaution_dict

# Load data and dictionaries
training, testing = load_data()
severityDictionary, description_list, precautionDictionary = load_dictionaries()

# Features and labels
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encode target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Use st.cache_resource to cache the trained model.
@st.cache_resource
def train_model(x_train, y_train):
    """Trains the Random Forest model."""
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    return model

model = train_model(x_train, y_train)
symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}

# ------------------ Symptom Synonyms & Translation ------------------
symptom_synonyms = {
    "stomach ache": "stomach_pain", "belly pain": "stomach_pain", "tummy pain": "stomach_pain",
    "abdominal pain": "stomach_pain", "belly ache": "stomach_pain", "gastric pain": "stomach_pain",
    "body ache": "muscle_pain", "muscle ache": "muscle_pain", "head ache": "headache",
    "head pain": "headache", "migraine": "headache", "chest pain": "chest_pain",
    "feaver": "fever", "loose motion": "diarrhea", "motions": "diarrhea", "khansi": "cough",
    "throat pain": "sore_throat", "runny nose": "chills", "sneezing": "chills",
    "shortness of breath": "breathlessness", "skin rash": "skin_rash", "itchy": "itching",
    "tiredness": "fatigue", "vomiting": "vomit", "nausea": "nausea", "dizzy": "dizziness",
    "sad": "depression", "anxiety": "anxiety",
}

async def translate_to_english(text):
    """Translates text to English using Gemini API."""
    if not text:
        return ""
    
    # Construct the payload for the Gemini API call
    payload = {
        "contents": [{"parts": [{"text": f"Translate the following text to English and provide only the translated text in a JSON format with the key 'translated_text': '{text}'"}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "translated_text": {"type": "STRING"}
                }
            }
        },
    }

    # API key is provided by the environment
    apiKey = "" 
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    try:
        response = await fetch(apiUrl, {
            "method": 'POST',
            "headers": {'Content-Type': 'application/json'},
            "body": JSON.stringify(payload)
        })

        if response.ok:
            result = await response.json()
            json_text = result.candidates[0].content.parts[0].text
            parsed_json = JSON.parse(json_text)
            return parsed_json.translated_text
        else:
            return text # Fallback to original text if API call fails
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text # Fallback

def extract_symptoms(user_input, all_symptoms):
    """Extracts symptoms from user input using synonyms, exact, and fuzzy matching."""
    extracted = []
    text = user_input.lower().replace("-", " ")

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(
            word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8
        )
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

def predict_disease(symptoms_list):
    """Predicts a disease based on a list of symptoms."""
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    
    input_df = pd.DataFrame([input_vector], columns=symptoms_dict.keys())
    pred_proba = model.predict_proba(input_df)[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)
    return disease, confidence, pred_proba

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="HealthCare Chatbot", page_icon="🩺")

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "home"
    st.session_state.name = ""
    st.session_state.symptoms_list = []
    st.session_state.initial_prediction = None
    st.session_state.final_prediction = None
    st.session_state.guided_symptoms = []

st.title("🩺 HealthCare Chatbot")
st.markdown("Hello! I am a chatbot designed to help you with preliminary symptom analysis. Please answer a few questions so I can understand your condition better.")

# --- Home Page: User Information and Initial Symptoms ---
if st.session_state.page == "home":
    with st.form(key="user_info_form"):
        st.session_state.name = st.text_input("What is your name? :")
        age = st.text_input("Please enter your age: ")
        gender = st.selectbox("What is your gender?", ["Male", "Female", "Other"])

        symptoms_input_raw = st.text_area("Describe your symptoms (any language):", height=100)
        
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if not st.session_state.name or not symptoms_input_raw:
            st.warning("Please enter your name and symptoms to continue.")
        else:
            with st.spinner('Thinking...'):
                symptoms_input_en = st.experimental_singleton(translate_to_english)(symptoms_input_raw)
                detected_symptoms = extract_symptoms(symptoms_input_en, cols)

                if not detected_symptoms:
                    st.error("❌ Sorry, I could not detect valid symptoms. Please try again with more details.")
                else:
                    st.session_state.symptoms_list = detected_symptoms
                    st.success(f"✅ Detected symptoms: {', '.join(st.session_state.symptoms_list).replace('_', ' ')}")
                    
                    # Initial prediction
                    initial_disease, confidence, _ = predict_disease(st.session_state.symptoms_list)
                    st.session_state.initial_prediction = {"disease": initial_disease, "confidence": confidence}

                    # Set up guided questions for the next page
                    disease_symptoms = list(training[training['prognosis'] == initial_disease].iloc[0][:-1].index[
                        training[training['prognosis'] == initial_disease].iloc[0][:-1] == 1])
                    
                    st.session_state.guided_symptoms = [
                        sym for sym in disease_symptoms if sym not in st.session_state.symptoms_list
                    ][:8] # Limit to 8 questions
                    
                    st.session_state.page = "guided_questions"
                    st.experimental_rerun()

# --- Guided Questions Page ---
elif st.session_state.page == "guided_questions":
    st.header("🤔 Guided Questions")
    initial_pred = st.session_state.initial_prediction
    st.info(f"Based on your initial symptoms, you may have **{initial_pred['disease']}** (Confidence: {initial_pred['confidence']}%).")
    st.write("To get a more accurate diagnosis, please answer a few more questions related to this condition.")

    if st.session_state.guided_symptoms:
        with st.form(key="guided_questions_form"):
            new_symptoms = []
            for symptom in st.session_state.guided_symptoms:
                if st.checkbox(f"Do you also have **{symptom.replace('_', ' ')}**?", key=symptom):
                    new_symptoms.append(symptom)
            
            submit_guided = st.form_submit_button("Get Final Prediction")

        if submit_guided:
            with st.spinner('Calculating final diagnosis...'):
                st.session_state.symptoms_list.extend(new_symptoms)
                final_disease, final_confidence, _ = predict_disease(st.session_state.symptoms_list)
                st.session_state.final_prediction = {"disease": final_disease, "confidence": final_confidence}
                st.session_state.page = "result"
                st.experimental_rerun()
    else:
        st.info("No further questions to ask. Click below for your final diagnosis.")
        if st.button("Get Final Prediction"):
            final_disease, final_confidence, _ = predict_disease(st.session_state.symptoms_list)
            st.session_state.final_prediction = {"disease": final_disease, "confidence": final_confidence}
            st.session_state.page = "result"
            st.experimental_rerun()

# --- Result Page ---
elif st.session_state.page == "result":
    st.header("✨ Diagnosis Result")
    
    final_pred = st.session_state.final_prediction
    disease = final_pred["disease"]
    confidence = final_pred["confidence"]
    
    st.subheader(f"🩺 Based on your answers, you may have **{disease}**")
    st.metric(label="Confidence Level", value=f"{confidence}%")
    
    st.markdown("---")

    # Display description
    st.subheader("📖 About")
    description = description_list.get(disease, 'No description available.')
    st.write(description)

    # Display precautions
    if disease in precautionDictionary:
        st.subheader("🛡️ Suggested Precautions")
        precautions = precautionDictionary[disease]
        for i, prec in enumerate(precautions, 1):
            st.write(f"{i}. {prec}")

    st.markdown("---")

    st.info("💡 " + random.choice([
        "🌸 Health is wealth, take care of yourself.",
        "💪 A healthy outside starts from the inside.",
        "☀️ Every day is a chance to get stronger and healthier.",
        "🌿 Take a deep breath, your health matters the most.",
        "🌺 Remember, self-care is not selfish.",
    ]))
    
    st.markdown(f"\nThank you for using the chatbot. Wishing you good health, **{st.session_state.name}**!")
    
    if st.button("Start Over"):
        st.session_state.clear()
        st.experimental_rerun()
