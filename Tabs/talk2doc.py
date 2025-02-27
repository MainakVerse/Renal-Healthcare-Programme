import streamlit as st
import google.generativeai as genai

# Load API Key from Streamlit secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


if not GEMINI_API_KEY:
    st.error("API key is missing! Add it to Streamlit secrets.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Function to ask Gemini AI about diabetes
def ask_gemini(query):
    prompt = f"""
    You are a medical chatbot specialized in kidney disease detection, disease analysis and its health implications. 
    Answer only kidney-related queries with medically accurate information. 
    If a question is unrelated to kidney diseases or kidney health, politely inform the user that you can only answer kidney or renal diseases-related questions.

    **User's Question:** {query}

    Provide a clear, concise, and accurate medical response.
    """

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    
    return response.text

# Streamlit UI
def app():
    st.title("🩺 Renal Medical Chatbot")
    st.image('./images/capsule.png')
    st.success("Please ask your queries related to kidney health and its health implications.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_query = st.text_input("Ask your question about kidney diseases:")

    if st.button("Get Answer"):
        if user_query:
            response = ask_gemini(user_query)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Chatbot", response))

    # Display chat history
    st.subheader("Chat History:")
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑‍⚕️ {role}:** {message}")
        else:
            st.markdown(f"**🤖 {role}:** {message}")

# Run the chatbot
if __name__ == "__main__":
    app()
