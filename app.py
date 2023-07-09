import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import nltk
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

vectorizer = pickle.load(open('cvVectorizer (2).pkl', 'rb'))
model = pickle.load(open('mnbModel (1).pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    filtered_words = [word for word in words if word.isalnum(
    ) and word not in stopwords.words('english') and word not in string.punctuation]

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return " ".join(stemmed_words)

  # Menu items
selected = option_menu(
    menu_title=None,
    options=['Home', 'Email Examples'],
    icons=['house', 'info', 'book'],
    default_index=0,
    orientation="horizontal",
    styles={
        "nav-link": {
            'background-color': 'blue'
        },
        "nav-link-selected": {"background-color": '#0390fc'}
    }
)

st.sidebar.image('spam.jpg', caption='Spam', width=150)
st.sidebar.markdown("""
            ### Building a Secure Big Data Environment to Detect Collaborative Spam
            ***
            Detect Email and SMS spam messages with this app in seconds!
            ***

            Although this system is built specifically for email spam detection, it can also detect spam SMS.
            

            The performance of the model is 97%.
    """)

if selected == 'Home':
    st.title('Collaborative Email Spam Detection System')
    st.subheader('Machine Learning Approach')

    email = st.text_area('Enter email message')

    if st.button('Detect'):

        # 1.preprocess
        transformed_email = transform_text(email)
        # 2 vectorize
        vectorized_email = vectorizer.transform([transformed_email])
        # 3. predict
        prediction = model.predict_proba(vectorized_email)[0][1] * 100
        # result = model.predict(vectorized_email)

        # 4. Display
        if prediction <= 50:
            st.markdown(
                f"#### This email is {prediction: .2f}% Not Spam.")
        elif prediction > 50:
            st.markdown(f"#### This email is a Spam email.")

        st.markdown("------------------------------------------")
        st.markdown(
            f"""
            ##### CONFIDENCE SCORE: {prediction: .2f}%
            """
        )

        st.markdown("------------------------------------------")

        st.markdown("##### YOUR INPUT:")
        st.markdown(f"```{email}````")


if selected == 'Email Examples':
    st.markdown("### Spam and Ham Email Exmaples")
    st.markdown("""
   |S/N|SPAM Exmples                                                          | NON-SPAM/HAM Examples|
   |--|----------------------------------------------------------------------|----------------------------|
   |1 | A family member needs help: URGENT Your grandson was arrested last night in Mexico. Need bail money immediately Western Union Wire $9,500: [link].| Dear Dan, Thank you for contacting Unity Bank. We sincerely apologize for the inconvenience experienced as a result of this. Kindly hold on for a possible auto reversal period of 24-hours however, if the fund is not reversed within the given period, please revert to enable us to address the issue.|
   |2 | Hi Dan, Your passport to the India's most futuristics AI Conference is within reach - and it's never been more affordable! But remember, every great deal has an expiration date. Our offer of up to a 35% discount winds up on June 30th. Don't miss this FINAL CALL to seize this outstanding opportunity and leap into the AI future at a bargain! Claim your offer |  Hello 👋🏽 I hope I caught you at a good time. I wanted to pop back into your inbox and let you know that you've been selected to participate in the second cohort of the DXMentorship program. I’m excited for you and look forward to learning amazing things together! As a next step, I’d like you to Join the Mentorship Community on Discord. Introduce yourself, and get to know your fellow mentees. |                                                                   
                                                                                              
    """)
