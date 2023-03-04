import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))


def text_transform(text):
    # Lower case
    text = text.lower()
    # Tokenization
    text = nltk.word_tokenize(text)
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


st.title("Email/SMS SPAM CLASSIFIER")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1 preprocess
    tranformed_sms = text_transform(input_sms)
    # 2 vectorize
    vector_input = tfidf.transform([tranformed_sms])
    # 3 predict
    result = model.predict(vector_input)
    # 4 display
    if result == 1:
        st.header("Spam message")
    else:
        st.header("Not spam message")



# Winning an unexpected prize sounds great, in theory. However, being notified of winning a contest you didn’t enter is a dead giveaway of a phishing text. If you’re unsure whether an offer is authentic, contact the business directly to verify.
#Notifications involving money owed to you are enticing, aren’t they? “Our records show you overpaid for (a product or service). Kindly supply your bank routing and account number to receive your refund.” Don’t fall for it.
#Government agencies like the IRS will not contact you via email, phone or text message. If any legitimate government agency needs to contact you, they will usually do so via mail or certified letter.
#free free free 100% free
#i like your video