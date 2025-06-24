import streamlit as st
import joblib



model=joblib.load('model.pkl')
vectorizer=joblib.load('vectorizer.pkl')
label_encoder=joblib.load('label_encoder.pkl')

st.title('movie genre prediction')
plot=st.text_area("enter your movie plot")
if st.button("Predict Genre"):
    if plot.strip():
        X = vectorizer.transform([plot])
        pred = model.predict(X)
        genre = label_encoder.inverse_transform(pred)[0]
        st.success(f"Predicted Genre: **{genre}**")
    else:
        st.warning("Please enter a plot.")
    




