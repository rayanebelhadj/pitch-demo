import os
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from apikey import apikey

os.environ["OPENAI_API_KEY"] = apikey

# Charger le modèle OpenAI
llm = OpenAI()

st.title("Analyseur de Réponses aux Sondages")
st.write("Téléchargez les réponses du sondage pour analyser les sentiments, extraire les thèmes, et obtenir des recommandations.")

# Téléchargement du fichier
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Lire le fichier CSV téléchargé
    survey_data = pd.read_csv(uploaded_file)
    st.write("Réponses du Sondage :")
    st.write(survey_data)

    # Définir les modèles de prompts
    sentiment_template = PromptTemplate(
        input_variables=["response"],
        template="Analysez le sentiment de la réponse suivante : {response}"
    )

    theme_template = PromptTemplate(
        input_variables=["response"],
        template="Identifiez les thèmes principaux dans la réponse suivante : {response}"
    )

    recommendation_template = PromptTemplate(
        input_variables=["response"],
        template="Fournissez des recommandations basées sur la réponse suivante : {response}"
    )

    # Définir les chaînes LLM
    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_template)
    theme_chain = LLMChain(llm=llm, prompt=theme_template)
    recommendation_chain = LLMChain(llm=llm, prompt=recommendation_template)

    # Générer l'analyse
    st.write("Résultats de l'Analyse :")

    results = []

    for index, row in survey_data.iterrows():
        respondent_id = row["IDRépondant"]
        report = {"IDRépondant": respondent_id}

        for question in survey_data.columns[1:]:
            response = row[question]

            sentiment = sentiment_chain.run({"response": response})
            theme = theme_chain.run({"response": response})
            recommendation = recommendation_chain.run({"response": response})

            report[f"{question}_Sentiment"] = sentiment
            report[f"{question}_Thèmes"] = theme
            report[f"{question}_Recommandation"] = recommendation

            st.write(f"**ID Répondant :** {respondent_id}")
            st.write(f"**Question :** {question}")
            st.write(f"**Réponse :** {response}")
            st.write(f"**Sentiment :** {sentiment}")
            st.write(f"**Thèmes :** {theme}")
            st.write(f"**Recommandation :** {recommendation}")
            st.write("---")

        results.append(report)

    # Enregistrer éventuellement les résultats dans un nouveau fichier CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv("resultats_analyse_sondage.csv", index=False)
    st.write("Les résultats de l'analyse ont été enregistrés dans 'resultats_analyse_sondage.csv'.")
