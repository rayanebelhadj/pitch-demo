import os
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Assurez-vous que la clé API est disponible dans l'environnement
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    st.error("L'API Key OpenAI n'est pas définie. Assurez-vous de la définir dans les variables d'environnement.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# Charger le modèle OpenAI
llm = OpenAI()

st.title("Analyseur de Réponses aux Sondages")
st.write("Téléchargez les réponses du sondage pour analyser les sentiments et obtenir des recommandations.")

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
        template="Analysez le sentiment des réponses suivantes : {response}"
    )

    recommendation_template = PromptTemplate(
        input_variables=["response"],
        template="Fournis des recommandations d'amélioration pour le propriétaire de l'entreprise selon les commentaires de ce client : {response}"
    )

    # Définir les chaînes LLM
    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_template)
    recommendation_chain = LLMChain(llm=llm, prompt=recommendation_template)

    # Fonction pour traiter les réponses d'un client
    def analyze_responses(respondent_id, combined_responses):
        sentiment = sentiment_chain.run({"response": combined_responses})
        recommendation = recommendation_chain.run({"response": combined_responses})

        return {
            "IDRépondant": respondent_id,
            "Réponses": combined_responses,
            "Sentiment": sentiment,
            "Recommandation": recommendation
        }

    # Combiner les réponses de chaque client
    combined_data = survey_data.groupby("IDRépondant").agg(lambda x: ' '.join(x.astype(str)))

    # Générer l'analyse
    st.write("Résultats de l'Analyse :")
    results = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for respondent_id, row in combined_data.iterrows():
            combined_responses = ' '.join(row[1:])
            futures.append(executor.submit(analyze_responses, respondent_id, combined_responses))

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            st.write(f"**ID Répondant :** {result['IDRépondant']}")
            st.write(f"**Réponses :** {result['Réponses']}")
            st.write(f"**Sentiment :** {result['Sentiment']}")
            st.write(f"**Recommandation :** {result['Recommandation']}")
            st.write("---")

    # Analyse globale
    all_sentiments = []
    all_recommendations = []

    for result in results:
        all_sentiments.append(result['Sentiment'])
        all_recommendations.extend(result['Recommandation'].split(", "))  # Assuming recommendations are comma-separated

    global_sentiments = dict(Counter(all_sentiments))
    global_recommendations = dict(Counter(all_recommendations))

    st.write("## Analyse Globale :")
    
    st.write("### Sentiments :")
    for sentiment, count in global_sentiments.items():
        st.write(f"{sentiment}")

    st.write("### Recommandations :")
    for recommendation, count in global_recommendations.items():
        st.write(f"{recommendation}")

    # Enregistrer éventuellement les résultats dans un nouveau fichier CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv("resultats_analyse_sondage.csv", index=False)
