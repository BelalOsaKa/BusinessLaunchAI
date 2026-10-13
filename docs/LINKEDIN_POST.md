# LinkedIn post draft (each member posts their own version)

🍽️ **Would you open a restaurant here? We built an AI tool to answer that.**

For our NTI Machine Learning & Data Analysis capstone, our team built **BusinessLaunch AI**. It evaluates any location in Philadelphia for a new restaurant or café using the Yelp Open Dataset (14,570 businesses, 700k reviews, 1.29M check-ins).

What it does:
📍 **Competition & popularity:** haversine-based competitor features and a gradient-boosting model (test R² 0.61 on a spatial hold-out)
💬 **Customer expectations:** TF-IDF + Logistic Regression sentiment (macro F1 0.944) and NMF topics that surface what customers complain about nearby
🗺️ **Market environments:** K-Means on 611 neighbourhood profiles, which finds 4 market types, from "downtown hub" to "busy outer corridor with weak competitors"
💰 **Financial risk:** break-even, scenarios and a 4,000-run Monte Carlo simulation
📋 **One explainable launch report** in a Streamlit app

My part: [describe your component in 1–2 lines].

Biggest lesson: feature engineering and leakage-safe validation mattered more than the choice of model. Adding business age and attributes took our regression from R² 0.06 to 0.61.

🔗 GitHub: [repository link]
Thanks to @NTI and our instructors, and to my teammates [@name] [@name].

#MachineLearning #DataScience #NLP #Streamlit #Python #NTI #Capstone
