# Real Time Personalized Recommendation System

## Features
NCF (Neural Collobrative Filtering) Model: The neural network learns latent factors for users and items.

Hybrid Input: Uses user, book, location, and age as inputs.

Interactive Dashboard: Allows you to get recommendations for a user or find similar books for a given book.

Visual Output: Displays recommended books with details and images.

## Requirements
`pip install torch streamlit pandas numpy scikit-learn`

## Run
`streamlit run app.py`

## Loss Function
I chose Bayesian Personalized Ranking Loss (bpr_loss) because it directly optimizes the ranking of recommended items rather than just predicting ratings. BPR loss is especially effective for implicit feedback data and helps the model learn to rank items a user prefers higher than those they do not. This makes it well-suited for real-world recommendation systems where the main goal is to show the most relevant items at the top.

## Metrics
Coverage: Measures the proportion of items that can be recommended by the model, reflecting how broad the recommendations are across the catalog.

Diversity: Quantifies how different the recommended items are from each other, encouraging varied recommendations.

Personalization: Indicates how much the recommended lists differ between users, capturing the system’s ability to tailor suggestions.

Novelty: Measures how unfamiliar or unexpected the recommended items are to the user, favoring less popular or previously unseen items.

I have decided to chose these metrics because a recommendation system should be personalized, recommend different items, recommend items which are cover not small amount of the dataset. If I would choose traditional metrics like precision, recall etc they would be very small because we want different recommendation than the books which the user ranked (I assume that the user have read the book which use ranked). If we wanted to find the books which the user ranked we would use metrics like precision, recall etc.

## Code, Data and Model Versioning
While we do not fully implement Data Version Control in this project, our approach would involve tracking changes in the dataset and any feature engineering steps. We would save raw, processed, and feature-enriched versions of the data with clear version numbers and metadata (such as processing date, applied scripts, and feature lists). For larger projects, we would use tools like DVC to manage data and feature pipelines, ensuring that every experiment can be reproduced with the same data snapshot.

To version models, we save each trained model with a unique identifier including the training date, hyperparameters, and feature set. Model artifacts (e.g., PyTorch .pth files). In a more production-oriented setting, tools like MLflow or even Apache Airflow can be used to automate model training, tracking, and deployment workflows, allowing us to compare different model versions and roll back if needed.

If we use Apache Airflow, we can automate the training and evaluation process, storing model artifacts and experiment logs as part of a pipeline. This enables consistent, automated model versioning, easy comparison of results, and reproducibility of the whole pipeline.

