import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Food Delivery Demand Predictor")

st.title("U.S. Food Delivery Market Analysis")
st.subheader("Demand Prediction + Price Elasticity")

# -----------------------------
# Load Data
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_sample.csv")

    df['price'] = df['price'].astype(str)
    df['price'] = df['price'].str.replace(' USD', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    df = df[(df['price'] > 0) & (df['price'] < 200)]

    # 🔥 NEW DEMAND DEFINITION (Price matters)
    median_price = df['price'].median()
    threshold = df['ratings'].quantile(0.70)

    df['high_demand'] = (
        (df['ratings'] > threshold) &
        (df['price'] < median_price)
    ).astype(int)

    return df

df = load_data()

# -----------------------------
# Train Model (Pipeline)
# -----------------------------

X = df[['price', 'score', 'ratings']]
y = df['high_demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
model = pipeline

# -----------------------------
# 📊 Simple Dashboard
# -----------------------------

st.markdown("---")
st.subheader("📊 Market Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Average Price ($)", round(df['price'].mean(), 2))
col3.metric("Median Price ($)", round(df['price'].median(), 2))

st.markdown("### Top 20 Most Common Prices")
st.bar_chart(df['price'].value_counts().head(20))

# -----------------------------
# Prediction Section
# -----------------------------

st.subheader("Predict Demand")

price_input = st.number_input("Enter Price ($)", 1.0, 200.0, 10.0)
score_input = st.slider("Enter Rating Score", 1.0, 5.0, 4.5)
ratings_input = st.number_input("Enter Number of Ratings", 0, 2000, 200)

if st.button("Predict High Demand Probability"):

    input_data = pd.DataFrame(
        [[price_input, score_input, ratings_input]],
        columns=['price', 'score', 'ratings']
    )

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write(f"Probability of High Demand: {round(probability * 100, 2)}%")

    if prediction == 1:
        st.success("High Demand Expected")
    else:
        st.error("Low Demand Expected")

# -----------------------------
# Price Elasticity Simulation
# -----------------------------

st.markdown("---")
st.subheader("📊 Price Elasticity Simulation")

price_increase = st.slider("Increase price by %", 0, 50, 10)

new_price = price_input * (1 + price_increase / 100)

original_input = pd.DataFrame(
    [[price_input, score_input, ratings_input]],
    columns=['price', 'score', 'ratings']
)

new_input = pd.DataFrame(
    [[new_price, score_input, ratings_input]],
    columns=['price', 'score', 'ratings']
)

original_prob = model.predict_proba(original_input)[0][1]
new_prob = model.predict_proba(new_input)[0][1]

st.write(f"Original Demand Probability: {original_prob:.4f}")
st.write(f"New Demand Probability: {new_prob:.4f}")

if original_prob > 0 and price_increase > 0:
    elasticity = ((new_prob - original_prob) / original_prob) / (price_increase / 100)
    st.write(f"Estimated Elasticity: {round(elasticity,2)}")

    if elasticity < -1:
        st.info("Demand is Elastic — customers are price sensitive.")
    elif -1 <= elasticity < 0:
        st.info("Demand is Inelastic — moderate sensitivity.")
    else:
        st.info("Weak price sensitivity detected.")
else:
    st.write("Elasticity cannot be computed.")