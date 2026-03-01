import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Food Delivery Demand Predictor")

st.title("U.S. Food Delivery Market Analysis")
st.subheader("Demand Prediction Dashboard")

# -----------------------------
# Load and Prepare Data
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_sample.csv")

    # Clean price column
    df['price'] = df['price'].astype(str)
    df['price'] = df['price'].str.replace(' USD', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Remove extreme or invalid prices
    df = df[(df['price'] > 0) & (df['price'] < 200)]

    # Define high demand as top 30% restaurants
    threshold = df['ratings'].quantile(0.70)
    df['high_demand'] = (df['ratings'] > threshold).astype(int)

    return df

df = load_data()

# -----------------------------
# Train Model
# -----------------------------

X = df[['price', 'score', 'ratings']]
y = df['high_demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute ALL numerical columns
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Top 20 Most Common Prices")
st.bar_chart(df['price'].value_counts().head(20))

# -----------------------------
# Prediction Section
# -----------------------------

st.subheader("Predict Demand")

price_input = st.number_input("Enter Price ($)", min_value=1.0, max_value=200.0, value=10.0)
score_input = st.slider("Enter Rating Score", 1.0, 5.0, 4.5)
ratings_input = st.number_input("Enter Total Ratings Count", min_value=0, max_value=5000, value=200)

if st.button("Predict High Demand Probability"):

    input_data = pd.DataFrame(
        [[price_input, score_input, ratings_input]],
        columns=['price', 'score', 'ratings']
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Probability of High Demand: {round(probability * 100, 2)}%")

    if prediction == 1:
        st.success("High Demand Expected")
    else:
        st.error("Low Demand Expected")

    # Strategic Interpretation
    st.markdown("### Business Insight")
    if price_input > df['price'].median():
        st.info("This price is above market median. Consider competitive pricing.")
    else:
        st.info("This price is within competitive market range.")