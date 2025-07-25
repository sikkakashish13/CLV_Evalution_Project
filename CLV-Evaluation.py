# Importing libraries
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
df = pd.read_csv("sample_customer_data.csv")  # Must contain 'features' & 'target' columns

# data cleaning
df['CustomerID'] = df['CustomerID'].str.replace("C",'')


# Splitting data
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop Factors Driving CLV:\n")
print(feature_importance.head(5))

# Recommendation Logic
def business_insight(feature, value):
    if feature == "purchase_frequency" and value > 10:
        return "Encourage loyalty upgrades"
    elif feature == "avg_order_value" and value < 100:
        return "Bundle products to raise order value"
    else:
        return "Maintain engagement strategy"
    

import streamlit as st
import matplotlib.pyplot as plt

st.title("CLV Evaluation Dashboard")

# Interactive Visualization
st.subheader("Feature Importance")
st.dataframe(feature_importance)

st.subheader("Model Performance")
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ax.matshow(confusion_matrix(y_test, y_pred), cmap="Blues")
st.pyplot(fig)