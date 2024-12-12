import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# GLOBAL VARIABLES
uploaded_df = None
model = None
selected_target = None
numeric_cols = []
categorical_cols = []
selected_features = []
feature_types = {}

st.title("XGBoost Regression Model Training and Prediction App")

#################################
# 1. Upload Component
#################################
st.subheader("Upload File")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    # If you have a custom preprocessing function from previous milestones, call it here.
    # For example: uploaded_df = preprocess_data(uploaded_df)
    # Make sure preprocess_data includes all your cleaning steps.
    
    # Identify numeric and categorical columns after preprocessing
    numeric_cols = uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = uploaded_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        feature_types[col] = 'numeric'
    for col in categorical_cols:
        feature_types[col] = 'categorical'

#################################
# 2. Select Target Component
#################################
if uploaded_df is not None:
    st.subheader("Select Target Variable")
    # Target must be numeric for regression
    selected_target = st.selectbox("Select the target variable:", options=numeric_cols)
    
    features = [col for col in uploaded_df.columns if col != selected_target]

#################################
# 3. Barcharts Components
#################################
if uploaded_df is not None and selected_target is not None:
    st.subheader("Data Analysis")
    # Radio buttons for categorical variables to show average target
    if len(categorical_cols) > 0:
        selected_cat_for_avg = st.radio("Select categorical variable for average target plot:", categorical_cols)
        avg_df = uploaded_df.groupby(selected_cat_for_avg)[selected_target].mean().reset_index()

        st.write(f"Average of {selected_target} by {selected_cat_for_avg}:")
        st.bar_chart(data=avg_df, x=selected_cat_for_avg, y=selected_target)

    # Correlation with target for numeric features
    corr_vals = {}
    for col in numeric_cols:
        if col != selected_target:
            corr_val = uploaded_df[[col, selected_target]].corr().iloc[0,1]
            corr_vals[col] = abs(corr_val)

    corr_df = pd.DataFrame(list(corr_vals.items()), columns=["Numeric Variables", "Correlation Strength"])
    corr_df = corr_df.sort_values(by="Correlation Strength", ascending=False)
    
    st.write("Correlation Strength of Numerical Variables with Target")
    st.bar_chart(data=corr_df, x="Numeric Variables", y="Correlation Strength")

#################################
# 4. Train Component
#################################
if uploaded_df is not None and selected_target is not None:
    st.subheader("Train Model")
    st.write("Select features to use in the model:")
    selected_features = []
    for col in features:
        chk = st.checkbox(col, value=False)
        if chk:
            selected_features.append(col)
    
    if st.button("Train"):
        if len(selected_features) == 0:
            st.warning("Please select at least one feature to train the model.")
        else:
            X = uploaded_df[selected_features]
            y = uploaded_df[selected_target]

            chosen_cat = [f for f in selected_features if f in categorical_cols]
            chosen_num = [f for f in selected_features if f in numeric_cols]

            # Numeric transformer: impute, scale, and optionally add polynomial features
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False))  # Adjust if needed
            ])

            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, chosen_num),
                    ('cat', categorical_transformer, chosen_cat)
                ]
            )

            # XGBRegressor chosen as best model
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(n_estimators=100, random_state=42))
            ])
            model.fit(X, y)
            
            # Compute R² on the training set
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            st.write(f"The R² score on the training set is: {r2:.4f}")

#################################
# 5. Predict Component
#################################
# The predict component should only work if we have a trained model and selected features
if model is not None and len(selected_features) > 0:
    st.subheader("Predict")
    st.write("Enter feature values in the following order (comma-separated):")
    st.write(", ".join(selected_features))
    
    user_input = st.text_input("Input values (comma-separated):", value="")
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter values.")
        else:
            input_values = [val.strip() for val in user_input.split(",")]
            
            if len(input_values) != len(selected_features):
                st.error(f"Expected {len(selected_features)} values, got {len(input_values)}")
            else:
                # Convert numeric features to float; categorical remain strings
                final_input = {}
                parsing_error = False
                for i, feat in enumerate(selected_features):
                    if feat in numeric_cols:
                        try:
                            final_input[feat] = float(input_values[i])
                        except ValueError:
                            st.error(f"Invalid value for numeric feature {feat}")
                            parsing_error = True
                            break
                    else:
                        final_input[feat] = input_values[i]
                
                if not parsing_error:
                    X_pred = pd.DataFrame([final_input])
                    pred = model.predict(X_pred)
                    st.write(f"Predicted {selected_target}: {pred[0]:.4f}")


