import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
# GLOBALS
uploaded_df = None
model = None
selected_target = None
numeric_cols = []
categorical_cols = []
selected_features = []

# Parameters learned at training time for transformations
numeric_col_means = {}
numeric_col_min = {}
numeric_col_max = {}
final_feature_cols = []  # After encoding
trained = False

st.title("XGBoost Regression Model Training and Prediction App")

#################################
# 1. Upload Component
#################################
st.subheader("Upload File")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    # Identify numeric and categorical columns
    numeric_cols = uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = uploaded_df.select_dtypes(exclude=[np.number]).columns.tolist()

#################################
# 2. Select Target Component
#################################
if uploaded_df is not None:
    st.subheader("Select Target Variable")
    selected_target = st.selectbox("Select the target variable:", options=numeric_cols)
    features = [col for col in uploaded_df.columns if col != selected_target]

#################################
# Helper functions for transformations
#################################
def fit_transform_training_data(X_train, chosen_cat, chosen_num):
    global numeric_col_means, numeric_col_min, numeric_col_max, final_feature_cols

    # Impute numeric columns with mean
    for col in chosen_num:
        mean_val = X_train[col].mean()
        numeric_col_means[col] = mean_val
        X_train[col] = X_train[col].fillna(mean_val)

    # Min-Max scale numeric
    for col in chosen_num:
        col_min = X_train[col].min()
        col_max = X_train[col].max()
        numeric_col_min[col] = col_min
        numeric_col_max[col] = col_max
        # Avoid division by zero if col_max == col_min
        if col_max != col_min:
            X_train[col] = (X_train[col] - col_min) / (col_max - col_min)
        else:
            # If there's no variation, set feature to 0
            X_train[col] = 0.0

    # One-hot encode categorical
    if len(chosen_cat) > 0:
        X_train = pd.get_dummies(X_train, columns=chosen_cat)
    # Store the final columns after encoding
    final_feature_cols = X_train.columns.tolist()
    return X_train

def transform_new_data(X_new):
    # Apply same transformations using stored parameters
    for col in numeric_col_means:
        if col in X_new.columns:
            X_new[col] = X_new[col].fillna(numeric_col_means[col])
        else:
            # If numeric col not in X_new, add it with mean
            X_new[col] = numeric_col_means[col]

    for col in numeric_col_min:
        if col in X_new.columns:
            col_min = numeric_col_min[col]
            col_max = numeric_col_max[col]
            if col_max != col_min:
                X_new[col] = (X_new[col] - col_min) / (col_max - col_min)
            else:
                X_new[col] = 0.0
        else:
            # If missing this numeric column, create it
            X_new[col] = 0.0

    # Handle categorical columns by re-applying get_dummies logic
    # First, get dummies
    X_new = pd.get_dummies(X_new)
    # Add missing columns if any
    for col in final_feature_cols:
        if col not in X_new.columns:
            X_new[col] = 0.0

    # Remove extra columns not in final_feature_cols
    X_new = X_new[final_feature_cols]

    return X_new

#################################
# 3. Barcharts Components
#################################
if uploaded_df is not None and selected_target is not None:
    st.subheader("Data Analysis")

    # Average target by chosen categorical variable
    if len(categorical_cols) > 0:
        selected_cat_for_avg = st.radio("Select categorical variable for average target plot:", categorical_cols)
        avg_df = uploaded_df.groupby(selected_cat_for_avg)[selected_target].mean().reset_index()
        st.write(f"Average of {selected_target} by {selected_cat_for_avg}:")
        st.bar_chart(data=avg_df, x=selected_cat_for_avg, y=selected_target)

    # Correlation with target for numeric features (using pandas)
    corr_vals = {}
    for col in numeric_cols:
        if col != selected_target:
            corr_val = uploaded_df[[col, selected_target]].corr().iloc[0,1]
            corr_vals[col] = abs(corr_val)

    if len(corr_vals) > 0:
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
            X = uploaded_df[selected_features].copy()
            y = uploaded_df[selected_target].copy()

            chosen_cat = [f for f in selected_features if f in categorical_cols]
            chosen_num = [f for f in selected_features if f in numeric_cols]

            # Transform the training data
            X_transformed = fit_transform_training_data(X, chosen_cat, chosen_num)

            # Train XGBoost directly
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_transformed, y)

            # Evaluate on training set
            y_pred = model.predict(X_transformed)
            # Compute R² manually
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot)
            st.write(f"The R² score on the training set is: {r2:.4f}")

            # Store model and trained flag globally
            trained = True

#################################
# 5. Predict Component
#################################
if 'model' in globals() and model is not None and trained and len(selected_features) > 0:
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
                # Create a dataframe from input
                input_dict = {}
                for i, feat in enumerate(selected_features):
                    # Determine if feat is numeric or categorical
                    if feat in numeric_cols:
                        try:
                            val = float(input_values[i])
                        except ValueError:
                            st.error(f"Invalid value for numeric feature {feat}")
                            break
                        input_dict[feat] = val
                    else:
                        input_dict[feat] = input_values[i]

                # If no break occurred, proceed
                if len(input_dict) == len(selected_features):
                    X_pred = pd.DataFrame([input_dict])
                    # Transform with the same parameters as training
                    X_pred_transformed = transform_new_data(X_pred)
                    pred = model.predict(X_pred_transformed)
                    st.write(f"Predicted {selected_target}: {pred[0]:.4f}")
