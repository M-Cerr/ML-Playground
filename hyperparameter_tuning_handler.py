import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt

def parse_value(val_str, target_type):
    """Attempt to convert a string to the given target type."""
    try:
        if target_type == bool:
            # For booleans, treat common true strings as True.
            return val_str.lower() in ["true", "1", "yes"]
        elif target_type == int:
            return int(val_str)
        elif target_type == float:
            return float(val_str)
        else:
            return val_str
    except Exception:
        return val_str  # Fallback: return the string

def display_hyperparameter_tuning(selected_dataset_name):
    """
    Displays the Hyperparameter Tuning interface.
    
    Assumes that:
      - The training and test datasets are saved in st.session_state["train_dataset"] and ["test_dataset"].
      - The target feature is stored in st.session_state["target_feature"].
      
    The interface allows the user to select one of four models:
      - Linear Regression
      - Ridge Regression
      - Lasso Regression
      - ElasticNet
      
    Essential hyperparameters are exposed by default.
    If "Show More Parameters" is checked, additional parameters are displayed (via inputs) for tuning.
    When the user clicks "Run Model", the model is trained on the training data and evaluated on the test data.
    Performance metrics (MSE and R²) are then displayed.
    
    Returns:
        A tuple (model, metrics_dict) if the model is run; otherwise, (None, None).
    """
    st.markdown("## Hyperparameter Tuning & Model Simulation")
    
    # Ensure train/test split and target feature exist.
    if "train_dataset" not in st.session_state or "test_dataset" not in st.session_state:
        st.error("Train/Test split not found. Please split the dataset before tuning.")
        return None, None


    train_df = st.session_state["train_dataset"]
    test_df = st.session_state["test_dataset"]
    target_feature = st.session_state.get(f"{selected_dataset_name}_target_feature")
    if target_feature is None:
        st.error("Please specify a target feature in the dataset setup.")
        return None, None

    # Check that the target feature is numeric.
    if not pd.api.types.is_numeric_dtype(train_df[target_feature]):
        st.error(f"The target feature '{target_feature}' is not numeric. Please select a numeric target for regression.")
        return None, None
    
    # Prepare training and test sets.
    # Drop the target from feature sets.
    X_train = train_df.drop(columns=[target_feature])
    y_train = train_df[target_feature]
    X_test  = test_df.drop(columns=[target_feature])
    y_test  = test_df[target_feature]


    # (This check assumes that if any non-numeric feature remains, the user did not encode everything.)
    non_numeric_train = [col for col in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[col])]
    non_numeric_test  = [col for col in X_test.columns if not pd.api.types.is_numeric_dtype(X_test[col])]
    all_non_numeric = set(non_numeric_train).union(non_numeric_test)
    if all_non_numeric:
        st.error("The following feature(s) are non-numeric. Please encode them before running the model: " +
                 ", ".join(all_non_numeric))
        return None, None

    # Model selection.
    model_choice = st.selectbox("Select Regression Model", 
                                ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"],
                                key="model_choice")
    
    model_descriptions = {
        "Linear Regression": (
            "Linear Regression is the simplest model that fits a straight line to your data. "
            "It does not include any regularization and typically does not require much tuning."
        ),
        "Ridge Regression": (
            "Ridge Regression adds an L2 penalty to the loss function. "
            "Tuning the alpha parameter helps control overfitting by shrinking large coefficients."
        ),
        "Lasso Regression": (
            "Lasso Regression adds an L1 penalty, which can shrink some coefficients to zero, "
            "effectively performing feature selection."
        ),
        "ElasticNet": (
            "ElasticNet combines both L1 and L2 penalties. Tuning both alpha and l1_ratio allows you "
            "to balance between Ridge and Lasso behavior."
        )
    }
    st.info(model_descriptions[model_choice])
    
    # Set up essential hyperparameters.
    params = {}
    if model_choice in ["Ridge Regression", "Lasso Regression", "ElasticNet"]:
        params["alpha"] = st.slider("Alpha (Regularization Strength)", min_value=0.0, 
                                    max_value=10.0, value=1.0, step=0.1,
                                    help="Higher alpha increases regularization.")
    if model_choice == "ElasticNet":
        params["l1_ratio"] = st.slider("L1 Ratio", min_value=0.0, max_value=1.0,
                                       value=0.5, step=0.05,
                                       help="0 = Ridge-like, 1 = Lasso-like.")
    if model_choice == "Linear Regression":
        params["fit_intercept"] = st.checkbox("Fit Intercept", value=True,
                                              help="Include an intercept term.")
        params["positive"] = st.checkbox("Force Non-Negative Coefficients", value=False,
                                         help="If checked, coefficients will be >= 0.")

    # Additional parameters: "Show More Parameters"
    show_more = st.checkbox(
        "Show More Parameters", 
        key="show_more_params", 
        help="Shows parameters that are configurable, but usually not that impactful on overall model performance."
    )

    # Retrieve the primary key if specified in session state.
    primary_key = st.session_state.get(f"{selected_dataset_name}_primary_key")
    drop_primary = False
    if primary_key is not None:
        drop_primary = st.checkbox("Drop Primary Key / ID", value=True,
                                   help=f"Drop the primary key column '{primary_key}' from the dataset before training.")

    if show_more:
        # Instantiate a temporary model with the essential parameters.
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        if model_choice == "Linear Regression":
            temp_model = LinearRegression(**params)
            default_keys = {"fit_intercept", "positive"}
        elif model_choice == "Ridge Regression":
            temp_model = Ridge(**params, random_state=42)
            default_keys = {"alpha"}
        elif model_choice == "Lasso Regression":
            temp_model = Lasso(**params, random_state=42)
            default_keys = {"alpha"}
        elif model_choice == "ElasticNet":
            temp_model = ElasticNet(**params, random_state=42)
            default_keys = {"alpha", "l1_ratio"}
        else:
            temp_model = None
            default_keys = set()
        
        if temp_model is not None:
            all_params = temp_model.get_params()
            additional_params = {k: v for k, v in all_params.items() if k not in default_keys}
            st.markdown("### More Parameters")
            new_params = {}
            # For each additional parameter, display a text input with the current default value.
            for key, value in additional_params.items():
                # Show the parameter only if it is of a basic type (int, float, bool, str).
                if isinstance(value, (int, float, bool, str)):
                    new_val_str = st.text_input(f"{key} (default: {value})", value=str(value), key=f"more_{key}")
                    new_params[key] = parse_value(new_val_str, type(value))
            params.update(new_params)
    
    # When the user clicks "Run Model", train and evaluate the model.
    if st.button("Run Model"):

        # If drop_primary is selected and primary_key exists, drop it from features.
        if drop_primary and primary_key in X_train.columns:
            X_train = X_train.drop(columns=[primary_key])
        if drop_primary and primary_key in X_test.columns:
            X_test = X_test.drop(columns=[primary_key])
        
        # Imports Models Locally
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        
        # Instantiate the chosen model with the complete set of parameters.
        if model_choice == "Linear Regression":
            model = LinearRegression(**params)
        elif model_choice == "Ridge Regression":
            model = Ridge(**params, random_state=42)
        elif model_choice == "Lasso Regression":
            model = Lasso(**params, random_state=42)
        elif model_choice == "ElasticNet":
            model = ElasticNet(**params, random_state=42)
        else:
            st.error("Unknown model selection.")
            return None, None
        
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test  = mean_squared_error(y_test, y_pred_test)
        r2_train  = r2_score(y_train, y_pred_train)
        r2_test   = r2_score(y_test, y_pred_test)
        
        st.markdown("### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train MSE", f"{mse_train:.2f}")
            st.metric("Train R²", f"{r2_train:.3f}")
            # Create a chart for training metrics.
            train_df = pd.DataFrame({
                "Metric": ["MSE", "R²"],
                "Value": [mse_train, r2_train]
            })
            train_chart = alt.Chart(train_df).mark_bar().encode(
                x=alt.X("Metric:N", title="Train Metric"),
                y=alt.Y("Value:Q", title="Value"),
                tooltip=["Metric", "Value"]
            ).properties(
                width=200,
                height=200,
                title="Training Metrics"
            )
            st.altair_chart(train_chart, use_container_width=True)

        with col2:
            st.metric("Test MSE", f"{mse_test:.2f}")
            st.metric("Test R²", f"{r2_test:.3f}")
            test_df = pd.DataFrame({
                "Metric": ["MSE", "R²"],
                "Value": [mse_test, r2_test]
            })
            test_chart = alt.Chart(test_df).mark_bar().encode(
                x=alt.X("Metric:N", title="Test Metric"),
                y=alt.Y("Value:Q", title="Value"),
                tooltip=["Metric", "Value"]
            ).properties(
                width=200,
                height=200,
                title="Testing Metrics"
            )
            st.altair_chart(test_chart, use_container_width=True)
        
        # Additional help explanations for the metrics.
        st.markdown("""
        **Performance Metrics Explained:**
        - **MSE (Mean Squared Error):** Lower values indicate better fit; it represents the average squared difference between predicted and actual values.
        - **R² (Coefficient of Determination):** Values range from 0 to 1 (higher is better); it indicates the proportion of variance in the target that is predictable from the features.
        """)
        
        # Train/Test Loss Graphs (Line Charts)
        train_residuals = y_train - y_pred_train
        test_residuals = y_test - y_pred_test
        # Create DataFrames for plotting: Predicted values vs Residuals.
        train_plot_df = pd.DataFrame({
            "Predicted": y_pred_train,
            "Residual": train_residuals
        })
        test_plot_df = pd.DataFrame({
            "Predicted": y_pred_test,
            "Residual": test_residuals
        })

        st.markdown("### Residual Plots")
        col3, col4 = st.columns(2)
        with col3:
            # Scatter plot for training residuals.
            train_scatter = alt.Chart(train_plot_df).mark_point().encode(
                x=alt.X("Predicted:Q", title="Predicted Values"),
                y=alt.Y("Residual:Q", title="Residuals")
            ).properties(
                width=300,
                height=200,
                title="Training Residuals"
            )
            # Horizontal rule at zero.
            train_rule = alt.Chart(train_plot_df).mark_rule(color='red').encode(y=alt.Y("0:Q"))
            st.altair_chart(train_scatter + train_rule, use_container_width=True)
        with col4:
            # Scatter plot for test residuals.
            test_scatter = alt.Chart(test_plot_df).mark_point().encode(
                x=alt.X("Predicted:Q", title="Predicted Values"),
                y=alt.Y("Residual:Q", title="Residuals")
            ).properties(
                width=300,
                height=200,
                title="Testing Residuals"
            )
            test_rule = alt.Chart(test_plot_df).mark_rule(color='red').encode(y=alt.Y("0:Q"))
            st.altair_chart(test_scatter + test_rule, use_container_width=True)

        st.markdown("""
        **Residual Plot Explanation:**
        - Each point represents a prediction. The x-axis shows the predicted value, and the y-axis shows the residual (actual minus predicted).
        - Ideally, the points should be randomly scattered around the horizontal line at 0.
        - Systematic patterns (like curvature) may indicate that the model isn’t capturing some underlying relationships.
        """)
        
        # Model Comparison: Check if a previous model was saved.
        if "saved_model_metrics" in st.session_state:
            saved = st.session_state["saved_model_metrics"]
            comp_df = pd.DataFrame({
                "Metric": ["Train MSE", "Train R²", "Test MSE", "Test R²"],
                "Saved": [saved["mse_train"], saved["r2_train"], saved["mse_test"], saved["r2_test"]],
                "Current": [mse_train, r2_train, mse_test, r2_test]
            })
            def compare_metric(metric, saved_val, current_val):
                # For MSE, lower is better; for R², higher is better.
                if metric in ["Train MSE", "Test MSE"]:
                    if current_val < saved_val:
                        return "🔼"
                    elif current_val > saved_val:
                        return "🔽"
                    else:
                        return "➖"
                else:
                    if current_val > saved_val:
                        return "🔼"
                    elif current_val < saved_val:
                        return "🔽"
                    else:
                        return "➖"
            comp_df["Improvement"] = [compare_metric(row["Metric"], row["Saved"], row["Current"]) for index, row in comp_df.iterrows()]
            st.markdown("### Model Comparison")
            st.dataframe(comp_df)
            if st.button("Save Current Model Settings (Overwrite Saved Model)"):
                st.session_state["saved_model_settings"] = {"model_choice": model_choice, "params": params.copy()}
                st.session_state["saved_model_metrics"] = {"mse_train": mse_train, "r2_train": r2_train, "mse_test": mse_test, "r2_test": r2_test}
                st.success("Current model settings saved.")
                st.rerun()
        else:
            # If no saved model, save the current settings.
            st.session_state["saved_model_settings"] = {"model_choice": model_choice, "params": params.copy()}
            st.session_state["saved_model_metrics"] = {"mse_train": mse_train, "r2_train": r2_train, "mse_test": mse_test, "r2_test": r2_test}

        st.success("Model training complete. You can modify the parameters and re-run the model as you'd like.")
        return model, {"mse_train": mse_train, "mse_test": mse_test, "r2_train": r2_train, "r2_test": r2_test}
    
    return None, None
