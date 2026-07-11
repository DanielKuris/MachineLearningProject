# This file contains the code to compute SHAP interaction values for a Random Forest model
# Implemented here for reproducibility and refactoring purposes

import shap
import pandas as pd
import numpy as np


def compute_rf_shap_stats(
    model,
    X_explain,
    y_explain,
    feature_names,
    config_name,
    top_k=10
):

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )

    print(f"\n[{config_name}] Computing SHAP values...")


    # =============================
    # Regular SHAP
    # =============================

    shap_values = explainer.shap_values(
        X_explain
    )


    if isinstance(shap_values, list):

        shap_values = shap_values[1]

    elif len(shap_values.shape) == 3:

        shap_values = shap_values[:, :, 1]


    shap_importance = (
        pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feature_names
        )
        .sort_values(
            ascending=False
        )
        .head(top_k)
    )


    print(
        f"[{config_name}] SHAP values completed."
    )


    # =============================
    # SHAP interactions
    # =============================

    print(
        f"[{config_name}] Computing SHAP interaction values..."
    )


    interaction_values = (
        explainer.shap_interaction_values(
            X_explain
        )
    )


    if isinstance(interaction_values, list):

        interaction_values = interaction_values[1]

    elif len(interaction_values.shape) == 4:

        interaction_values = interaction_values[:, :, :, 1]


    print(
        "Interaction shape:",
        interaction_values.shape
    )


    # =============================
    # Interaction extraction helper
    # =============================

    def extract_interactions(values, statistic):

        if statistic == "mean":

            matrix = np.abs(values).mean(axis=0)

        elif statistic == "median":

            matrix = np.median(
                np.abs(values),
                axis=0
            )


        interactions = []

        for i in range(len(feature_names)):

            for j in range(i + 1, len(feature_names)):

                interactions.append(
                    {
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "interaction": matrix[i, j]
                    }
                )


        return (
            pd.DataFrame(interactions)
            .sort_values(
                "interaction",
                ascending=False
            )
            .head(top_k)
        )


    # =============================
    # Global interactions
    # =============================

    global_mean = extract_interactions(
        interaction_values,
        "mean"
    )

    global_median = extract_interactions(
        interaction_values,
        "median"
    )


    # =============================
    # Fraud interactions
    # =============================

    fraud_mask = (
        np.array(y_explain) == 1
    )

    fraud_values = (
        interaction_values[fraud_mask]
    )


    fraud_mean = extract_interactions(
        fraud_values,
        "mean"
    )

    fraud_median = extract_interactions(
        fraud_values,
        "median"
    )


    # =============================
    # Normal interactions
    # =============================

    normal_mask = (
        np.array(y_explain) == 0
    )

    normal_values = (
        interaction_values[normal_mask]
    )


    normal_mean = extract_interactions(
        normal_values,
        "mean"
    )

    normal_median = extract_interactions(
        normal_values,
        "median"
    )


    return {

        "shap_importance": shap_importance,

        "interaction_global_mean": global_mean,
        "interaction_global_median": global_median,

        "interaction_fraud_mean": fraud_mean,
        "interaction_fraud_median": fraud_median,

        "interaction_normal_mean": normal_mean,
        "interaction_normal_median": normal_median
    }