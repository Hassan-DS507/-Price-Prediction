
# Dynamic Pricing Prediction Engine: A Proof-of-Concept

## Executive Summary

This project develops a high-accuracy machine learning model to predict ride-hailing service prices. Using Uber data from Boston, we engineered a robust pipeline that processes spatial, temporal, and service-type features to forecast fares. Our Artificial Neural Network (ANN) model achieved **92.32% accuracy** (7.68% MAPE), significantly outperforming a strong linear regression baseline. This work serves as a foundational Proof-of-Concept (PoC) for implementing a dynamic, data-driven pricing system in related domains, such as optimizing service allocation and transparent pricing models.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Business Motivation](#business-motivation)
3.  [Dataset & Preprocessing](#dataset--preprocessing)
4.  [Methodology](#methodology)
5.  [Modeling](#modeling)
6.  [Results](#results)
7.  [Key Insights & Limitations](#key-insights--limitations)
8.  [Future Work & Deployment](#future-work--deployment)
9.  [Conclusion](#conclusion)
10. [Authors](#authors)

---

## Project Overview

The dynamic pricing algorithms used by ride-hailing platforms like Uber and Lyft are complex systems influenced by numerous factors. This project deconstructs this problem by leveraging a public dataset to identify key price drivers and build predictive models. The success of this PoC demonstrates the potential for transferring this methodology to other on-demand service industries.

**Primary Objective:** To build and evaluate machine learning models for predicting ride-hailing fares based on trip details, service type, and location.

**Key Achievement:** An ANN model that predicts prices with **92.32% accuracy**, validating the feasibility of a data-driven pricing engine.

## Business Motivation

Accurate price prediction is not limited to ride-hailing. For our company's core application—dynamic nurse/patient matching—this capability is critical for:

*   **Optimized Resource Allocation:** Ensuring fair compensation for nurses and availability during peak demand periods.
*   **Improved Financial Planning:** Enabling reliable forecasting of operational revenues and costs.
*   **Enhanced User Trust:** Building transparency through predictable and explainable pricing for end-users.

## Dataset & Preprocessing

The analysis is based on a combined dataset of Uber/Lyft ride estimates and Boston weather data from November-December 2018.

**Data Source:** [Publicly available dataset on Kaggle/etc.] *(Note: Add a link or citation here if available)*

**Initial Data Characteristics:** ~385,000 records with features including price, distance, service type, location, and timestamp.

**Preprocessing Pipeline:**

1.  **Data Cleaning:**
    *   Filtered data to include only Uber rides.
    *   Removed records with missing target values (`price`), resulting in **330,568 valid samples**.
2.  **Feature Selection:**
    *   **Retained Features:** `distance`, `cab_type`, `name` (product type), `source`, `destination`.
    *   **Removed Features:**
        *   `id`: Unique identifier with no predictive power.
        *   `time_stamp`: Invalid data (all set to 1970).
        *   `surge_multiplier`: Considered an output of the pricing system, not an input feature.
3.  **Feature Engineering:**
    *   Applied One-Hot Encoding to all categorical variables (`name`, `source`, `destination`).

## Methodology

The project followed a standard CRISP-DM methodology:
1.  **Business Understanding:** Defined the project goals and success metrics.
2.  **Data Understanding:** Conducted Exploratory Data Analysis (EDA) to identify distributions, correlations, and outliers.
3.  **Data Preparation:** Engineered and cleaned features as described above.
4.  **Modeling:** Trained and evaluated multiple regression models.
5.  **Evaluation:** Compared model performance using robust metrics (R², MAPE).
6.  **Deployment Planning:** Outlined the path to production for the PoC.

## Modeling

We implemented and compared two model architectures:

### 1. Baseline: Linear Regression
*   **Purpose:** To establish a strong, interpretable performance baseline.
*   **Implementation:** Built using a `Scikit-Learn` pipeline with a `ColumnTransformer` for efficient preprocessing and a `LinearRegression` estimator.

### 2. Primary Model: Artificial Neural Network (ANN)
*   **Purpose:** To capture complex, non-linear relationships in the data for superior predictive accuracy.
*   **Architecture:**
    *   **Input Layer:** Receives the processed feature vector.
    *   **Hidden Layers:** 
        *   Dense (64 units, ReLU) + Dropout (0.3)
        *   Dense (32 units, ReLU) + Dropout (0.3)
        *   Dense (16 units, ReLU) + Dropout (0.2)
    *   **Output Layer:** Dense (1 unit, linear activation for regression).
*   **Training Parameters:**
    *   **Optimizer:** Adam (learning rate = 0.001)
    *   **Loss Function:** Mean Absolute Error (MAE)
    *   **Validation:** Trained with a held-out validation set for early stopping.

## Results

Model performance was evaluated on a unseen test set. The results are summarized below:

| Model | R² Score | Mean Absolute Percentage Error (MAPE) | Approximate Accuracy |
| :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 0.9208 | 11.64% | ~88.36% |
| **Artificial Neural Network** | N/A | **7.68%** | **~92.32%** |

**Conclusion:** The ANN model demonstrated a **33% reduction in error** (from 11.64% to 7.68% MAPE) compared to the already strong baseline, confirming its superior ability to model the pricing dynamics.

## Key Insights & Limitations

### Insights
*   **Distance is the primary driver** of price, but not the only one.
*   **Service type** (e.g., UberX vs. Uber Black) is a significant contributor to fare variations.
*   **Geographical features** (source and destination) are critical for capturing location-based demand and pricing surges.
*   **Non-linear models** like ANNs are essential for achieving the highest accuracy in dynamic pricing problems.

### Limitations (PoC Scope)
*   **Temporal Scope:** Data is limited to a two-week period, lacking seasonal trends.
*   **Geographical Scope:** Model is trained exclusively on Boston data.
*   **Data Fidelity:** Prices are estimates from the API, not actual completed trip fares.
*   **Excluded Features:** External factors like weather and real-time traffic were not integrated into the final model.

## Future Work & Deployment

This PoC lays the groundwork for a production-grade system. The immediate next steps include:

1.  **Data Expansion:** Ingesting data over a longer timeframe and integrating real-time APIs for traffic, weather, and public events.
2.  **Model Advancement:** Experimenting with other advanced algorithms (e.g., Gradient Boosted Machines like XGBoost, LightGBM) and more complex ANN architectures.
3.  **Hyperparameter Optimization:** Conducting a systematic search (e.g., with KerasTuner or Optuna) to refine the ANN model.
4.  **Model Deployment:** Containerizing the chosen model with Docker and deploying it as a scalable REST API using a framework like FastAPI or Flask, ready for integration into a scheduling application.

## Conclusion

This project successfully demonstrates that machine learning can accurately model dynamic pricing systems. The ANN model's **92.32% accuracy** provides a strong technical foundation for building a dynamic pricing engine for our nurse/patient matching platform. By transitioning this PoC into a production system, we can achieve more efficient operations, predictable financials, and greater user satisfaction.

## Authors

- **Mohamed Mahmoud**
- **Hassan Abdul-razaq**

*Project Report Date: September 18, 2025*
```
