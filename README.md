# SmartPlantCare

# Smart Dorm Room Plant Care Advisor

## 1. Problem Statement

Novice plant owners, particularly students in dorm environments with limited space and experience, often struggle with providing appropriate care for their houseplants. This frequently leads to unhealthy or dying plants due to a lack of readily accessible, personalized, and easy-to-understand guidance. This project aims to address this by creating a basic AI-driven advisor to help users make better plant care decisions.

## 2. Project Overview

This project implements a simple machine learning pipeline in a Google Colab notebook to act as a "Smart Plant Care Advisor". It leverages a synthetically generated dataset to train three different machine learning models, each serving a unique purpose in assisting the user:

1.  **Growth Prediction (Linear Regression):** Predicts how much a plant is likely to grow in the next month based on its current care routine.
2.  **Risk Assessment (Linear Classification):** Classifies the plant's current health into a risk category ("Low", "Moderate", or "High") based on its observable conditions.
3.  **Care Recommendation (K-Nearest Neighbors):** Finds similar plants from a historical dataset and suggests care adjustments based on what worked for those similar, healthy plants.

## 3. Technologies and Libraries Used

This project is built using Python 3 and relies on the following core data science libraries:

-   **NumPy:** For numerical operations and creating the synthetic dataset.
-   **Pandas:** For data manipulation and management, primarily using its powerful DataFrame structure.
-   **Scikit-learn (sklearn):** The primary machine learning library used for:
    -   Data preprocessing (`StandardScaler`, `OneHotEncoder`, `LabelEncoder`, `train_test_split`).
    -   Implementing the three ML models (`LinearRegression`, `LogisticRegression`, `KNeighborsClassifier`/`NearestNeighbors`).
    -   Model evaluation (`accuracy_score`, `r2_score`, `confusion_matrix`, etc.).
-   **Matplotlib & Seaborn:** For data visualization and creating plots to explore the data and evaluate model performance.
-   **Google Colab:** As the development environment for running the Jupyter notebook.

## 4. Dataset

The dataset for this project is **synthetically generated** within the notebook. Since finding a public dataset with all the required features and target variables was unfeasible for this basic project, a custom dataset of 650 plant scenarios was created.

Key features include:
-   `Plant_Type`
-   `Sunlight_Hours_Per_Day`
-   `Watering_Frequency_Per_Week`
-   `Room_Temperature_C`
-   `Soil_Moisture_Reading`
-   `Leaf_Appearance`
-   `Observed_Pests`

Target variables generated based on these features are:
-   `Actual_Growth_In_1_Month_cm` (for Linear Regression)
-   `Risk_Category` (for Linear Classification)

## 5. The Three-Algorithm Approach

### a. Linear Regression
-   **Objective:** To predict a continuous numerical value: `Actual_Growth_In_1_Month_cm`.
-   **Process:** A Linear Regression model is trained to find a linear relationship between input features (like sunlight, water, fertilizer) and plant growth.
-   **Use Case:** Gives the user a quantitative forecast of how their current care routine might impact future growth.

### b. Linear Classification (using Logistic Regression)
-   **Objective:** To predict a discrete category: `Risk_Category` ('Low_Risk', 'Moderate_Risk_Needs_Monitoring', 'High_Risk_Immediate_Attention').
-   **Process:** A Logistic Regression model is trained to classify a plant into a risk group based on its features (like leaf appearance, soil moisture, pests).
-   **Use Case:** Acts as an early warning system, immediately flagging potential health issues for the user.

### c. K-Nearest Neighbors (KNN)
-   **Objective:** To provide actionable, data-driven recommendations.
-   **Process:** This algorithm is used not just for classification, but to find the 'k' most similar plant scenarios from the training data compared to a user's input plant.
-   **Use Case:** If a user's plant is struggling, KNN finds similar plants from the dataset that were healthy. The system then shows the care parameters (e.g., watering frequency, sunlight hours) of those successful neighbors, offering a concrete, example-based suggestion for what the user could try.

## 6. Project Structure

The project is contained within a single Jupyter Notebook (`.ipynb` file) which is structured as follows:

1.  **Synthetic Dataset Generation:** Code to create the project's dataset.
2.  **Exploratory Data Analysis (EDA):** Visualizing the data to understand distributions and relationships.
3.  **Data Preprocessing:** Cleaning and preparing the data for the models (scaling, encoding, train-test split).
4.  **Algorithm 1: Linear Regression:** Training, prediction, and evaluation.
5.  **Algorithm 2: Linear Classification:** Training, prediction, and evaluation.
6.  **Algorithm 3: K-Nearest Neighbors:** Demonstrating KNN as a classifier and, more importantly, as a tool for recommendation based on user input.
7.  **(Optional) Unified Advisor:** A final interactive cell that combines all three models to provide a holistic plant health assessment from a single user input.

## 7. How to Run the Project

1.  Clone this repository to your local machine.
2.  Open the `.ipynb` notebook file in a Jupyter environment like Google Colab or Jupyter Lab.
3.  Run the cells in sequential order from top to bottom.
4.  When you reach the interactive KNN recommendation section, you will be prompted to enter your own plant's details to get personalized advice.

## 8. Conclusion and Future Improvements

This project successfully demonstrates how three distinct machine learning algorithms can be integrated into a single pipeline to solve a simple, real-world problem.

**Future Improvements could include:**
-   Using a real-world, collected dataset instead of a synthetic one.
-   Implementing more complex models or using hyperparameter tuning to improve accuracy.
-   Building a simple web interface (using Flask or Streamlit) to make the advisor more user-friendly.
-   Expanding the dataset with more plant types and environmental factors.
