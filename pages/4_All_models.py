import streamlit as st
import os

st.markdown(
    """
    # Welcome to the Machine Learning Course

    ## About the Course
    Welcome to this comprehensive Machine Learning course! Whether you're a beginner or have some experience, this course is designed to take you on a journey through the fascinating world of Machine Learning.

    ## What You'll Learn
    - Understand the fundamentals of Machine Learning.
    - Explore different types of Machine Learning, including Supervised, Unsupervised, and Reinforcement Learning.
    - Gain hands-on experience building and deploying Machine Learning models using Streamlit.

    ## Prerequisites
    - Basic programming knowledge (preferably in Python).
    - Enthusiasm to explore and learn about Machine Learning concepts.

    Let's dive in and unlock the power of Machine Learning together!
    """
)

st.markdown("""
            # Module 1: Introduction to Machine Learning

            ## What is Machine Learning?
            Machine Learning is a field of artificial intelligence that focuses on developing algorithms and models that allow computers to learn from data. Instead of being explicitly programmed, machines can improve their performance over time through experience.

            ## Types of Machine Learning
            1. **Supervised Learning:** Involves training a model on a labeled dataset, where the algorithm learns to map input data to the correct output.
            2. **Unsupervised Learning:** In this type, the algorithm is given unlabeled data and must find patterns or structure on its own.
            3. **Reinforcement Learning:** Agents learn by interacting with an environment, receiving feedback in the form of rewards or penalties.

            Throughout this course, we will explore these types of Machine Learning in detail, providing hands-on examples and practical applications.
            """)

st.markdown("""# Module 2: Basics of Data in Machine Learning

## The Heart of Machine Learning: Data
Data is the fuel that powers Machine Learning algorithms. Understanding its importance is crucial for anyone diving into the world of ML.

## Why is Data Important?
- **Training Models:** Machine Learning models learn patterns and make predictions based on historical data.
- **Quality Matters:** The quality of your data directly impacts the performance of your models.
- **Garbage In, Garbage Out:** Inaccurate or biased data can lead to flawed predictions.

## Data Types
1. **Numerical Data:** Quantitative data represented by numbers.
2. **Categorical Data:** Qualitative data with distinct categories.
3. **Text Data:** Unstructured data in the form of text.

## Sources of Data
- **Public Datasets:** Available for public use, covering various domains.
- **Company Databases:** Internal data collected by organizations.
- **Sensor Data:** Capturing information from physical sensors.

Understanding where your data comes from is the first step in effective Machine Learning.
""")

st.markdown("""# Linear Regression Parameters

Linear regression is a fundamental statistical technique used for modeling the relationship between dependent and independent variables. The scikit-learn library provides the `LinearRegression` class for implementing linear regression models. Below are the key parameters associated with this class:

## `linear_model=LinearRegression()`

Instantiate the Linear Regression model using the `LinearRegression` class.

## Model Parameters

### `linear_model.get_params()` 
For retrieve the current parameters of the Linear Regression model.

### Parameters:

| Parameter            | Description                                                     |
|-----------------------|-----------------------------------------------------------------|
| 1. `copy_X`           | - Determines whether to copy input data `X` (independent variables).   |
|                       | - `True`: Copy `X`.                                               |
|                       | - `False`: May overwrite the data.                                |
|                       |                                                                 |
| 2. `fit_intercept`    | - Indicates whether to calculate the intercept for the model.      |
|                       | - `True`: Allows the model to shift up or down, introducing bias.  |
|                       |                                                                 |
| 3. `n_jobs`           | - Specifies the number of processors for parallel processing.      |
|                       | - `-1`: Use all available processors.                              |
|                       | - `None`: No parallel processing.                                  |
|                       |                                                                 |
| 4. `positive`         | - Applicable when `fit_intercept` is `True`.                       |
|                       | - Forces coefficients to be positive.                              |
|                       | - Useful when a positive relationship is expected.                |
|                       |                                                                 |

## Example Parameter Values

```python
parameter = {
    'copy_X': True,
    'fit_intercept': True,
    'n_jobs': None,
    'positive': False
}
```
Adjust these parameters based on your specific needs when implementing the Linear Regression model in scikit-learn. This concise guide provides a quick reference for understanding and customizing the Linear Regression model.
""")