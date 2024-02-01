import streamlit as st
import os
def app():
    st.write('This is the `All ML models` page of the multi-page app.')
def app():
    st.title('Explore ML Models')
    st.write('This app aims to create an engaging and educational experience for users interested in understanding various machine learning models. Users can explore the models, learn about their characteristics, and potentially make informed decisions about which models to use in different scenarios.')
    st.write('Discover and interact with various machine learning models. Choose a model to explore and visualize its performance on different datasets.')
    st.write('Here is the brief description of all the models that are available in this app.')

    # List of ML models with descriptions
    ml_models_info = {
        'Linear Regression': 'A linear approach to modeling the relationship between a dependent variable and one or more independent variables.',
        'Logistic Regression': 'A regression model for binary classification that predicts the probability of the occurrence of an event.',
        'K-Nearest Neighbors': 'A type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data.',
        'Support Vector Machines': 'A supervised learning model used for classification and regression analysis. It creates a hyperplane that separates data points into classes.',
        'Decision Trees': 'A tree-like model where an internal node represents a feature or attribute, the branch represents a decision rule, and each leaf node represents the outcome.',
        'Random Forest': 'An ensemble learning method that operates by constructing a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.',
        'AdaBoost': 'An ensemble learning method that combines weak classifiers to form a strong classifier.',
        'Gradient Boosting': 'An ensemble learning technique for regression, classification, and ranking problems, producing a prediction model in the form of an ensemble of weak prediction models.',
        'Principal Component Analysis (PCA)': 'A technique for reducing the dimensionality of data while retaining as much of the variation in the data as possible.'
    }

    # Add local paths or URLs to images for each model
    base_dir = 'images'

    model_images = {
        'Linear Regression': os.path.join(base_dir, 'linear.png'),
        'Logistic Regression': os.path.join(base_dir, 'logistic.png'),
        'K-Nearest Neighbors': os.path.join(base_dir, 'knn.png'),
        'Support Vector Machines': os.path.join(base_dir, 'svm.png'),
        'Decision Trees': os.path.join(base_dir, 'dcision.png'),
        'Random Forest': os.path.join(base_dir, 'forest.png'),
        'AdaBoost': os.path.join(base_dir, 'ada.png'),
        'Gradient Boosting': os.path.join(base_dir, 'gradient.png'),
        'Principal Component Analysis (PCA)': os.path.join(base_dir, 'pca.jpg')
    }

    # Display model descriptions and images in a 3x3 grid
    col1, col2, col3 = st.columns(3)
    for model, image_path_or_url in model_images.items():
        with col1, col2, col3:
            col = col1 if model_images[model] in (model_images['Linear Regression'], model_images['Logistic Regression'], model_images['K-Nearest Neighbors']) else \
                col2 if model_images[model] in (model_images['Support Vector Machines'], model_images['Decision Trees'], model_images['Random Forest']) else col3

            # Display the image and description in the current column
            col.image(image_path_or_url, caption=model, use_column_width=True)
            col.write(ml_models_info[model])

if __name__ == "__main__":
    app()
