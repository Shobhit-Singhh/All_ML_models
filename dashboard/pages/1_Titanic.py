import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
import scipy.cluster.hierarchy as sch
import io
w.filterwarnings("ignore")


def show_dataset_shape(df):
    st.write(df.shape)
def show_dataset_info(df):
    st.write("This step provides a brief overview of the dataset and its features.")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
def show_sample_dataset(df):
    sample_size = st.slider("Enter the sample size of data ", 1, 100, 10)
    rows = st.radio("Select the rows", ("Top rows", "Bottom rows", "Random rows"))
    if rows == "Top rows":
        st.write(df.head(sample_size))
    elif rows == "Bottom rows":
        st.write(df.tail(sample_size))
    else:
        st.write(df.sample(sample_size))
def show_dataset_description(df):
    st.write(df.describe().T)
def show_null_values(df):
    st.write(pd.DataFrame(df.isnull().sum(), columns=['Null Count']).T)
def show_duplicate_values(df):
    st.write(df.duplicated().sum())
def show_correlation_metric(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_columns = st.multiselect("Select columns for correlation", numeric_columns)
    if selected_columns:
        st.write("Correlation Matrix:")
        st.write(df[selected_columns].corr())
    else:
        st.write("No columns selected for correlation.")
def know_data(df):
    st.header("Know your data")
    st.write("The first step in any data science project is to understand the data. This section provides a brief overview of the dataset and its features.")

    show_shape = st.checkbox("Show dataset shape")
    if show_shape:
        show_dataset_shape(df)

    show_info = st.checkbox("Show dataset info")
    if show_info:
        show_dataset_info(df)

    show_sample = st.checkbox("Show sample of the dataset")
    if show_sample:
        show_sample_dataset(df)

    show_describe = st.checkbox("Show dataset description")
    if show_describe:
        show_dataset_description(df)

    show_null = st.checkbox("Show null values")
    if show_null:
        show_null_values(df)

    show_duplicate = st.checkbox("Show duplicate values")
    if show_duplicate:
        show_duplicate_values(df)

    show_corr = st.checkbox("Show correlation metric")
    if show_corr:
        show_correlation_metric(df)


def remove_null_values(df):
    try:
        df.dropna(inplace=True)
        st.write("Null values removed successfully.")
    except Exception as e:
        st.error(f"Error removing null values: {str(e)}")
def remove_duplicate_rows(df):
    try:
        df.drop_duplicates(inplace=True)
        st.write("Duplicate rows removed successfully.")
    except Exception as e:
        st.error(f"Error removing duplicate rows: {str(e)}")
def remove_columns(df):
    cols = st.multiselect("Select columns to remove", df.columns)
    if cols:
        try:
            df.drop(cols, axis=1, inplace=True)
            st.write("Columns removed successfully.")
        except Exception as e:
            st.error(f"Error removing columns: {str(e)}")
def rename_columns(df):
    st.write("Enter new column names:")
    selected_column = st.selectbox("Select a column:", df.columns)
    new_name = st.text_input("Enter new column name:")
    col1, col2 = st.columns(2)
    with col1:
        rename_button = st.button("Rename Column")
        if rename_button:
            try:
                df.rename(columns={selected_column: new_name}, inplace=True)
                st.success(f"Successfully renamed {selected_column} to {new_name}.")
            except Exception as e:
                st.error(f"Error renaming {selected_column} to {new_name}: {str(e)}")
    with col2:
        show = st.checkbox("Show the change")
    if show:
        show_dataset_info(df)
def convert_data_types(df):
    col1, col2 = st.columns(2)
    selected_column = st.selectbox("Select a column:", df.columns)
    new_dtype = st.selectbox("Select new data type:", ["object", "int64", "float64"])

    with col1:
        convert_button = st.button("Convert Data Type")
        if convert_button:
            try:
                df[selected_column] = df[selected_column].astype(new_dtype)
                st.success(f"Successfully converted {selected_column} to {new_dtype}.")
            except Exception as e:
                st.error(f"Error converting {selected_column} to {new_dtype}: {str(e)}")
    with col2:
        show = st.checkbox("Show the change")
    if show:
        show_dataset_info(df)
def add_bucket_column(df):
    col1, col2 = st.columns(2)
    selected_column = st.selectbox("Select a column:", df.columns)
    new_column_name = st.text_input("Enter new column name:", f"{selected_column}_bucket")
    bucket_type = st.selectbox("Select bucket type:", ["qcut", "cut"])
    bucket_size = st.slider("Select bucket size:", 1, 100, 10)

    with col1:
        bucket_button = st.button("Add Bucket Column")
        if bucket_button:
            try:
                if bucket_type == "qcut":
                    df[new_column_name] = pd.qcut(df[selected_column], q=bucket_size)
                elif bucket_type == "cut":
                    df[new_column_name] = pd.cut(df[selected_column], bins=bucket_size)
                st.success(f"Successfully added {new_column_name} as a bucket column.")
            except Exception as e:
                st.error(f"Error adding {new_column_name} as a bucket column: {str(e)}")
    with col2:
        show = st.checkbox("Show the change")
    if show:
        show_dataset_info(df)
def save_cleaned_data(df):
    try:
        df.to_csv("cleaned_data.csv", index=False)
        st.write("Cleaned data saved successfully.")
    except Exception as e:
        st.error(f"Error saving cleaned data: {str(e)}")
def clean_data(df):
    st.header("Clean your data")
    st.write("This section allows you to clean the dataset by removing null values and duplicate rows.")

    remove_null = st.checkbox("Remove null values")
    if remove_null:
        remove_null_values(df)

    remove_duplicate = st.checkbox("Remove duplicate rows")
    if remove_duplicate:
        remove_duplicate_rows(df)

    remove_columns_checkbox = st.checkbox("Remove columns")
    if remove_columns_checkbox:
        remove_columns(df)

    rename_column_checkbox = st.checkbox("Rename columns")
    if rename_column_checkbox:
        rename_columns(df)

    convert_columns_checkbox = st.checkbox("Convert data types")
    if convert_columns_checkbox:
        convert_data_types(df)

    add_bucket_checkbox = st.checkbox("Add bucket column")
    if add_bucket_checkbox:
        add_bucket_column(df)

    save_checkbox = st.checkbox("Save cleaned data")
    if save_checkbox:
        save_cleaned_data(df)


def split_dataset(df):
    unique_value = st.slider("Enter the number of unique values for threshold", 1, 20, 5)
    continous_columns = []
    categorical_columns = []
    discrete_columns = []

    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > unique_value:
            discrete_columns.append(col)
            
        elif df[col].nunique() <= unique_value:
            categorical_columns.append(col)
            
        else:
            continous_columns.append(col)
            

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Continuous Columns:")
        st.write(continous_columns)

    with col2:
        st.write("Categorical Columns:")
        st.write(categorical_columns)
    
    with col3:
        st.write("Discrete Columns:")
        st.write(discrete_columns)
    
    return continous_columns, categorical_columns, discrete_columns
def Scatter_plot(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    x_column = st.sidebar.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = st.sidebar.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.sidebar.selectbox("Hue", [None] + list(categorical_columns))
    size_column = st.sidebar.selectbox("Size", [None] + list(categorical_columns))
    style_column = st.sidebar.selectbox("Style", [None] + list(categorical_columns))
    palette = st.sidebar.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    sizes_tuple = st.sidebar.slider("Select Sizes Range", min_value=0, max_value=200, value=(50, 200))
    fig, ax = plt.subplots()
    try:
        sns.scatterplot(x=x_column, y=y_column, data=df, hue=hue_column, size=size_column, style=style_column, palette=palette, sizes=sizes_tuple, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting scatter plot: {str(e)}")
def Bar_plot(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    x_column = st.sidebar.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = st.sidebar.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.sidebar.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.sidebar.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    fig, ax = plt.subplots()
    try:
        sns.barplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting bar plot: {str(e)}")
def Box_plot(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    x_column = st.sidebar.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = st.sidebar.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.sidebar.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.sidebar.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    fig, ax = plt.subplots()
    try:
        sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting box plot: {str(e)}")
def Histogram(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    x_column = st.sidebar.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    bins = st.sidebar.slider("Select Number of Bins", min_value=1, max_value=100, value=30)
    color = st.sidebar.color_picker("Select Color", value="#1f77b4")

    fig, ax = plt.subplots()
    try:
        sns.histplot(df[x_column], bins=bins, color=color, kde=False)
        plt.xlabel(x_column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {x_column}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting histogram: {str(e)}")
def Heatmap(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    cmap = st.sidebar.selectbox("Select Colormap", ("viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"))
    fig, ax = plt.subplots()
    numeric_columns = df.select_dtypes(include=['number']).columns
    try:
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap=cmap,fmt=".1f")
        plt.title("Correlation Heatmap")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting heatmap: {str(e)}")
def Count_plot(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    x_column = st.sidebar.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.sidebar.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.sidebar.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        sns.countplot(x=x_column, data=df, hue=hue_column, palette=palette, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting count plot: {str(e)}")
def Pie_plot(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    column = st.sidebar.selectbox("Column", [None] + list(categorical_columns))
    palette = st.sidebar.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, colormap=palette, ax=ax)
        plt.title(f"Pie Plot of {column}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting pie plot: {str(e)}")
def Distplot(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    x_column = st.sidebar.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    color = st.sidebar.color_picker("Select Color", value="#1f77b4")
    bins_slider = st.sidebar.slider("Select Number of Bins", min_value=1, max_value=100)

    fig, ax = plt.subplots()
    try:
        sns.distplot(df[x_column], bins=bins_slider, color=color,hist_kws=dict(edgecolor="k", linewidth=.5))
        plt.xlabel(x_column)
        plt.ylabel("Frequency")
        plt.title(f"Distribution Plot of {x_column}")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting distplot: {str(e)}")
def Line_plot(df,continous_columns, categorical_columns, discrete_columns):
    st.sidebar.header("Select Plot Options")
    x_column = st.sidebar.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = st.sidebar.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = st.sidebar.selectbox("Hue", [None] + list(categorical_columns))
    palette = st.sidebar.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        sns.lineplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting line plot: {str(e)}")
def plot_data(df):
    st.header("Plot your data")
    st.write("This section allows you to Plot the dataset.")
    graph = st.sidebar.selectbox("Select Graph", ("Scatter Plot", "Bar Plot", "Box Plot", "Histogram", "Heatmap", "Count Plot", "Pie Plot", "Distplot", "Line Plot"))
    split_data = st.checkbox("Split data into continous and categorical columns")
    continous_columns, categorical_columns, discrete_columns = np.array(df.columns), np.array(df.columns), np.array(df.columns)
    if split_data:    
        continous_columns, categorical_columns, discrete_columns = split_dataset(df)
    
    if graph == "Scatter Plot":
        Scatter_plot(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Bar Plot":
        Bar_plot(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Box Plot":
        Box_plot(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Histogram":
        Histogram(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Heatmap":
        Heatmap(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Count Plot":
        Count_plot(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Pie Plot":
        Pie_plot(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Distplot":
        Distplot(df,continous_columns, categorical_columns, discrete_columns)
    elif graph == "Line Plot":
        Line_plot(df,continous_columns, categorical_columns, discrete_columns)


def app(df):
    # Section 1: Know your data
    know_data(df)
    
    # Section 2: Clean your data
    clean_data(df)
    
    # Section 3: Plot your data
    plot_data(df)


if __name__ == "__main__":
    st.title("Analyzing the Your Dataset")
    st.write("This app aims to create an engaging and educational experience for users interested in understanding various machine learning models. Users can explore the models, learn about their characteristics, and potentially make informed decisions about which models to use in different scenarios.")
    # uplode data or use default data
    st.sidebar.header("Upload your dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('/Users/shobhitsingh/Desktop/project/ML All/data/Titanic.csv')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    app(df)