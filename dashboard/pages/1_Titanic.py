import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
import scipy.cluster.hierarchy as sch
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


w.filterwarnings("ignore")


def compare_distribution(df, dummy, col=None):
    if col is None:
        col = df.columns
    for i in col:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Distribution for {i}'])
        # Add histograms for the current column to the corresponding subplot
        fig.add_trace(go.Histogram(x=df[i], nbinsx=50, name='Before', marker=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Histogram(x=dummy[i], nbinsx=50, name='After', marker=dict(color='red', opacity=0.5)), row=1, col=1)
        fig.update_layout(title_text=f'Distribution Comparison for {i}', showlegend=False)
        st.plotly_chart(fig)
def compare_covariance(df, dummy, col,lg=True):
    num_cols = df.select_dtypes(include=['number']).columns
    
    for i in col:
        if i in num_cols:
            fig = make_subplots(rows=1, cols=1, subplot_titles=[f'Before {i}', f'After {i}'])
            # Add histograms for the current column to the corresponding subplot
            fig.add_trace(go.Histogram(x=df[i], nbinsx=50, name='Before', marker=dict(color='green')), row=1, col=1)
            fig.add_trace(go.Histogram(x=dummy[i], nbinsx=50, name='After', marker=dict(color='red', opacity=0.5)), row=1, col=1)
            fig.update_layout(title_text=f'Covariance Comparison for {i}', showlegend=lg)
            st.plotly_chart(fig)
            num_df = df[num_cols]
            num_dummy = dummy[num_cols]
            st.write(num_df.corr()[i].to_frame().T)
            st.write(num_dummy.corr()[i].to_frame().T)
        else:
            st.warning(f"{i} is not a numerical column")
def plot_categorical_distribution(df, dummy, columns, lg=True):
    num_columns = len(columns)
    fig = make_subplots(rows=1, cols=1, subplot_titles=sum([[f'Before {col}', f'After {col}'] for col in columns], []))

    for i, col in enumerate(columns, start=1):
        # Plotting the distribution before imputation
        before_trace = px.histogram(df, x=col, title=f'Distribution of {col}').data[0]
        before_trace.marker.color = 'red'
        fig.add_trace(before_trace, row=1, col=1)

        after_trace = px.histogram(dummy, x=col, title=f'Distribution of {col}').data[0]
        after_trace.marker.color = 'green'
        fig.add_trace(after_trace, row=1, col= 1)

        fig.update_layout(title_text=f'Distribution Comparison for {", ".join(columns)}', showlegend=lg)
        fig.update_xaxes(type='category')
        fig.update_layout(title_text=f'distribution Comparison for {col}', showlegend=lg)
        st.plotly_chart(fig)
def uplode_and_reset():
    with right_bar:
        st.header("Upload dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        reset_button = st.button("Reset dataset")
        
        if st.session_state.get('df') is not None:
            df = st.session_state.df
        elif uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        else:
            df = pd.read_csv('/Users/shobhitsingh/Desktop/project/ML All/data/Titanic.csv')
            st.session_state.df = df
            
        if reset_button:
            st.session_state.df = None
            st.success("Successfully reset the dataset.")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv('/Users/shobhitsingh/Desktop/project/ML All/data/Titanic.csv')
        st.write("To make the uploaded file visible")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    return df
def sidebar(df):
    with right_bar:
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
        
        save_button = st.button("Save")
        if save_button:
            df.to_csv("output.csv", index=False)
            st.success("Successfully saved the dataset.") 
        
        # st.title("Analyzing the Your Dataset")
        # st.write("This app aims to create an engaging and educational experience for users interested in understanding various machine learning models. Users can explore the models, learn about their characteristics, and potentially make informed decisions about which models to use in different scenarios.")


def show_dataset_shape(df):
    mid.write(f"The shape of the dataframe = {df.shape}")
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_sample_dataset(df):
    mid.header("Sample of the dataset")
    sample_size = mid.slider("Enter the sample size of data ", 1, 100, 10)
    rows = mid.radio("Select the rows", ("Top rows", "Bottom rows", "Random rows"))
    if rows == "Top rows":
        mid.write(df.head(sample_size))
    elif rows == "Bottom rows":
        mid.write(df.tail(sample_size))
    else:
        mid.write(df.sample(sample_size))
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_dataset_description(df):
    mid.header("Dataset description")
    mid.write(df.describe().T)
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_null_values(df):
    mid.header("Null values")
    mid.write(pd.DataFrame(df.isnull().sum(), columns=['Null Count']).T)
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_duplicate_values(df):
    mid.header("Duplicate values")
    mid.write(f"Total duplicate rows in the dataset = {df.duplicated().sum()}")
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def show_correlation_metric(df):
    mid.header("Correlation metric")
    numeric_columns = df.select_dtypes(include=['number']).columns
    selected_columns = mid.multiselect("Select columns for correlation", numeric_columns)
    if selected_columns:
        mid.write("Correlation Matrix:")
        mid.write(df[selected_columns].corr())
    else:
        mid.write("No columns selected for correlation.")
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def know_data(df):
    expander_1 = st.sidebar.expander("Know your data")
    expander_1.header("Know your data")
    expander_1.write("The first step in any data science project is to understand the data. This section provides a brief overview of the dataset and its features.")
    expander_1.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
    
    show_shape = expander_1.checkbox("Show dataset shape")
    if show_shape:
        show_dataset_shape(df)

    show_sample = expander_1.checkbox("Show sample of the dataset")
    if show_sample:
        show_sample_dataset(df)

    show_describe = expander_1.checkbox("Show dataset description")
    if show_describe:
        show_dataset_description(df)

    show_null = expander_1.checkbox("Show null values")
    if show_null:
        show_null_values(df)

    show_duplicate = expander_1.checkbox("Show duplicate values")
    if show_duplicate:
        show_duplicate_values(df)

    show_corr = expander_1.checkbox("Show correlation metric")
    if show_corr:
        show_correlation_metric(df)


def remove_duplicate_rows(df):
    try:
        df.drop_duplicates(inplace=True)
        mid.success("Duplicate rows removed successfully.")
    except Exception as e:
        mid.error(f"Error removing duplicate rows: {str(e)}")
    mid.session_state.df = df
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def remove_columns(df):
    mid.header("Remove columns")
    cols = mid.multiselect("Select columns to remove", df.columns)
    remove_button = mid.button("Remove Columns")
    if remove_button:
        try:
            df.drop(cols, axis=1, inplace=True)
            mid.success(f"{cols} Columns removed successfully.")
        except Exception as e:
            mid.error(f"Error removing columns: {str(e)}")
    mid.session_state.df = df
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def rename_columns(df):
    mid.header("Rename columns")
    mid.write("Enter new column names:")
    selected_column = mid.selectbox("Select a column for rename:", df.columns)
    new_name = mid.text_input("Enter new column name:")
    
    rename_button = mid.button("Rename Column")
    if rename_button:
        try:
            df.rename(columns={selected_column: new_name}, inplace=True)
            mid.success(f"Successfully renamed {selected_column} to {new_name}.")
        except Exception as e:
            mid.error(f"Error renaming {selected_column} to {new_name}: {str(e)}")
    mid.session_state.df = df
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def convert_data_types(df):
    mid.header("Convert data types")
    selected_column = mid.selectbox("Select a column for conversion:", df.columns)
    new_dtype = mid.selectbox("Select new data type:", ["object", "int64", "float64"])
    convert_button = mid.button("Convert Data Type")
    if convert_button:
        try:
            df[selected_column] = df[selected_column].astype(new_dtype)
            mid.success(f"Successfully converted {selected_column} to {new_dtype}.")
        except Exception as e:
            mid.error(f"Error converting {selected_column} to {new_dtype}: {str(e)}")
    mid.session_state.df = df
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def add_bucket_column(df):
    mid.header("Add bucket column")
    selected_column = mid.selectbox("Select a column to add the bucket for:", df.columns)
    new_column_name = mid.text_input("Enter new column name:", f"{selected_column}_bucket")
    bucket_type = mid.selectbox("Select bucket type:", ["Select the number of bins", "Select the bin intervals"])
    bucket_size = mid.slider("Select bucket size:", 1, 100, 10)
    bucket_button = mid.button("Add Bucket Column")
    if bucket_button:
        try:
            if bucket_type == "Select the number of bins":
                df[new_column_name] = pd.qcut(df[selected_column], q=bucket_size)
            elif bucket_type == "Select the bin intervals":
                df[new_column_name] = pd.cut(df[selected_column], bins=bucket_size)
            mid.success(f"Successfully added {new_column_name} as a bucket column.")
        except Exception as e:
            mid.error(f"Error adding {new_column_name} as a bucket column: {str(e)}")
    mid.session_state.df = df
    mid.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def clean_data(df):
    expander_2 = st.sidebar.expander("Clean your data")
    expander_2.header("Clean your data")
    expander_2.write("This section allows you to clean the dataset by removing null values and duplicate rows.")
    expander_2.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)

    remove_duplicate = expander_2.checkbox("Remove duplicate rows")
    if remove_duplicate:
        remove_duplicate_rows(df)

    rename_column_checkbox = expander_2.checkbox("Rename columns")
    if rename_column_checkbox:
        rename_columns(df)

    convert_columns_checkbox = expander_2.checkbox("Convert data types")
    if convert_columns_checkbox:
        convert_data_types(df)

    add_bucket_checkbox = expander_2.checkbox("Add bucket column")
    if add_bucket_checkbox:
        add_bucket_column(df)


def split_dataset(df):
    unique_value = mid.slider("Enter the number of unique values for threshold", 1, 20, 5)
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
            

    col1, col2, col3 = mid.columns(3)

    col1.write("Continuous Columns:")
    col1.write(continous_columns)


    col2.write("Categorical Columns:")
    col2.write(categorical_columns)


    col3.write("Discrete Columns:")
    col3.write(discrete_columns)
    
    return continous_columns, categorical_columns, discrete_columns
def Scatter_plot(df, continous_columns, categorical_columns, discrete_columns, show_legend=True):
    mid.header("Select Plot Options")
    x_column = mid.selectbox("X-axis", [None] + list(categorical_columns) + list(continous_columns))
    y_column = mid.selectbox("Y-axis", [None] + list(categorical_columns) + list(continous_columns))
    hue_column = mid.selectbox("Hue", [None] + list(categorical_columns))
    size_column = mid.selectbox("Size", [None] + list(categorical_columns))
    style_column = mid.selectbox("Style", [None] + list(categorical_columns))
    palette = mid.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    sizes_tuple = mid.slider("Select Sizes Range", min_value=0, max_value=200, value=(50, 200))
    fig, ax = plt.subplots()

    try:
        sns.scatterplot(x=x_column, y=y_column, data=df, hue=hue_column, size=size_column, style=style_column, palette=palette, sizes=sizes_tuple, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)  # Turn off legend
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting scatter plot: {str(e)}")
def Bar_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    mid.header("Select Plot Options")
    x_column = mid.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = mid.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = mid.selectbox("Hue", [None] + list(categorical_columns))
    palette = mid.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    fig, ax = plt.subplots()
    try:
        sns.barplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting bar plot: {str(e)}")
def Box_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    mid.header("Select Plot Options")
    x_column = mid.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = mid.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = mid.selectbox("Hue", [None] + list(categorical_columns))
    palette = mid.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])
    fig, ax = plt.subplots()
    try:
        sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting box plot: {str(e)}")
def Histogram(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    mid.header("Select Plot Options")
    x_column = mid.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    bins = mid.slider("Select Number of Bins", min_value=1, max_value=100, value=30)
    color = mid.color_picker("Select Color", value="#1f77b4")

    fig, ax = plt.subplots()
    try:
        sns.histplot(df[x_column], bins=bins, color=color, kde=False)
        plt.xlabel(x_column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {x_column}")
        if not show_legend:
            ax.legend().set_visible(False)
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting histogram: {str(e)}")
def Heatmap(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    mid.header("Select Plot Options")
    cmap = mid.selectbox("Select Colormap", ("viridis", "plasma", "inferno", "magma", "cividis", "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"))
    fig, ax = plt.subplots()
    numeric_columns = df.select_dtypes(include=['number']).columns
    try:
        sns.heatmap(df[numeric_columns].corr(), annot=show_legend, cmap=cmap,fmt=".1f")
        plt.title("Correlation Heatmap")
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting heatmap: {str(e)}")
def Count_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    mid.header("Select Plot Options")
    x_column = mid.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = mid.selectbox("Hue", [None] + list(categorical_columns))
    palette = mid.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        sns.countplot(x=x_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting count plot: {str(e)}")
def Pie_plot(df, continous_columns, categorical_columns, discrete_columns, show_legend=True):
    mid.header("Select Plot Options")
    column = mid.selectbox("Column", [None] + list(categorical_columns))
    palette = mid.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        counts = df[column].value_counts()
        counts.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False, colormap=palette, ax=ax)
        plt.title(f"Pie Plot of {column}")
        
        if not show_legend:
            ax.legend().set_visible(False)  # Turn off legend
            
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting pie plot: {str(e)}")
def Distplot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    mid.header("Select Plot Options")
    x_column = mid.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    color = mid.color_picker("Select Color", value="#1f77b4")
    bins_slider = mid.slider("Select Number of Bins", min_value=1, max_value=100)

    fig, ax = plt.subplots()
    try:
        sns.distplot(df[x_column], bins=bins_slider, color=color,hist_kws=dict(edgecolor="k", linewidth=.5))
        plt.xlabel(x_column)
        plt.ylabel("Frequency")
        plt.title(f"Distribution Plot of {x_column}")
        if not show_legend:
            ax.legend().set_visible(False)
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting distplot: {str(e)}")
def Line_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend):
    mid.header("Select Plot Options")
    x_column = mid.selectbox("X-axis", [None] + list(categorical_columns)+ list(continous_columns))
    y_column = mid.selectbox("Y-axis", [None] + list(categorical_columns)+ list(continous_columns))
    hue_column = mid.selectbox("Hue", [None] + list(categorical_columns))
    palette = mid.selectbox("Select Palette", ["Set1", "Set2", "Set3", "viridis", "plasma", "inferno", "magma", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"])

    fig, ax = plt.subplots()
    try:
        sns.lineplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, ax=ax)
        if not show_legend:
            ax.legend().set_visible(False)
        mid.pyplot(fig)
    except Exception as e:
        mid.error(f"Error plotting line plot: {str(e)}")
def plot_data(df):
    extender_3 = st.sidebar.expander("Plot your data")
    extender_3.header("Plot your data")
    graph = extender_3.selectbox("Select Graph", ("None","Scatter Plot", "Bar Plot", "Box Plot", "Histogram", "Heatmap", "Count Plot", "Pie Plot", "Distplot", "Line Plot"))
    split_data = extender_3.checkbox("Split data into continous and categorical columns")
    show_legend = extender_3.checkbox("Show Legend", value=True)
    continous_columns, categorical_columns, discrete_columns = np.array(df.columns), np.array(df.columns), np.array(df.columns)
    
    if split_data:    
        continous_columns, categorical_columns, discrete_columns = split_dataset(df)
    if graph == "None":
        pass
    elif graph == "Scatter Plot":
        Scatter_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Bar Plot":
        Bar_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Box Plot":
        Box_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Histogram":
        Histogram(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Heatmap":
        Heatmap(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Count Plot":
        Count_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Pie Plot":
        Pie_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Distplot":
        Distplot(df,continous_columns, categorical_columns, discrete_columns, show_legend)
    elif graph == "Line Plot":
        Line_plot(df,continous_columns, categorical_columns, discrete_columns, show_legend)


def remove_missing_values(df):
    with mid:
        try:
            st.header("Cautions: Complete Case Analysis")
            st.write("Complete case analysis (CCA), also called listwise deletion of cases, consists in discarding observations where values in any of the variables are missing. Result loss of information.")
            st.write("Assuming that the data are missing completely at random (MCAR), the complete case analysis is unbiased. However, if data are missing at random (MAR) or not at random (MNAR), then complete case analysis leads to biased results.")
            st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
            st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
            tab_rmv_miss1,tab_rmv_miss2=st.tabs(["Remove Colums","Remove Rows"])
            with tab_rmv_miss1:
                col = st.multiselect("Select the column", df.columns)
                if col:
                    df.drop(col, axis=1, inplace=True)
                    st.session_state.df = df
                    st.success(f"{col} column(s) removed successfully.")

            with tab_rmv_miss2:
                percent_missing_lowerlimit, percent_missing_upperlimit  = st.slider("Select the range of missing values: Recommended (0-5) ", 0, 100, (0, 5))
                col_missing_percent = [var for var in df.columns if df[var].isnull().mean()*100 > percent_missing_lowerlimit and df[var].isnull().mean()*100 < percent_missing_upperlimit]
                col = st.multiselect("Select the column", col_missing_percent)
                dummy = df.dropna(subset=col)
                compare_distribution(df,dummy)
                
                if st.checkbox("Confirm the removal of rows with missing values"):
                    df.dropna(subset=col, inplace=True)
                    st.session_state.df = df
                    st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
                    st.success(f"{col} column(s) removed successfully.")

        except Exception as e:
            st.error(f"Error removing missing values: {str(e)}")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def missing_values_imputation(df):
    with mid:
        st.header("Missing Values Imputation")
        st.write("Missing values imputation is the process of replacing missing data with substituted values. This section allows you to impute missing values in the dataset.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
        lg = st.checkbox("Show Legend")
        st.header("For Numerical Columns:")
        numerical_cols_withNA = df.select_dtypes(include=['number']).columns[df.select_dtypes(include=['number']).isnull().any()]
        col = st.multiselect("Select the column", numerical_cols_withNA)

        tab_num_imput1,tab_num_imput2,tab_num_imput3,tab_num_imput4,tab_num_imput5,tab_num_imput6=st.tabs(["Mean","Median","Mode","Random","End of Distribution","KNN"])

        with tab_num_imput1:
            mean_dummy = df.copy()
            st.warning("Warning: Mean imputation is sensitive to outliers, coversion and distribution.")
            for c in col:
                mean_dummy[c].fillna(df[c].mean(), inplace=True)
            compare_covariance(df,mean_dummy,col,lg)
            if st.checkbox("Confirm the mean imputation"):
                df=mean_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

        with tab_num_imput2:
            median_dummy = df.copy()
            st.warning("Warning: Median imputation is sensitive to outliers, coversion and distribution.")
            for c in col:
                median_dummy[c].fillna(df[c].median(), inplace=True)
            compare_covariance(df,median_dummy,col,lg)
            if st.checkbox("Confirm the median imputation"):
                df=median_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

        with tab_num_imput3:
            mode_dummy = df.copy()
            st.warning("Warning: Mode imputation is sensitive to outliers, coversion and distribution.")
            for c in col:
                mode_dummy[c].fillna(df[c].mode().mean(), inplace=True)
            compare_covariance(df,mode_dummy,col,lg)
            if st.checkbox("Confirm the mode imputation"):
                df=mode_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

        with tab_num_imput4:
            random_dummy = df.copy()
            for c in col:
                random_sample = df[c].dropna().sample(df[c].isnull().sum())
                random_sample.index = df[df[c].isnull()].index
                random_dummy.loc[df[c].isnull(), c] = random_sample
            compare_covariance(df, random_dummy, col,lg)
            if st.checkbox("Confirm the random imputation"):
                df=random_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

        with tab_num_imput5:
            st.write("End of Distribution")
            operation = st.selectbox("Select the column", ['None','-1','max','min','0','max + 3*sd','min - 3*sd'])
            summery_5 = pd.DataFrame([df[col].min(),df[col].quantile(0.25),df[col].quantile(0.5),df[col].quantile(0.75),df[col].max()],index=['min','Q1','Q2','Q3','max']).T
            st.write(summery_5)
            eod_dummy = df.copy()
            for c in col:
                if operation == '-1':
                    eod_dummy[c].fillna(-1, inplace=True)
                elif operation == 'max':
                    eod_dummy[c].fillna(df[c].max(), inplace=True)
                elif operation == 'min':
                    eod_dummy[c].fillna(df[c].min(), inplace=True)
                elif operation == '0':
                    eod_dummy[c].fillna(0, inplace=True)
                elif operation == 'max + 3*sd':
                    # Set values greater than max + 3*sd to max + 3*sd
                    threshold = eod_dummy[c].max() + 3 * eod_dummy[c].std()
                    eod_dummy[c].fillna(threshold, inplace=True)
                elif operation == 'min - 3*sd':
                    # Set values less than min - 3*sd to min - 3*sd
                    threshold = eod_dummy[c].min() - 3 * eod_dummy[c].std()
                    eod_dummy[c].fillna(threshold, inplace=True)
            compare_covariance(df,eod_dummy,col,lg)
            if st.checkbox("Confirm the end of distribution imputation"):
                df=eod_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

        with tab_num_imput6:
            st.write("KNN")
            neighbour = st.slider("Select the number of neighbors", 1, 10, 3)
            knn_dummy = df.copy()
            for c in col:
                imputer = KNNImputer(n_neighbors=neighbour)
                knn_dummy[c] = imputer.fit_transform(df[[c]])[:, 0]
            compare_covariance(df,knn_dummy,col,lg)
            if st.checkbox("Confirm the KNN imputation"):
                df=knn_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

        st.header("For Categorical Columns:")
        
        cat_cols_withNA = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isnull().any()]
        col_cat = st.multiselect("Select the column", cat_cols_withNA)

        tab_cat_imput1, tab_cat_imput2 = st.tabs(["Mode", "Replace with 'Missing value' tag"])

        with tab_cat_imput1:
            mode_cat_dummy = df.copy()
            st.warning("Warning: Mode imputation is sensitive to outliers, coversion, and distribution for categorical columns.")
            for c in col_cat:
                mode_cat_dummy[c].fillna(df[c].mode(), inplace=True)
            plot_categorical_distribution(df, mode_cat_dummy, col_cat,lg)
            if st.checkbox("Confirm the Categorical Mode imputation"):
                df=mode_cat_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)

        with tab_cat_imput2:
            missing_value_tag_cat_dummy = df.copy()
            for c in col_cat:
                missing_value_tag_cat_dummy[c].fillna("Missing value", inplace=True)
            plot_categorical_distribution(df, missing_value_tag_cat_dummy, col_cat,lg)
            if st.checkbox("Confirm the 'Missing' tag imputation"):
                df=missing_value_tag_cat_dummy
                st.session_state.df = df
                st.write(pd.DataFrame((df.isnull().mean() * 100).round(2).astype(str) + '%', columns=['Null values']).T)
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def outliers_detection(df):
    with mid:
        st.header("Outliers Detection")
        st.write("An outlier is a data point that differs significantly from other observations. This section allows you to detect outliers in the dataset.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        col = st.multiselect("Select the column ", df.select_dtypes(include=['number']).columns)
        tab_Zscore, tab_IQR, tab_percentile = st.tabs(["Z score", "IQR", "Percentile"])
        
        with tab_Zscore:
            st.write("Z score")
            threshold = st.slider("Select the threshold Z score", 0.0, 3.0, 1.5, step=0.01)
            z_score_dummy = df.copy()
            for c in col:
                z_scores = (z_score_dummy[c] - z_score_dummy[c].mean()) / z_score_dummy[c].std()
                z_score_dummy[c] = np.where(np.abs(z_scores) > threshold, np.nan, z_score_dummy[c])
            compare_distribution(df, z_score_dummy, col)
            if st.checkbox("Confirm the Z-score outliers detection"):
                df = z_score_dummy
                st.session_state.df = df
        
        with tab_IQR:
            st.write("IQR")
            threshold = st.slider("Select the threshold for IQR multiplication", 0.0, 3.0, 1.5, step=0.01)
            IQR_dummy = df.copy()
            IQR=IQR_dummy[c].quantile(0.75) - IQR_dummy[c].quantile(0.25)
            for c in col:
                lower_limit = IQR_dummy[c].quantile(0.25) - threshold * IQR
                upper_limit = IQR_dummy[c].quantile(0.75) + threshold * IQR
                IQR_dummy[c] = np.where((IQR_dummy[c] < lower_limit) | (IQR_dummy[c] > upper_limit), np.nan, IQR_dummy[c])
            compare_distribution(df, IQR_dummy,col)
            if st.checkbox("Confirm the IQR outliers detection"):
                df = IQR_dummy
                st.session_state.df = df

        with tab_percentile:
            st.write("Percentile")
            lower_percentile, upper_percentile = st.slider("Select the range of percentile", 0, 100, (0, 5))
            percentile_dummy = df.copy()
            for c in df.select_dtypes(include=['number']).columns:
                lower_limit = percentile_dummy[c].quantile(lower_percentile / 100)
                upper_limit = percentile_dummy[c].quantile(upper_percentile / 100)
                percentile_dummy[c] = np.where((percentile_dummy[c] < lower_limit) | (percentile_dummy[c] > upper_limit), np.nan, percentile_dummy[c])
            compare_distribution(df, percentile_dummy,col)
            if st.checkbox("Confirm the percentile outliers detection"):
                df = percentile_dummy
                st.session_state.df = df

        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
def feature_encoding(df):
    with mid:
        st.header("Feature Encoding")
        st.write("Feature encoding is the technique of converting categorical data into numerical data. This section allows you to encode features in the dataset.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        cardinality_threshold = st.slider("Cardinality: Enter the number of unique values for threshold", 1, 20, 5)
        col = df.columns[df.apply(lambda x: x.nunique()) < cardinality_threshold]

        st.write([col])
        tab_onehot, tab_ordinal, tab_frequency, tab_target = st.tabs(["One Hot", "Ordinal", "Frequency", "Target"])

        with tab_onehot:
            st.write("One-hot encoding is preferred when dealing with categorical variables with no inherent order, as it avoids introducing unintended ordinal relationships and allows machine learning models to treat each category independently, preserving the nominal nature of the data.")
            target_col = st.multiselect("Select the column for One-hot Encoding", col)

            if st.checkbox("Show One-hot encoding") and target_col:
                one_hot_dummy = df.copy()
                one_hot_dummy = pd.get_dummies(one_hot_dummy, columns=target_col, drop_first=True)

                # Filter columns that start with the selected target_col
                selected_columns = [col for col in one_hot_dummy.columns if any(col.startswith(prefix) for prefix in target_col)]

                unique_values_df = pd.DataFrame({
                    "Column": selected_columns,
                    "Unique_Values_after_One_Hot_Encoding": [one_hot_dummy[col].unique() for col in selected_columns]
                })

                # Convert the 'Unique_Values_after_One_Hot_Encoding' column values to strings
                unique_values_df['Unique_Values_after_One_Hot_Encoding'] = unique_values_df['Unique_Values_after_One_Hot_Encoding'].astype(str)

                st.write(unique_values_df)

                if st.checkbox("Confirm the One Hot encoding"):
                    df = one_hot_dummy
                    st.session_state.df = df
        with tab_ordinal:
            st.write("Ordinal encoding is preferred when dealing with categorical variables with an inherent order, as it introduces ordinal relationships and allows machine learning models to treat each category independently, preserving the ordinal nature of the data.")
            target_col = st.selectbox("Select the column for Ordinal Encoding", col)

            unique_values = df[target_col].unique().tolist()
            order = st.multiselect("Select the order of unique values", unique_values)

            if  st.checkbox("Show Ordinal encoding") and order:
                ordinal_dummy = df.copy()
                oe = OrdinalEncoder(categories=[order])
                ordinal_dummy[target_col+'_ordinal'] = oe.fit_transform(ordinal_dummy[[target_col]]).astype(int)

                unique_values_df = pd.DataFrame({
                    "Column": [col for col in ordinal_dummy.columns if col.startswith(target_col)],
                    "Unique_Values_after_Ordinal_Encoding": [ordinal_dummy[col].unique() for col in ordinal_dummy.columns if col.startswith(target_col)]
                })

                # Convert the 'Unique_Values_after_Ordinal_Encoding' column values to strings
                unique_values_df['Unique_Values_after_Ordinal_Encoding'] = unique_values_df['Unique_Values_after_Ordinal_Encoding'].astype(str)

                st.write(unique_values_df)

                if st.checkbox("Confirm the Ordinal encoding"):
                    df = ordinal_dummy
                    st.session_state.df = df
            else:
                st.warning("Please select the order of unique values for ordinal encoding.")
        with tab_frequency:
            st.write("Frequency encoding is the technique of converting categorical data into their respective frequency count offering an informative numerical representation, particularly useful for high-cardinality features.")
            target_col = st.multiselect("Select the column for Frequency Encoding", col)

            if st.checkbox("Show Frequency encoding") and target_col:
                frequency_dummy = df.copy()  # Copy the original DataFrame outside the loop

                for c in target_col:
                    frequency_encoding = frequency_dummy[c].value_counts().to_dict()
                    frequency_dummy[c+'_freq'] = frequency_dummy[c].map(frequency_encoding)

                unique_values_df = pd.DataFrame({
                    "Column": [col for col in frequency_dummy.columns if any(col.startswith(prefix) for prefix in target_col)],
                    "Unique_Values_after_Frequency_Encoding": [frequency_dummy[col].unique() for col in frequency_dummy.columns if any(col.startswith(prefix) for prefix in target_col)]
                })

                # Convert the 'Unique_Values_after_Frequency_Encoding' column values to strings
                unique_values_df['Unique_Values_after_Frequency_Encoding'] = unique_values_df['Unique_Values_after_Frequency_Encoding'].astype(str)

                st.write(unique_values_df)

                if st.checkbox("Confirm the Frequency encoding"):
                    df = frequency_dummy
                    st.session_state.df = df
        with tab_target:
            st.write("Target encoding is a technique where each category of unique value is replaced with the mean of the target variable for that category.")
            target_col = st.multiselect("Select the column for Mean Target Encoding", col)
            target = st.selectbox("Select the target column", df.columns)

            if st.checkbox("Show Mean Target encoding") and target:
                target_dummy = df.copy()

                for col in target_col:
                    target_map = target_dummy.groupby(col)[target].mean().to_dict()
                    target_dummy[col + '_target'] = target_dummy[col].map(target_map)

                unique_values_df = pd.DataFrame({
                    "Column": [col + '_target' for col in target_col],
                    "Unique_Values_after_Target_Encoding": [target_dummy[col + '_target'].unique() for col in target_col]
                })

                # Convert the 'Unique_Values_after_Target_Encoding' column values to strings
                unique_values_df['Unique_Values_after_Target_Encoding'] = unique_values_df['Unique_Values_after_Target_Encoding'].astype(str)

                st.write(unique_values_df)

                if st.checkbox("Confirm the Target encoding"):
                    df = target_dummy
                    st.session_state.df = df

def feature_scaling(df):
    st.header("Feature Scaling")
    st.write("Feature scaling is a method used to normalize the range of independent variables or features of data. This section allows you to scale features in the dataset.")
    st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)


def feature_engineering(df):
    expander_4 = st.sidebar.expander("Feature Engineering")
    with expander_4:
        st.header("Feature Engineering")
        st.write("Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. This section allows you to create new features from existing features in the dataset.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        
    if expander_4.checkbox("Remove Missing Values"):
        remove_missing_values(df)
    if expander_4.checkbox("Missing Values Imputation"):
        missing_values_imputation(df)
    if expander_4.checkbox("Outliers Detection"):
        outliers_detection(df)
    if expander_4.checkbox("Feature Encoding"):
        feature_encoding(df)
    if expander_4.checkbox("Feature Scaling"): 
        feature_scaling(df)


def feature_construction(df):
    expander_5 = st.sidebar.expander("Feature Construction")
    with expander_5:
        st.header("Feature Construction")
        st.write("An outlier is a data point that differs significantly from other observations. This section allows you to detect outliers in the dataset.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)


def feature_selection(df):
    expander_6 = st.sidebar.expander("Feature Selection")
    with expander_6:
        st.header("Feature Selection")
        st.write("Feature selection is the process of selecting a subset of relevant features for use in model construction. This section allows you to select features from the dataset.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        
        st.checkbox("Filter Methods")
        st.checkbox("Wrapper Methods")
        st.checkbox("Embedded Methods")


def feature_extraction(df):
    expander_7 = st.sidebar.expander("Feature Extraction")
    with expander_7:
        st.header("Feature Extraction")
        st.write("Feature extraction is the process of extracting features from raw data via data mining techniques. This section allows you to extract features from the dataset.")
        st.markdown("<hr style='margin: 0.2em 0;'>", unsafe_allow_html=True)
        
        st.checkbox("Principal Component Analysis (PCA)")
        st.checkbox("Linear Discriminant Analysis (LDA)")
        st.checkbox("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        st.checkbox("Autoencoders")


def app():
    # uplode file
    df = uplode_and_reset()
    
    # Section 1: Know your data
    know_data(df)
    
    # Section 2: Clean your data
    clean_data(df)
    
    # Section 3: Plot your data
    plot_data(df)
    
    # Section 4: Feature Engineering
    feature_engineering(df)
    
    # Section 5: Feature Construction 
    feature_construction(df)
    
    # Section 6: Feature Selection
    feature_selection(df)
    
    # Section 7: Feature Extraction
    feature_extraction(df)
    
    # sidebar of the page
    sidebar(df)


if __name__ == "__main__":
    mid, right_bar = st.columns([3,1])
    
    app()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)