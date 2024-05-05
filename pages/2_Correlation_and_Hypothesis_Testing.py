import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, wilcoxon
import streamlit as st
from scipy.stats import ttest_ind, ks_2samp, wilcoxon

def t_test(group1, group2):
    # Hypothesis summary
    st.subheader("Hypothesis Summary")
    st.markdown("### Research Question")
    st.write("Is there a significant difference between the distributions of a continuous variable for individuals with and without heart disease?")
    
    st.markdown("### Null Hypothesis (H0)")
    st.write("There is no significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Alternative Hypothesis (H1)")
    st.write("There is a significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Variables")
    st.write("Group 1: Continuous Variable (Individuals with heart disease)")
    st.write("Group 2: Continuous Variable (Individuals without heart disease)")
    
    st.markdown("### Expected Relationship")
    st.write("There is no specific expectation on the direction of the relationship. We are testing for any significant difference in the distributions.")

    
    st.markdown("### Significance Level")
    st.write("Typically set at 0.05.")
    
    # Perform T-Test for independent samples
    t_statistic, p_value = ttest_ind(group1, group2)
    
    # Display T-Test information
    st.subheader("T-Test")
    st.markdown("The T-Test is a parametric test used to determine if there is a significant difference between the means of two independent samples.")
    
    summary_data = {
        "T Statistic": f"{t_statistic:.2f}",
        "P-Value": f"{p_value:.2e}"
    }
    
    if p_value < 0.05:
        st.success("The difference in means is statistically significant. Therefore, we reject the null hypothesis.")
    else:
        st.success("The difference in means is not statistically significant. Therefore, we fail to reject the null hypothesis.")
    
    st.table(summary_data)
    
    # Additional metrics
    st.subheader("Additional Metrics")
    additional_metrics_data = {
        "Metric": ["Mean", "Median", "Standard Deviation"],
        "Group 1": [f"{group1.mean():.2e}", f"{group1.median():.2e}", f"{group1.std():.2e}"],
        "Group 2": [f"{group2.mean():.2e}", f"{group2.median():.2e}", f"{group2.std():.2e}"]
    }
    st.table(additional_metrics_data)

def kolmogorov_smirnov_test(group1, group2):
    # Hypothesis summary
    st.subheader("Hypothesis Summary")
    st.markdown("### Research Question")
    st.write("Is there a significant difference between the distributions of a continuous variable for individuals with and without heart disease?")
    
    st.markdown("### Null Hypothesis (H0)")
    st.write("There is no significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Alternative Hypothesis (H1)")
    st.write("There is a significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Variables")
    st.write("Group 1: Continuous Variable (Individuals with heart disease)")
    st.write("Group 2: Continuous Variable (Individuals without heart disease)")
    
    st.markdown("### Expected Relationship")
    st.write("There is no specific expectation on the direction of the relationship. We are testing for any significant difference in the distributions.")
    
    st.markdown("### Significance Level")
    st.write("Typically set at 0.05.")
    
    # Perform Kolmogorov-Smirnov Test
    statistic, p_value = ks_2samp(group1, group2)
    
    # Display Kolmogorov-Smirnov Test information
    st.subheader("Kolmogorov-Smirnov Test")
    st.markdown("The Kolmogorov-Smirnov Test is a non-parametric test used to determine if two samples are drawn from the same distribution.")
    
    summary_data = {
        "Statistic": f"{statistic:.2f}",
        "P-Value": f"{p_value:.2e}"
    }
    
    if p_value < 0.05:
        st.success("The difference in distributions is statistically significant. Therefore, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant. Therefore, we fail to reject the null hypothesis.")
    
    st.table(summary_data)
    
    # Additional metrics
    st.subheader("Additional Metrics")
    additional_metrics_data = {
        "Metric": ["Mean", "Median", "Standard Deviation"],
        "Group 1": [f"{group1.mean():.2e}", f"{group1.median():.2e}", f"{group1.std():.2e}"],
        "Group 2": [f"{group2.mean():.2e}", f"{group2.median():.2e}", f"{group2.std():.2e}"]
    }
    st.table(additional_metrics_data)

def mann_whitney_u_test(group1, group2):
    # Hypothesis summary
    st.subheader("Hypothesis Summary")
    st.markdown("### Research Question")
    st.write("Is there a significant difference between the distributions of a continuous variable for individuals with and without heart disease?")
    
    st.markdown("### Null Hypothesis (H0)")
    st.write("There is no significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Alternative Hypothesis (H1)")
    st.write("There is a significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Variables")
    st.write("Group 1: Continuous Variable (Individuals with heart disease)")
    st.write("Group 2: Continuous Variable (Individuals without heart disease)")
    
    st.markdown("### Expected Relationship")
    st.write("There is no specific expectation on the direction of the relationship. We are testing for any significant difference in the distributions.")
    
    st.markdown("### Significance Level")
    st.write("Typically set at 0.05.")
    
    # Perform Mann-Whitney U test for two independent samples
    u_statistic, p_value = mannwhitneyu(group1, group2)
    
    # Display Mann-Whitney U test information
    st.subheader("Mann-Whitney U Test")
    st.markdown("The Mann-Whitney U test is a non-parametric test used to determine if there is a significant difference between the distributions of two independent samples.")
    
    summary_data = {
        "U Statistic": f"{u_statistic:.2f}",
        "P-Value": f"{p_value:.2e}"
    }
    
    if u_statistic < 0:
        st.success("The distribution of group 1 is less than the distribution of group 2.")
    else:
        st.success("The distribution of group 1 is greater than the distribution of group 2.")
        
    st.table(summary_data)
    
    if p_value < 0.05:
        st.success("The difference in distributions is statistically significant. Therefore, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant. Therefore, we fail to reject the null hypothesis.")
    
    # Additional metrics
    st.subheader("Additional Metrics")
    additional_metrics_data = {
        "Metric": ["Mean", "Median", "Standard Deviation"],
        "Group 1": [f"{group1.mean():.2e}", f"{group1.median():.2e}", f"{group1.std():.2e}"],
        "Group 2": [f"{group2.mean():.2e}", f"{group2.median():.2e}", f"{group2.std():.2e}"]
    }
    st.table(additional_metrics_data)

def wilcoxon_signed_rank_test(data):
    # Hypothesis summary
    st.subheader("Hypothesis Summary")
    st.markdown("### Research Question")
    st.write("Is there a significant difference between the distributions of a continuous variable for individuals with and without heart disease?")
    
    st.markdown("### Null Hypothesis (H0)")
    st.write("There is no significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Alternative Hypothesis (H1)")
    st.write("There is a significant difference between the distributions of the continuous variable for individuals with and without heart disease.")
    
    st.markdown("### Variables")
    st.write("Group 1: Continuous Variable (Individuals with heart disease)")
    st.write("Group 2: Continuous Variable (Individuals without heart disease)")
    
    st.markdown("### Expected Relationship")
    st.write("There is no specific expectation on the direction of the relationship. We are testing for any significant difference in the distributions.")
    
    st.markdown("### Significance Level")
    st.write("Typically set at 0.05.")
    
    # Perform Wilcoxon Signed-Rank Test
    statistic, p_value = wilcoxon(data)
    
    # Display Wilcoxon Signed-Rank Test information
    st.subheader("Wilcoxon Signed-Rank Test")
    st.markdown("The Wilcoxon Signed-Rank Test is a non-parametric test used to determine if there is a significant difference between paired observations.")
    
    summary_data = {
        "Statistic": f"{statistic:.2f}",
        "P-Value": f"{p_value:.2e}"
    }
    
    if p_value < 0.05:
        st.success("The difference in paired observations is statistically significant. Therefore, we reject the null hypothesis.")
    else:
        st.success("The difference in paired observations is not statistically significant. Therefore, we fail to reject the null hypothesis.")
    
    st.table(summary_data)
    
    st.subheader("Additional Metrics")
    additional_metrics_data = {
        "Metric": ["Mean", "Median", "Standard Deviation"],
        "Group 1": [f"{group1.mean():.2e}", f"{group1.median():.2e}", f"{group1.std():.2e}"],
        "Group 2": [f"{group2.mean():.2e}", f"{group2.median():.2e}", f"{group2.std():.2e}"]
    }
    st.table(additional_metrics_data)


def chi_square_test(confusion_matrix):
    # Perform Chi-Square Test
    chi2, p_value, dof, ex = chi2_contingency(confusion_matrix)

    # Display Chi-Square Test information
    st.subheader("Chi-Square Test")

    # Hypothesis Summary
    st.markdown("### Hypothesis Summary")
    st.markdown("#### Research Question")
    st.write("Is there a significant association between two categorical variables?")
    
    st.markdown("#### Null Hypothesis (H0)")
    st.write("There is no significant association between the two categorical variables.")
    
    st.markdown("#### Alternative Hypothesis (H1)")
    st.write("There is a significant association between the two categorical variables.")
    
    st.markdown("#### Variables")
    st.write("Variable 1: Categorical Variable 1")
    st.write("Variable 2: Categorical Variable 2")

    # Summary data
    summary_data = {
        "Chi-Square Statistic": f"{chi2:.2e}",
        "P-Value": f"{p_value:.2e}",
        "Degrees of Freedom": dof
    }
    st.table(summary_data)

    # Interpretation
    if p_value < 0.05:
        st.success("The difference in distributions is statistically significant. Therefore, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant. Therefore, we fail to reject the null hypothesis.")



def fisher_exact_test(confusion_matrix):
    # Perform Fisher's Exact Test
    odds_ratio, p_value = fisher_exact(confusion_matrix)
    
    # Display Fisher's Exact Test information
    st.subheader("Fisher's Exact Test")
    st.markdown("Fisher's exact test is used when the sample size is small. It tests the independence between two categorical variables.")
    
    # Hypothesis Summary
    st.markdown("### Hypothesis Summary")
    st.markdown("#### Research Question")
    st.write("Is there a significant association between two categorical variables?")
    
    st.markdown("#### Null Hypothesis (H0)")
    st.write("There is no significant association between the two categorical variables.")
    
    st.markdown("#### Alternative Hypothesis (H1)")
    st.write("There is a significant association between the two categorical variables.")
    
    st.markdown("#### Variables")
    st.write("Variable 1: Categorical Variable 1")
    st.write("Variable 2: Categorical Variable 2")

    # Summary data
    summary_data = {
        "Odds Ratio": f"{odds_ratio:.2e}",
        "P-Value": f"{p_value:.2e}"
    }
    st.table(summary_data)

    # Interpretation
    if p_value < 0.05:
        st.success("The difference in distributions is statistically significant. Therefore, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant. Therefore, we fail to reject the null hypothesis.")


def odds_ratio(confusion_matrix):
    # Compute chi-square test for odds ratio
    _, p_value, _, _ = chi2_contingency(confusion_matrix)
    
    # Display Odds Ratio information
    st.subheader("Odds Ratio")
    st.markdown("The odds ratio is a measure of association between two categorical variables. It quantifies the odds of an event occurring in one group compared to another.")
    
    # Assuming the columns in confusion_matrix represent the groups
    a, b = confusion_matrix.iloc[0]
    c, d = confusion_matrix.iloc[1]
    
    # Calculate odds ratio if possible
    if a == 0 or b == 0 or c == 0 or d == 0:
        st.write("The odds ratio cannot be calculated because one of the cells in the contingency table is 0.")
    else:
        odds_ratio = (a * d) / (b * c)
    
        # Summary data
        summary_data = {
            "Odds Ratio": f"{odds_ratio:.2e}",
            "P-Value": f"{p_value:.2e}"
        }
        st.table(summary_data)
        
        # Interpretation
        if p_value < 0.05:
            st.success("The difference in distributions is statistically significant. Therefore, we reject the null hypothesis.")
        else:
            st.success("The difference in distributions is not statistically significant. Therefore, we fail to reject the null hypothesis.")


def qualitative_analysis(df):
    with mid:
        st.subheader("Categorical Analysis")
        
        bool_col = df.select_dtypes(include='bool').columns
        num_col = df.select_dtypes(include=['number']).columns
        non_bool_num_col = [col for col in num_col if df[col].dtype != 'bool']

        target_bool = st.selectbox("Select Feature of Interest", bool_col)
        show_df = pd.DataFrame(df[target_bool])

        fig = px.histogram(show_df,x=target_bool)
        st.plotly_chart(fig)

        # st.subheader("Mosaic Plot")
        # mosaic_fig = px.histogram(show_df, x=bool_col, color=target_bool, marginal="mosaic",
        #                             title=f"Mosaic Plot ({target_bool} vs {bool_col})")
        # st.plotly_chart(mosaic_fig)
        
        # summerize of the data
        count = df.shape[0]
        proportion_of_ones = show_df[target_bool].mean()
        proportion_of_zeros = 1 - proportion_of_ones

        summary_data = {
            'Count': [count],
            'Percentage of Ones': ["%.2f%%" % (proportion_of_ones * 100)],
            'Percentage of Zeros': ["%.2f%%" % (proportion_of_zeros * 100)]
        }

        summary_df = pd.DataFrame(summary_data)
        st.subheader("Summary of the Distribution of the Feature of Interest:")
        st.table(summary_df)

        Qualitative1, Quantitative1 = st.tabs(["Analise with Categorical feature", "Analise with Continuous feature"])
        
        with Qualitative1:
            st.subheader("Categorical vs Categorical")
            bool_feature = st.selectbox("Select the feature for analysis", bool_col)
            
            st.subheader("Distribution")
            fig = px.histogram(df, x=bool_feature, color=target_bool,labels={'color': target_bool},title=f"Distribution of {bool_feature} by {target_bool}")
            st.plotly_chart(fig)
            
            st.subheader("")
            
            
            
            
            
            
            st.subheader("Cotigency Table")
            confusion_matrix = pd.crosstab(df[target_bool], df[bool_feature])
            
            confusion_matrix.index = [f'{target_bool}-{idx}' for idx in confusion_matrix.index]
            confusion_matrix.columns = [f'{bool_feature}-{col}' for col in confusion_matrix.columns]
            
            styled_matrix = confusion_matrix.style.highlight_max(axis=0, color='lightgreen').highlight_max(axis=1, color='lightblue').set_table_styles([{'selector': 'thead th','props': [('text-align', 'center')]}])
            st.table(styled_matrix.set_caption(f"Confusion Matrix ({target_bool} vs {bool_feature})").set_table_attributes("class='styled-table'"))
            
            hypothesis_testing = st.selectbox("Select Hypothesis Testing", ['None', "Chi-Square Test", "Fisher's Exact Test", "Odd Ratio"])
            
            if hypothesis_testing == 'Chi-Square Test':
                chi_square_test(confusion_matrix)
            elif hypothesis_testing == "Fisher's Exact Test":
                fisher_exact_test(confusion_matrix)
            elif hypothesis_testing == 'Odd Ratio':
                odds_ratio(confusion_matrix)


        with Quantitative1:  
            st.subheader("Categorical vs Continuous")
            num_feature = st.selectbox("Select the feature for analysis", non_bool_num_col)
            
            # Plot Density Plot
            st.subheader("Density Plot")
            fig_density = px.histogram(df, x=num_feature, color=target_bool, marginal='rug', labels={'color': target_bool}, title=f"Distribution of {num_feature} by {target_bool}")
            st.plotly_chart(fig_density)
            
            st.subheader("5-Number Summary")

            # Function to calculate 5-number summary
            def five_number_summary(data):
                minimum = np.min(data)
                q1 = np.percentile(data, 25)
                median = np.percentile(data, 50)
                q3 = np.percentile(data, 75)
                maximum = np.max(data)
                return minimum, q1, median, q3, maximum
            
            # Calculate summary for each category
            summary_data = []
            for category, group_data in df.groupby(target_bool)[num_feature]:
                summary_data.append([category] + list(five_number_summary(group_data)))

            summary_df = pd.DataFrame(summary_data, columns=[target_bool, "Minimum", "1st Quartile", "Median", "3rd Quartile", "Maximum"])
            st.write(summary_df)
            
            hypothesis_testing = st.selectbox("Select Hypothesis Testing", ['T-Test', "Kolmogorov-Smirnov Test", "Mann-Whitney U Test"])
            
            if hypothesis_testing == 'T-Test':
                group1 = df[df[target_bool] == True][num_feature]
                group2 = df[df[target_bool] == False][num_feature]
                t_test(group1, group2)
            elif hypothesis_testing == "Kolmogorov-Smirnov Test":
                group1 = df[df[target_bool] == True][num_feature]
                group2 = df[df[target_bool] == False][num_feature]
                kolmogorov_smirnov_test(group1, group2)
            elif hypothesis_testing == "Mann-Whitney U Test":
                group1 = df[df[target_bool] == True][num_feature]
                group2 = df[df[target_bool] == False][num_feature]
                mann_whitney_u_test(group1, group2)
            # elif hypothesis_testing == "Wilcoxon Signed-Rank Test":
            #     data = df[num_feature]
            #     wilcoxon_signed_rank_test(data)
            
        st.markdown('---')


def quantitative_analysis(df):
    with mid:
        Quantitative2 = st.subheader("Continuous Analysis")
        
        num_col = df.select_dtypes(include=['number']).columns
        non_bool_num_col = [col for col in num_col if df[col].dtype != 'bool']
        
        target_quant = st.selectbox("Select Target Variable", non_bool_num_col)
        show_df_quantitative = pd.DataFrame(df[target_quant])

        fig_quantitative = px.histogram(show_df_quantitative, x=target_quant)
        st.plotly_chart(fig_quantitative)

        count_quantitative = df.shape[0]
        mean_quantitative = show_df_quantitative[target_quant].mean()
        std_quantitative = show_df_quantitative[target_quant].std()
        Q1 = show_df_quantitative[target_quant].quantile(0.25)
        Q2 = show_df_quantitative[target_quant].quantile(0.50)
        Q3 = show_df_quantitative[target_quant].quantile(0.75)
        IQR = Q3 - Q1

        summary_data_quantitative = {
            'Count': [count_quantitative],
            'Mean': ["%.2f" %mean_quantitative],
            'Standard Deviation': ["%.2f" %std_quantitative],
            'Q1': ["%.2f" %Q1],
            'Q2': ["%.2f" %Q2],
            'Q3': ["%.f" %Q3],
            'IQR': ["%.2f" %IQR]
        }

        summary_df_quantitative = pd.DataFrame(summary_data_quantitative)
        st.subheader("Summary Statistics for the Continuous Variable:")
        st.table(summary_df_quantitative)
        Qualitative2,Quantitative2 = st.tabs(["Analise with Categorical feature", "Analise with Continuous feature"])
        
        
        with Quantitative2:
            st.subheader("Continuous vs Categorical")
            bool_feature = st.selectbox("Select the Continuous feature for analysis", non_bool_num_col)
            
            # Plot Box Plot
            st.subheader("Box Plot")
            fig_box_quant_qual = px.box(df, x=bool_feature, y=target_quant, labels={'y': target_quant, 'color': bool_feature}, title=f"Box Plot of {target_quant} by {bool_feature}")
            st.plotly_chart(fig_box_quant_qual)
            
            hypothesis_testing_quant_qual = st.selectbox("Select Hypothesis Testing", ['T-Test', "Kolmogorov-Smirnov Test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"])
            
            if hypothesis_testing_quant_qual == 'T-Test':
                group1_quant_qual = df[df[bool_feature] == True][target_quant]
                group2_quant_qual = df[df[bool_feature] == False][target_quant]
                t_test(group1_quant_qual, group2_quant_qual)
            elif hypothesis_testing_quant_qual == "Kolmogorov-Smirnov Test":
                group1_quant_qual = df[df[bool_feature] == True][target_quant]
                group2_quant_qual = df[df[bool_feature] == False][target_quant]
                kolmogorov_smirnov_test(group1_quant_qual, group2_quant_qual)
            elif hypothesis_testing_quant_qual == "Mann-Whitney U Test":
                group1_quant_qual = df[df[bool_feature] == True][target_quant]
                group2_quant_qual = df[df[bool_feature] == False][target_quant]
                mann_whitney_u_test(group1_quant_qual, group2_quant_qual)
            # elif hypothesis_testing_quant_qual == "Wilcoxon Signed-Rank Test":
            #     data_quant_qual = df[target_quant]
            #     wilcoxon_signed_rank_test(data_quant_qual)
            # elif hypothesis_testing == 'Chi-Square Test':
            #     chi_square_test(confusion_matrix)
        
        
def side_bar(df):
    with st.sidebar:
        if st.checkbox("Categorical Analysis"):
            qualitative_analysis(df)
        if st.checkbox("Continuous Analysis"):
            quantitative_analysis(df)


def app():
    # over write df if no file is apploded
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  
    else:
        df = pd.read_csv('data/lifestyle.csv')
        st.sidebar.markdown("Default DataFrame is loaded:""Sleep Health and Lifestyle [Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)")

    side_bar(df)


if __name__ == "__main__":
    st.set_page_config(layout="wide",initial_sidebar_state="auto")
    
    st.title("Correlation and Hypothesis Testing")
    mid,_ = st.columns([100,1])
    app()

