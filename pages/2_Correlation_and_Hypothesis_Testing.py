import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, wilcoxon

def t_test(group1, group2):
    # Perform independent two-sample t-test
    t_statistic, p_value = ttest_ind(group1, group2)
    
    # Display t-test information
    st.subheader("Two-Sample t-Test")
    st.markdown("The two-sample t-test is used to determine if there is a significant difference between the means of two independent groups.")
    
    summary_data = {"Parameter": ["T-Statistic", "P-Value"],
                    "Value": [f"{t_statistic:.2f}", f"{p_value:.2e}"]
                    }
    if t_statistic < 0:
        st.success("The mean of group 1 is less than the mean of group 2.")
    else:
        st.success("The mean of group 1 is greater than the mean of group 2.")
    
    st.table(summary_data)
    if p_value < 0.05:
        st.success("The difference in means is statistically significant.So, we reject the null hypothesis.")
    else:
        st.success("The difference in means is not statistically significant.So, we fail to reject the null hypothesis.")
        

def kolmogorov_smirnov_test(group1, group2):
    # Perform Kolmogorov-Smirnov test for two independent samples
    ks_statistic, p_value = ks_2samp(group1, group2)
    
    # Display KS test information
    st.subheader("Kolmogorov-Smirnov Test")
    st.markdown("The Kolmogorov-Smirnov test is used to compare the distributions of two independent samples.")
    
    summary_data = {"Parameter": ["KS Statistic", "P-Value"],
                    "Value": [f"{ks_statistic:.2f}", f"{p_value:.2e}"]
                    }
    if ks_statistic < 0:
        st.success("The distribution of group 1 is less than the distribution of group 2.")
    else:
        st.success("The distribution of group 1 is greater than the distribution of group 2.")
    st.table(summary_data)
    if p_value < 0.05:
        st.success("The difference in distributions is statistically significant.So, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant.So, we fail to reject the null hypothesis.")
def mann_whitney_u_test(group1, group2):
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
        st.success("The difference in distributions is statistically significant.So, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant.So, we fail to reject the null hypothesis.")
def wilcoxon_signed_rank_test(data):
    # Perform Wilcoxon signed-rank test for paired samples
    _, p_value = wilcoxon(data)
    
    # Display Wilcoxon signed-rank test information
    st.subheader("Wilcoxon Signed-Rank Test")
    st.markdown("The Wilcoxon signed-rank test is used to determine if there is a significant difference between paired samples.")
    
    summary_data = {
        "P-Value": f"{p_value:.2e}"
    }
    st.table(summary_data)
    if p_value < 0.05:
        st.success("The difference in distributions is statistically significant.So, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant.So, we fail to reject the null hypothesis.")
def chi_square_test(confusion_matrix):
    chi2, p, dof, ex = chi2_contingency(confusion_matrix)

    # Display the Chi-Square Test information
    st.subheader("Chi-Square Test")
    st.markdown("The chi-square test of independence is used to determine if there is a significant association between two categorical variables. The test compares the observed distribution of the variables to an expected distribution if the variables were independent.")
    summary_data = {
        "Chi-Square": f"{chi2:.2f}",
        "P-Value": f"{p:.2e}",
        "Degrees of Freedom": f"{dof}"
    }
    if chi2 < 0:
        st.success("The distribution of group 1 is less than the distribution of group 2.")
    else:
        st.success("The distribution of group 1 is greater than the distribution of group 2.")
        
    st.table(summary_data)
    if p < 0.05:
        st.success("The difference in distributions is statistically significant.So, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant.So, we fail to reject the null hypothesis.")

    # Display Expected Frequencies
    st.subheader("Expected Frequencies")
    st.markdown("Expected frequencies are the values that would be expected in each cell of the contingency table if there were no association between the variables. These values are based on the assumption of independence.")

    st.subheader("Contingency Table")
    st.table(ex)
def fisher_exact_test(confusion_matrix):
    odds_ratio, p_value = fisher_exact(confusion_matrix)
    
    # Display Fisher's Exact Test information
    st.subheader("Fisher's Exact Test")
    st.markdown("Fisher's exact test is used when the sample size is small. It tests the independence between two categorical variables.")
    
    summary_data = {
        "Odds Ratio": f"{odds_ratio:.2f}",
        "P-Value": f"{p_value:.2e}"
    }
    
    if odds_ratio < 0:
        st.success("The distribution of group 1 is less than the distribution of group 2.")
    else:
        st.success("The distribution of group 1 is greater than the distribution of group 2.")
    
    st.table(summary_data)
    
    if p_value < 0.05:
        st.success("The difference in distributions is statistically significant.So, we reject the null hypothesis.")
    else:
        st.success("The difference in distributions is not statistically significant.So, we fail to reject the null hypothesis.")

def odds_ratio(confusion_matrix):
    _, p_value, _, _ = chi2_contingency(confusion_matrix)
    
    # Display Odds Ratio information
    st.subheader("Odds Ratio")
    st.markdown("The odds ratio is a measure of association between two categorical variables. It quantifies the odds of an event occurring in one group compared to another.")
    
    # Assuming the columns in confusion_matrix represent the groups
    a, b = confusion_matrix.iloc[0]
    c, d = confusion_matrix.iloc[1]
    if a == 0 or b == 0 or c == 0 or d == 0:
        st.write("The odds ratio cannot be calculated because one of the cells in the contingency table is 0.")
        
    else:
        odds_ratio = (a * d) / (b * c)
    
        summary_data = {
            "Odds Ratio": f"{odds_ratio:.2f}",
            "P-Value": f"{p_value:.2e}"
        }
        if odds_ratio < 0:
            st.success("The distribution of group 1 is less than the distribution of group 2.")
        else:
            st.success("The distribution of group 1 is greater than the distribution of group 2.") 
        
        st.table(summary_data)
        
        if p_value < 0.05:
            st.success("The difference in distributions is statistically significant.So, we reject the null hypothesis.")
        else:
            st.success("The difference in distributions is not statistically significant.So, we fail to reject the null hypothesis.")


def qualitative_analysis(df):
    with mid:
        st.subheader("Qualitative Analysis")
        
        bool_col = df.select_dtypes(include='bool').columns
        num_col = df.select_dtypes(include=['number']).columns
        non_bool_num_col = [col for col in num_col if df[col].dtype != 'bool']

        target_bool = st.selectbox("Select Feature of Interest", bool_col)
        show_df = pd.DataFrame(df[target_bool])

        fig = px.histogram(show_df,x=target_bool)
        st.plotly_chart(fig)

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

        Qualitative1, Quantitative1 = st.tabs(["Analise with Qualitative feature", "Analise with Qualitative feature"])
        
        with Qualitative1:
            st.subheader("Qualitative vs Qualitative")
            bool_feature = st.selectbox("Select the feature for analysis", bool_col)
            
            # Plot Histogram
            st.subheader("Distribution")
            fig = px.histogram(df, x=bool_feature, color=target_bool, 
                            labels={'color': target_bool},
                            title=f"Distribution of {bool_feature} by {target_bool}")
            st.plotly_chart(fig)
            
            st.subheader("Confusion Matrix")
            
            confusion_matrix = pd.crosstab(df[target_bool], df[bool_feature])
            
            # Replace 'True' and 'False' with column names
            confusion_matrix.index = [f'{target_bool}-{idx}' for idx in confusion_matrix.index]
            confusion_matrix.columns = [f'{bool_feature}-{col}' for col in confusion_matrix.columns]
            
            styled_matrix = confusion_matrix.style.highlight_max(axis=0, color='lightgreen').highlight_max(axis=1, color='lightblue').set_table_styles([{'selector': 'thead th','props': [('text-align', 'center')]}])
            st.table(styled_matrix.set_caption(f"Confusion Matrix ({target_bool} vs {bool_feature})").set_table_attributes("class='styled-table'"))
            
            hypothesis_testing = st.selectbox("Select Hypothesis Testing", ['Chi-Square Test', "Fisher's Exact Test","Odd Ratio"])
            
            if hypothesis_testing == 'Chi-Square Test':
                chi_square_test(confusion_matrix)
            elif hypothesis_testing == "Fisher's Exact Test":
                fisher_exact_test(confusion_matrix)
            elif hypothesis_testing == 'Odd Ratio':
                odds_ratio(confusion_matrix)


        with Quantitative1:  
            st.subheader("Qualitative vs Quantitative")
            num_feature = st.selectbox("Select the feature for analysis", non_bool_num_col)
            
            # Plot Density Plot
            st.subheader("Density Plot")
            fig_density = px.histogram(df, x=num_feature, color=target_bool, marginal='rug', labels={'color': target_bool}, title=f"Distribution of {num_feature} by {target_bool}")
            st.plotly_chart(fig_density)
            
            # Plot Box Plot
            st.subheader("Box Plot")
            fig_box = px.box(df, x=target_bool, y=num_feature, labels={'y': num_feature, 'color': target_bool}, title=f"Box Plot of {num_feature} by {target_bool}")
            st.plotly_chart(fig_box)
            
            hypothesis_testing = st.selectbox("Select Hypothesis Testing", ['T-Test', "Kolmogorov-Smirnov Test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"])
            
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
            elif hypothesis_testing == "Wilcoxon Signed-Rank Test":
                data = df[num_feature]
                wilcoxon_signed_rank_test(data)
            
        st.markdown('---')


def quantitative_analysis(df):
    with mid:
        Quantitative2 = st.subheader("Quantitative Analysis")
        
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
        st.subheader("Summary Statistics for the Quantitative Variable:")
        st.table(summary_df_quantitative)
        Qualitative2,Quantitative2 = st.tabs(["Analise with Qualitative feature", "Analise with Quantitative feature"])
        
        
        with Quantitative2:
            st.subheader("Quantitative vs Qualitative")
            bool_feature = st.selectbox("Select the quantitative feature for analysis", non_bool_num_col)
            
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
            elif hypothesis_testing_quant_qual == "Wilcoxon Signed-Rank Test":
                data_quant_qual = df[target_quant]
                wilcoxon_signed_rank_test(data_quant_qual)
        
        
def side_bar(df):
    with st.sidebar:
        if st.checkbox("Qualitative Analysis"):
            qualitative_analysis(df)
        if st.checkbox("Quantitative Analysis"):
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

