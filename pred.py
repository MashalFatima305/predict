#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Updated data for confusion matrices
confusion_matrices = {
    'OLS': [
        [-1.0, 0.0, 1.0, 2.0, 3.0, 197142903.0],
        [-1.0, 0, 0, 0, 0, 0],
        [0.0, 5, 1137, 170, 1, 0],
        [1.0, 1, 10, 36, 2, 1],
        [2.0, 0, 0, 0, 0, 0],
        [3.0, 0, 0, 0, 0, 0],
        [197142903.0, 0, 0, 0, 0, 0]
    ],
    'Ridge': [
        [0, 0, 0, 0, 0, 0],
        [5, 1129, 179, 0, 0, 0],
        [0, 7, 41, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ],
    'Lasso': [
        [826, 487, 0],
        [20, 30, 1],
        [0, 0, 0]
    ],
    'RFE Logistic Regression': [
        [726, 4],
        [587, 47]
    ],
    'Polynomial Logistic Regression': [
        [1106, 207],
        [30, 21]
    ]
}

# Updated metrics based on your classification reports
metrics = {
    'Model': ['OLS', 'Ridge', 'Lasso', 'RFE Logistic Regression', 'Polynomial Logistic'],
    'Accuracy': [0.86, 0.86, 0.63, 0.57, 0.83],
    'Recall': [0.86, 0.86, 0.63, 0.57, 0.83],
    'Precision': [0.96, 0.96, 0.94, 0.96, 0.94],
    'F1 Score': [0.90, 0.90, 0.74, 0.69, 0.88]
}
df_metrics = pd.DataFrame(metrics)

# Function to display confusion matrices as tables
def display_confusion_matrix_table(cm, model_name):
    try:
        cm_df = pd.DataFrame(cm)
        st.write(f'Confusion Matrix for {model_name}')
        st.table(cm_df)
    except Exception as e:
        st.error(f"Error displaying confusion matrix for {model_name}: {e}")

# Function to plot bar graphs using Altair
def plot_bar_graph(metric):
    try:
        bar_chart = alt.Chart(df_metrics).mark_bar().encode(
            x='Model',
            y=metric,
            color='Model'
        ).properties(
            title=f'{metric} Comparison'
        )
        st.altair_chart(bar_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting bar graph for {metric}: {e}")

# Function to plot histograms using Altair
def plot_histogram(data, title):
    try:
        df = pd.DataFrame({'values': data})
        hist = alt.Chart(df).mark_bar().encode(
            alt.X('values:Q', bin=alt.Bin(maxbins=30)),
            y='count()'
        ).properties(
            title=title
        )
        st.altair_chart(hist, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting histogram for {title}: {e}")

# Streamlit app layout
st.title('Bankruptcy Prediction Model Comparison')

# Display confusion matrices as tables
for model_name, cm in confusion_matrices.items():
    display_confusion_matrix_table(cm, model_name)

# Display bar graphs for metrics
metrics_list = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
for metric in metrics_list:
    plot_bar_graph(metric)

# Example data for histograms
hist_data = {
    'Model Accuracy Comparison': [0.86, 0.86, 0.63, 0.57, 0.83],
    'Model Recall Comparison': [0.86, 0.86, 0.63, 0.57, 0.83],
    'Model Precision Comparison': [0.96, 0.96, 0.94, 0.96, 0.94],
    'Model F1 Score Comparison': [0.90, 0.90, 0.74, 0.69, 0.88]
}

# Display histograms
for title, data in hist_data.items():
    plot_histogram(data, title)


# In[ ]:




