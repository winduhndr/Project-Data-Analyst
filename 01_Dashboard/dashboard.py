import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from babel.numbers import format_currency
st.set_option('deprecation.showPyplotGlobalUse', False)

# Build a method
def create_monthly_info(df, start_date, end_date):
    fmt = df[df['order_status'] != 'canceled']
    fmt = fmt[(fmt['order_monthly'] >= start_date) & 
              (fmt['order_monthly'] <= end_date)]
    monthly_orders_df = fmt.set_index('order_monthly').resample(rule='M').agg({
        "order_id": "nunique",
        "payment_value": "sum"
    })
    monthly_orders_df.reset_index(inplace=True)
    monthly_orders_df.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    return monthly_orders_df

def create_product_info(df, start_date, end_date):
    ftp = df[df['order_status'] != 'canceled']
    ftp = ftp[(ftp['order_monthly'] >= start_date) & 
              (ftp['order_monthly'] <= end_date)]
    product_df = ftp.set_index('order_monthly').groupby('product_category_name')['product_id'].count().reset_index()
    product_df.rename(columns={
        "product_id" : "product_count"
    }, inplace=True)
    product_df = product_df.sort_values(by='product_count', ascending=False)
    return product_df

def create_correlation_info(df):
    contingency_table = pd.crosstab(df['review_score'], df['order_status'])
    chi2_stat, _, _, _ = chi2_contingency(contingency_table)
    observed_chi2 = chi2_stat / len(df)
    n_rows, n_cols = contingency_table.shape
    cramers_v = np.sqrt(observed_chi2 / min((n_cols - 1), (n_rows - 1)))
    return cramers_v, contingency_table

# Set page configuration
st.set_page_config(
    page_title="E-commerce Dashboard",
    page_icon=":bar_chart:"
)

# Load Dataset
all_df = pd.read_csv("./01_Dashboard/all_data.csv")
all_df['order_monthly'] = pd.to_datetime(all_df['order_monthly'], format='%Y-%m')

# Sidebar
with st.sidebar:
    # Title
    st.title("Windu Handaru")
    # Logo Image
    st.image("./01_Dashboard/stp.png")
    # Date Range
    min_date = all_df['order_monthly'].min()
    max_date = all_df['order_monthly'].max()
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )   

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Main
monthly_info = create_monthly_info(all_df, start_date, end_date)
product_info = create_product_info(all_df, start_date, end_date)
cramers_v, contingency_table = create_correlation_info(all_df)

main_df = all_df[(all_df["order_monthly"] >= start_date) & 
                 (all_df["order_monthly"] <= end_date)]

# Title
st.header("An E-Commerce Dashboard for Learning :bar_chart:")

# Monthly Transaction
st.subheader("Monthly Transaction")

filtered_mdf = main_df[main_df['order_status'] != "canceled"]

col1, col2 = st.columns(2)
with col1:
    total_order = filtered_mdf['order_id'].nunique()
    st.markdown(f"Total Order : **{total_order}**")

with col2:
    total_revenue = format_currency(filtered_mdf['payment_value'].sum(), "BRL", locale="pt_BR")
    st.markdown(f"Total Revenue : **{total_revenue}**")

st.altair_chart(alt.Chart(monthly_info).mark_line().encode(
    x=alt.X('order_monthly', title='Date'),
    y=alt.Y('order_count', title='Value'),
    tooltip=['order_monthly', 'order_count']
).properties(
    width=700,
    height=400
).interactive())

# Top Product
st.subheader("Top Product")

col1, col2 = st.columns(2)
with col1:
    total_product = product_info['product_count'].head(10).sum()
    st.markdown(f"Total Items : **{total_product}**")

with col2:
    avg_product = product_info["product_count"].mean()
    st.markdown(f"Average Items: **{avg_product}**")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=product_info['product_category_name'].head(10), 
            y=product_info['product_count'].head(10))

plt.title("Top Product", fontsize=15)
plt.xlabel("Product Category")
plt.ylabel("Value")
plt.xticks(rotation=45, fontsize=12)
st.pyplot(fig)

st.write("")
st.write("")

# Correlation Rating VS Total Orders
st.subheader("Correlation Rating VS Total Orders")

st.write("")

plt.figure(figsize=(6, 4))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", cbar=False)
plt.title(f"Correlation rating VS Total Orders: {cramers_v:.2f}")
plt.xlabel('Order Status')
plt.ylabel('Review Score')
plt.xticks(rotation=45, fontsize=8)
st.pyplot()

st.caption('Copyright (C) Windu Handaru. 2024')
