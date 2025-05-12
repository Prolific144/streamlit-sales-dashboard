import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from uuid import uuid4

# Function to load and clean data
def load_and_clean_data(file_path):
    try:
        df = pd.read_excel(file_path)  # Read Excel file
        
        # Clean data
        df = df.dropna()  # Remove rows with missing values
        df = df.drop_duplicates()  # Remove duplicates
        
        # Convert numeric columns
        numeric_cols = ['Selling_Price', 'Items_Sold', 'Revenue']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid numeric values
        df = df.dropna(subset=numeric_cols)
        
        # Ensure correct data types
        df['Main_category'] = df['Main_category'].astype(str)
        df['Sub-category'] = df['Sub-category'].astype(str)
        df['BRAND'] = df['BRAND'].astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to aggregate data by category
def aggregate_by_category(df, group_by_cols):
    agg_df = df.groupby(group_by_cols).agg({
        'Revenue': 'sum',
        'Items_Sold': 'sum',
        'Selling_Price': 'mean'
    }).reset_index()
    agg_df['Average_Price'] = agg_df['Selling_Price'].round(2)
    return agg_df

# Function to find top products by revenue
def get_top_products(df, n=10):
    return df.nlargest(n, 'Revenue')[['Product_Name', 'Revenue', 'Items_Sold', 'Selling_Price']]

# Function to find top brands by revenue
def get_top_brands(df, n=5):
    brand_agg = df.groupby('BRAND').agg({
        'Revenue': 'sum',
        'Items_Sold': 'sum'
    }).reset_index()
    return brand_agg.nlargest(n, 'Revenue')

# Function to calculate revenue-to-items-sold ratio
def calculate_revenue_per_item(df):
    df['Revenue_per_Item'] = df['Revenue'] / df['Items_Sold']
    return df.nlargest(1, 'Revenue_per_Item')[['Product_Name', 'Revenue_per_Item', 'Revenue', 'Items_Sold']]

# Function to create bar plot for revenue by category
def plot_revenue_by_category(df, group_by_cols, title):
    agg_df = aggregate_by_category(df, group_by_cols)
    fig = px.bar(
        agg_df,
        x=group_by_cols[-1],
        y='Revenue',
        color=group_by_cols[0] if len(group_by_cols) > 1 else None,
        title=title,
        labels={'Revenue': 'Total Revenue (KES)', group_by_cols[-1]: group_by_cols[-1].replace('_', ' ')},
        text_auto='.2s'
    )
    fig.update_layout(xaxis_title=group_by_cols[-1].replace('_', ' '), yaxis_title='Total Revenue (KES)')
    return fig

# Function to create pie chart for revenue distribution
def plot_revenue_distribution(df, category_col, title):
    agg_df = aggregate_by_category(df, [category_col])
    fig = px.pie(
        agg_df,
        values='Revenue',
        names=category_col,
        title=title,
        hole=0.3
    )
    fig.update_traces(textinfo='percent+label')
    return fig

# Function to create scatter plot for price vs items sold
def plot_price_vs_items_sold(df, title):
    fig = px.scatter(
        df,
        x='Selling_Price',
        y='Items_Sold',
        color='Main_category',
        size='Revenue',
        hover_data=['Product_Name', 'BRAND'],
        title=title,
        labels={'Selling_Price': 'Selling Price (KES)', 'Items_Sold': 'Items Sold'}
    )
    fig.update_layout(xaxis_title='Selling Price (KES)', yaxis_title='Items Sold')
    return fig

# Function to create bar plot for top products
def plot_top_products(df, n=10, title="Top 10 Products by Revenue"):
    top_df = get_top_products(df, n)
    fig = px.bar(
        top_df,
        x='Product_Name',
        y='Revenue',
        title=title,
        labels={'Revenue': 'Total Revenue (KES)', 'Product_Name': 'Product'},
        text_auto='.2s'
    )
    fig.update_layout(xaxis_title='Product', yaxis_title='Total Revenue (KES)', xaxis_tickangle=45)
    return fig

# Function to create bar plot for top brands
def plot_top_brands(df, n=5, title="Top 5 Brands by Revenue"):
    top_brands = get_top_brands(df, n)
    fig = px.bar(
        top_brands,
        x='BRAND',
        y='Revenue',
        title=title,
        labels={'Revenue': 'Total Revenue (KES)', 'BRAND': 'Brand'},
        text_auto='.2s'
    )
    fig.update_layout(xaxis_title='Brand', yaxis_title='Total Revenue (KES)')
    return fig

# Main Streamlit app
def main():
    st.set_page_config(page_title="Sales Data Dashboard", layout="wide")
    st.title("📊 Sales Data Dashboard")

    # Load data
    file_path = "Python\Analysis\jumia\data\salesdata.xlsx"  # Adjust path if needed
    df = load_and_clean_data(file_path)

    if df is None:
        st.stop()

    # Sidebar for filtering
    st.sidebar.header("Filters")
    main_categories = ['All'] + sorted(df['Main_category'].unique())
    selected_category = st.sidebar.selectbox("Select Main Category", main_categories)

    # Filter data
    if selected_category != 'All':
        df = df[df['Main_category'] == selected_category]

    # Key metrics
    total_revenue = df['Revenue'].sum()
    total_items_sold = df['Items_Sold'].sum()
    avg_price = df['Selling_Price'].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue (KES)", f"{total_revenue:,.2f}")
    with col2:
        st.metric("Total Items Sold", f"{total_items_sold:,}")
    with col3:
        st.metric("Average Price (KES)", f"{avg_price:,.2f}")

    # Visualizations
    st.header("Revenue Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = plot_revenue_by_category(df, ['Main_category', 'Sub-category'], "Revenue by Category and Sub-category")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = plot_revenue_distribution(df, 'Main_category', "Revenue Distribution by Main Category")
        st.plotly_chart(fig2, use_container_width=True)

    st.header("Product and Pricing Insights")
    col1, col2 = st.columns(2)

    with col1:
        fig3 = plot_price_vs_items_sold(df, "Selling Price vs. Items Sold")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = plot_top_products(df, 10, "Top 10 Products by Revenue")
        st.plotly_chart(fig4, use_container_width=True)

    st.header("Brand Performance")
    fig5 = plot_top_brands(df, 5, "Top 5 Brands by Revenue")
    st.plotly_chart(fig5, use_container_width=True)

    # Interesting Fact
    st.header("Interesting Fact")
    top_ratio = calculate_revenue_per_item(df)
    product_name = top_ratio['Product_Name'].iloc[0]
    revenue_per_item = top_ratio['Revenue_per_Item'].iloc[0]
    st.write(
        f"The product **{product_name}** has the highest revenue-to-items-sold ratio at **{revenue_per_item:,.2f} KES per item**, "
        f"indicating it generates significant revenue per unit sold!"
    )

if __name__ == "__main__":
    main()