import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os

# Function to load and clean data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_and_clean_data(file_path=None, uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel(file_path)
        
        # Clean data
        df = df.dropna()
        df = df.drop_duplicates()
        
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
        
        # Add date-related columns if a date column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month_name()
            df['Day_of_Week'] = df['Date'].dt.day_name()
            df['Quarter'] = df['Date'].dt.quarter
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to calculate revenue-to-items-sold ratio (used in insights)
def calculate_revenue_per_item(df):
    df['Revenue_per_Item'] = df['Revenue'] / df['Items_Sold']
    return df.nlargest(1, 'Revenue_per_Item')[['Product_Name', 'Revenue_per_Item', 'Revenue', 'Items_Sold']]

# Function to aggregate data by category
def aggregate_by_category(df, group_by_cols):
    agg_df = df.groupby(group_by_cols).agg({
        'Revenue': ['sum', 'mean', 'count'],
        'Items_Sold': 'sum',
        'Selling_Price': ['mean', 'median']
    }).reset_index()
    
    # Flatten multi-index columns
    agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns]
    
    agg_df = agg_df.rename(columns={
        'Revenue_sum': 'Revenue',
        'Revenue_mean': 'Avg_Revenue_per_Item',
        'Revenue_count': 'Transaction_Count',
        'Items_Sold_sum': 'Items_Sold',
        'Selling_Price_mean': 'Avg_Price',
        'Selling_Price_median': 'Median_Price'
    })
    
    return agg_df

# Function to find top products
def get_top_products(df, n=10, metric='Revenue'):
    return df.nlargest(n, metric)[['Product_Name', 'Revenue', 'Items_Sold', 'Selling_Price']]

# Function to find top brands
def get_top_brands(df, n=5, metric='Revenue'):
    brand_agg = df.groupby('BRAND').agg({
        'Revenue': ['sum', 'mean', 'count'],
        'Items_Sold': 'sum',
        'Selling_Price': 'mean'
    }).reset_index()
    
    brand_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in brand_agg.columns]
    
    brand_agg = brand_agg.rename(columns={
        'Revenue_sum': 'Revenue',
        'Revenue_mean': 'Avg_Revenue_per_Transaction',
        'Revenue_count': 'Transaction_Count',
        'Items_Sold_sum': 'Items_Sold',
        'Selling_Price_mean': 'Avg_Price'
    })
    
    return brand_agg.nlargest(n, metric)

# Function to create bar plot with advanced styling
def plot_bar(df, x, y, color=None, title="", xaxis_title="", yaxis_title="", hover_data=None, barmode='relative'):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        labels={y: yaxis_title, x: xaxis_title},
        text_auto='.2s',
        hover_data=hover_data,
        barmode=barmode
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode='closest',
        font=dict(family="Arial", size=12, color="#2c3e50")
    )
    
    fig.update_traces(
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.8
    )
    
    return fig

# Function to create pie chart with advanced styling
def plot_pie(df, values, names, title="", hole=0.3):
    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        hole=hole
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12, color="#2c3e50"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_traces(
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=1))
    )
    
    return fig

# Function to create scatter plot with advanced styling
def plot_scatter(df, x, y, color=None, size=None, title="", xaxis_title="", yaxis_title="", hover_data=None):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title,
        labels={x: xaxis_title, y: yaxis_title},
        hover_data=hover_data
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family="Arial", size=12, color="#2c3e50")
    )
    
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='DarkSlateGrey'),
            opacity=0.7
        )
    )
    
    return fig

# Function to create line chart for time series
def plot_time_series(df, x, y, color=None, title="", xaxis_title="", yaxis_title=""):
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        labels={y: yaxis_title, x: xaxis_title}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode='x unified',
        font=dict(family="Arial", size=12, color="#2c3e50")
    )
    
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=8)
    )
    
    return fig

# Function to create a KPI card
def kpi_card(title, value, delta=None, delta_color="normal"):
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )

# Main Streamlit app
def main():
    # Set page config as the first Streamlit command
    st.set_page_config(
        page_title="Advanced Sales Analytics Dashboard",
        layout="wide",
        page_icon="📈"
    )
    
    # Apply custom CSS styling
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stMetric {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stPlotlyChart {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .css-1aumxhk {
            background-color: #ffffff;
            background-image: none;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
        }
        .highlight {
            background-color: #fffacd;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #ffd700;
        }
        .footer {
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.1);
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom header
    st.markdown("""
    <div style="background-color:#2c3e50;padding:20px;border-radius:10px;margin-bottom:20px">
        <h1 style="color:white;text-align:center;">Advanced Sales Analytics Dashboard</h1>
        <p style="color:white;text-align:center;">Interactive insights for your sales performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader for salesdata.xlsx
    file_path = "salesdata.xlsx"
    df = None
    
    if os.path.exists(file_path):
        df = load_and_clean_data(file_path=file_path)
    else:
        st.warning("File 'salesdata.xlsx' not found in the current directory.")
        uploaded_file = st.file_uploader("Please upload 'salesdata.xlsx'", type=['xlsx'])
        if uploaded_file is not None:
            df = load_and_clean_data(uploaded_file=uploaded_file)
    
    if df is None:
        st.error("Failed to load data. Please provide 'salesdata.xlsx'.")
        st.stop()
    
    # Sidebar for filtering
    with st.sidebar:
        st.markdown("""
        <div style="padding:10px;background-color:#34495e;border-radius:5px;margin-bottom:20px">
            <h3 style="color:white;">Filters & Controls</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Date filter (if date column exists)
        if 'Date' in df.columns:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & 
                        (df['Date'] <= pd.to_datetime(date_range[1]))]
        
        # Category filter
        main_categories = ['All'] + sorted(df['Main_category'].unique())
        selected_category = st.selectbox("Select Main Category", main_categories)
        
        # Sub-category filter
        if selected_category != 'All':
            sub_categories = ['All'] + sorted(df[df['Main_category'] == selected_category]['Sub-category'].unique())
            selected_subcategory = st.selectbox("Select Sub-category", sub_categories)
            
            if selected_subcategory != 'All':
                df = df[df['Sub-category'] == selected_subcategory]
        else:
            selected_subcategory = 'All'
        
        # Price range filter
        min_price, max_price = st.slider(
            "Price Range (KES)",
            float(df['Selling_Price'].min()),
            float(df['Selling_Price'].max()),
            (float(df['Selling_Price'].min()), float(df['Selling_Price'].max()))
        )
        df = df[(df['Selling_Price'] >= min_price) & (df['Selling_Price'] <= max_price)]
        
        # Brand filter
        brands = ['All'] + sorted(df['BRAND'].unique())
        selected_brand = st.multiselect("Select Brands", brands, default='All')
        
        if 'All' not in selected_brand and selected_brand:
            df = df[df['BRAND'].isin(selected_brand)]
        
        # Additional options
        st.markdown("---")
        st.markdown("**Display Options**")
        show_raw_data = st.checkbox("Show Raw Data", False)
        show_insights = st.checkbox("Show Automated Insights", True)
    
    # Key metrics
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_revenue = df['Revenue'].sum()
        kpi_card("Total Revenue", f"KES {total_revenue:,.2f}")
    
    with col2:
        total_items = df['Items_Sold'].sum()
        kpi_card("Total Items Sold", f"{total_items:,}")
    
    with col3:
        avg_price = df['Selling_Price'].mean()
        kpi_card("Average Price", f"KES {avg_price:,.2f}")
    
    with col4:
        avg_rev_per_item = total_revenue / total_items if total_items > 0 else 0
        kpi_card("Avg. Revenue/Item", f"KES {avg_rev_per_item:,.2f}")
    
    # Second row of KPIs
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        unique_products = df['Product_Name'].nunique()
        kpi_card("Unique Products", f"{unique_products}")
    
    with col6:
        unique_brands = df['BRAND'].nunique()
        kpi_card("Unique Brands", f"{unique_brands}")
    
    with col7:
        avg_items_per_transaction = total_items / len(df) if len(df) > 0 else 0
        kpi_card("Avg. Items/Transaction", f"{avg_items_per_transaction:,.1f}")
    
    with col8:
        if 'Date' in df.columns:
            days_covered = (df['Date'].max() - df['Date'].min()).days + 1
            daily_revenue = total_revenue / days_covered if days_covered > 0 else 0
            kpi_card("Avg. Daily Revenue", f"KES {daily_revenue:,.2f}")
        else:
            kpi_card("Transactions", f"{len(df):,}")
    
    # Revenue Analysis Section
    st.markdown("---")
    st.markdown("### 📈 Revenue Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Revenue by category and sub-category
        agg_df = aggregate_by_category(df, ['Main_category', 'Sub-category'])
        fig = plot_bar(
            agg_df,
            x='Sub-category',
            y='Revenue',
            color='Main_category',
            title="Revenue by Category and Sub-category",
            yaxis_title="Total Revenue (KES)",
            hover_data=['Main_category', 'Sub-category', 'Revenue', 'Items_Sold', 'Avg_Price']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue distribution
        fig = plot_pie(
            aggregate_by_category(df, ['Main_category']),
            values='Revenue',
            names='Main_category',
            title="Revenue Distribution by Main Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis (if date column exists)
    if 'Date' in df.columns:
        st.markdown("---")
        st.markdown("### 🗓️ Time Series Analysis")
        
        time_agg = df.groupby('Date').agg({'Revenue': 'sum', 'Items_Sold': 'sum'}).reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_time_series(
                time_agg,
                x='Date',
                y='Revenue',
                title="Daily Revenue Trend",
                yaxis_title="Revenue (KES)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_time_series(
                time_agg,
                x='Date',
                y='Items_Sold',
                title="Daily Items Sold Trend",
                yaxis_title="Items Sold"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly/Quarterly analysis
        time_period = st.radio("Select Time Period", ['Monthly', 'Quarterly'], horizontal=True)
        
        if time_period == 'Monthly':
            period_col = 'Month'
            time_agg = df.groupby(['Month', 'Main_category']).agg({'Revenue': 'sum'}).reset_index()
            time_agg['Month'] = pd.Categorical(time_agg['Month'], categories=[
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ], ordered=True)
            time_agg = time_agg.sort_values('Month')
        else:
            period_col = 'Quarter'
            time_agg = df.groupby(['Quarter', 'Main_category']).agg({'Revenue': 'sum'}).reset_index()
        
        fig = plot_bar(
            time_agg,
            x=period_col,
            y='Revenue',
            color='Main_category',
            title=f"Revenue by {time_period} and Category",
            yaxis_title="Total Revenue (KES)",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Product and Pricing Insights
    st.markdown("---")
    st.markdown("### 🛍️ Product & Pricing Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        # Price vs Items Sold
        fig = plot_scatter(
            df,
            x='Selling_Price',
            y='Items_Sold',
            color='Main_category',
            size='Revenue',
            title="Price vs. Items Sold Analysis",
            xaxis_title="Selling Price (KES)",
            yaxis_title="Items Sold",
            hover_data=['Product_Name', 'BRAND', 'Revenue']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top products
        top_metric = st.selectbox("Top Products By", ['Revenue', 'Items_Sold'], key='top_products_metric')
        top_products = get_top_products(df, 10, top_metric)
        fig = plot_bar(
            top_products,
            x='Product_Name',
            y=top_metric,
            title=f"Top 10 Products by {top_metric}",
            yaxis_title=top_metric,
            hover_data=['Revenue', 'Items_Sold', 'Selling_Price']
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Brand Performance
    st.markdown("---")
    st.markdown("### 🏷️ Brand Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        # Top brands
        brand_metric = st.selectbox("Top Brands By", ['Revenue', 'Items_Sold'], key='top_brands_metric')
        top_brands = get_top_brands(df, 5, brand_metric)
        fig = plot_bar(
            top_brands,
            x='BRAND',
            y=brand_metric,
            title=f"Top 5 Brands by {brand_metric}",
            yaxis_title=brand_metric,
            hover_data=['Revenue', 'Items_Sold', 'Avg_Price', 'Transaction_Count']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Brand market share
        brand_agg = aggregate_by_category(df, ['BRAND'])
        fig = plot_pie(
            brand_agg.nlargest(5, 'Revenue'),
            values='Revenue',
            names='BRAND',
            title="Top 5 Brands by Revenue Share",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Automated Insights
    if show_insights:
        st.markdown("---")
        st.markdown("### 🔍 Automated Insights")
        
        # Calculate interesting metrics
        top_product = get_top_products(df, 1).iloc[0]
        top_brand = get_top_brands(df, 1).iloc[0]
        revenue_per_item = calculate_revenue_per_item(df)
        
        # Create columns for insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight">
                <h4>🏆 Top Performing Product</h4>
                <p><b>{}</b> generated the most revenue at <b>KES {:,}</b> from selling <b>{:,}</b> units at an average price of <b>KES {:,}</b>.</p>
            </div>
            """.format(
                top_product['Product_Name'],
                top_product['Revenue'],
                top_product['Items_Sold'],
                top_product['Selling_Price']
            ), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="highlight">
                <h4>💰 High-Value Products</h4>
                <p><b>{}</b> has the highest revenue-to-items-sold ratio at <b>KES {:.2f}</b> per item, indicating premium pricing or high-margin sales.</p>
            </div>
            """.format(
                revenue_per_item['Product_Name'].iloc[0],
                revenue_per_item['Revenue_per_Item'].iloc[0]
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight">
                <h4>🏅 Top Performing Brand</h4>
                <p><b>{}</b> generated the most revenue at <b>KES {:,}</b> from <b>{:,}</b> items sold across <b>{:,}</b> transactions.</p>
            </div>
            """.format(
                top_brand['BRAND'],
                top_brand['Revenue'],
                top_brand['Items_Sold'],
                top_brand['Transaction_Count']
            ), unsafe_allow_html=True)
            
            # Price distribution insight
            price_stats = df['Selling_Price'].describe()
            st.markdown("""
            <div class="highlight">
                <h4>💵 Price Distribution</h4>
                <p>The average product price is <b>KES {:.2f}</b>, with 50% of products priced between <b>KES {:.2f}</b> and <b>KES {:.2f}</b>.</p>
            </div>
            """.format(
                price_stats['mean'],
                price_stats['25%'],
                price_stats['75%']
            ), unsafe_allow_html=True)
    
    # Raw data display
    if show_raw_data:
        st.markdown("---")
        st.markdown("### 📋 Raw Data Preview")
        st.dataframe(df.head(100))
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        Developed by Slevin Neko
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()