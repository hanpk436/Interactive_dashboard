import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Advanced Sales Analytics",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🚀 Advanced Sales Analytics Dashboard")
st.markdown("Interactive analysis with drill-down capabilities")


# Create sample data
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range('2023-07-01', '2023-12-31', freq='D')

    data = pd.DataFrame({
        'date': np.random.choice(dates, size=500, replace=True),
        'transaction_id': [f'TXN{1000 + i}' for i in range(500)],
        'sales_amount': np.random.normal(1000, 300, 500),
        'cost_amount': np.random.normal(600, 150, 500),
        'quantity': np.random.randint(1, 10, 500),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home Goods', 'Books'], 500),
        'customer_segment': np.random.choice(['Retail', 'Wholesale', 'Online'], 500),
        'sales_rep': np.random.choice(['Rep_A', 'Rep_B', 'Rep_C', 'Rep_D'], 500)
    })

    data['profit'] = data['sales_amount'] - data['cost_amount']
    data['profit_margin'] = (data['profit'] / data['sales_amount'] * 100).round(2)
    data['cost_amount'] = np.where(data['profit'] < 0,
                                   data['sales_amount'] * np.random.uniform(0.5, 0.8),
                                   data['cost_amount'])
    data['profit'] = data['sales_amount'] - data['cost_amount']
    data['profit_margin'] = (data['profit'] / data['sales_amount'] * 100).round(2)

    return data.sort_values('date').reset_index(drop=True)


# Load data
df = load_data()

# SIDEBAR FILTERS
st.sidebar.header("🎛️ Filters")

# Date range and period in sidebar
st.sidebar.subheader("Time Settings")
start_date = st.sidebar.date_input("Start Date", df['date'].min())
end_date = st.sidebar.date_input("End Date", df['date'].max())

period = st.sidebar.selectbox(
    "View Period",
    ["Weekly", "Monthly", "Quarterly"],
    help="Select the time period for aggregation"
)

# Business dimension filters in sidebar
st.sidebar.subheader("Business Dimensions")
regions = st.sidebar.multiselect(
    "Regions",
    options=df['region'].unique(),
    default=df['region'].unique()
)

categories = st.sidebar.multiselect(
    "Product Categories",
    options=df['product_category'].unique(),
    default=df['product_category'].unique()
)

segments = st.sidebar.multiselect(
    "Customer Segments",
    options=df['customer_segment'].unique(),
    default=df['customer_segment'].unique()
)

# Apply filters
filtered_df = df[
    (df['date'] >= pd.to_datetime(start_date)) &
    (df['date'] <= pd.to_datetime(end_date)) &
    (df['region'].isin(regions)) &
    (df['product_category'].isin(categories)) &
    (df['customer_segment'].isin(segments))
    ].copy()

# REAL-TIME METRICS
st.markdown("---")
st.subheader("📊 Live Performance Metrics")

# Calculate metrics
total_sales = filtered_df['sales_amount'].sum()
total_cost = filtered_df['cost_amount'].sum()
total_profit = filtered_df['profit'].sum()
avg_margin = filtered_df['profit_margin'].mean()
total_transactions = len(filtered_df)

# Display metrics
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("Total Sales", f"${total_sales:,.0f}")

with metric_col2:
    st.metric("Total Cost", f"${total_cost:,.0f}")

with metric_col3:
    st.metric("Total Profit", f"${total_profit:,.0f}")

with metric_col4:
    st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")

with metric_col5:
    st.metric("Total Transactions", f"{total_transactions}")

# INTERACTIVE VISUALIZATIONS WITH DRILL-DOWN
st.markdown("---")
st.subheader("📈 Interactive Analytics")

# Tabbed interface for different views
tab1, tab2, tab3 = st.tabs(["Financial Trends", "Performance Analysis", "Data Explorer"])

with tab1:
    # Financial Trends Tab
    st.write(f"**Financial Performance Trend ({period} View)**")


    # Aggregate data by selected period
    def aggregate_by_period(data, period):
        data_copy = data.copy()

        if period == "Weekly":
            data_copy['period'] = data_copy['date'].dt.to_period('W').apply(lambda r: r.start_time)
        elif period == "Monthly":
            data_copy['period'] = data_copy['date'].dt.to_period('M').apply(lambda r: r.start_time)
        elif period == "Quarterly":
            data_copy['period'] = data_copy['date'].dt.to_period('Q').apply(lambda r: r.start_time)

        aggregated = data_copy.groupby('period').agg({
            'sales_amount': 'sum',
            'cost_amount': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean',
            'transaction_id': 'count'
        }).reset_index()

        aggregated.rename(columns={'transaction_id': 'transaction_count'}, inplace=True)
        return aggregated


    agg_data = aggregate_by_period(filtered_df, period)

    col1, col2 = st.columns(2)

    with col1:
        # Financial trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg_data['period'], y=agg_data['sales_amount'],
                                 mode='lines+markers', name='Sales', line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=agg_data['period'], y=agg_data['cost_amount'],
                                 mode='lines+markers', name='Cost', line=dict(color='#ff7f0e', width=3)))
        fig.add_trace(go.Scatter(x=agg_data['period'], y=agg_data['profit'],
                                 mode='lines+markers', name='Profit', line=dict(color='#2ca02c', width=3)))

        fig.update_layout(
            title=f"Sales, Cost & Profit Trend ({period})",
            xaxis_title="Period",
            yaxis_title="Amount ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Profit margin trend with drill-down options
        st.write("**Profit Margin Analysis**")

        # Drill-down selection
        drill_down_by = st.radio(
            "View Profit Margin by:",
            ["Overall Trend", "Product Category", "Region", "Customer Segment"],
            horizontal=True,
            key="drill_down"
        )

        if drill_down_by == "Overall Trend":
            # Overall profit margin trend
            fig = go.Figure()
            fig.add_trace(go.Bar(x=agg_data['period'], y=agg_data['profit_margin'],
                                 name='Profit Margin', marker_color='lightgreen'))

            avg_margin_line = agg_data['profit_margin'].mean()
            fig.add_hline(y=avg_margin_line, line_dash="dash", line_color="red",
                          annotation_text=f"Average: {avg_margin_line:.1f}%")

            fig.update_layout(
                title=f"Profit Margin Trend ({period})",
                xaxis_title="Period",
                yaxis_title="Profit Margin (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif drill_down_by == "Product Category":
            # Profit margin by product category over time
            category_trend_data = filtered_df.copy()

            if period == "Weekly":
                category_trend_data['period'] = category_trend_data['date'].dt.to_period('W').apply(
                    lambda r: r.start_time)
            elif period == "Monthly":
                category_trend_data['period'] = category_trend_data['date'].dt.to_period('M').apply(
                    lambda r: r.start_time)
            elif period == "Quarterly":
                category_trend_data['period'] = category_trend_data['date'].dt.to_period('Q').apply(
                    lambda r: r.start_time)

            category_margin_trend = category_trend_data.groupby(['period', 'product_category'])[
                'profit_margin'].mean().reset_index()

            fig = px.line(category_margin_trend, x='period', y='profit_margin', color='product_category',
                          title=f"Profit Margin Trend by Product Category ({period})",
                          markers=True)
            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Profit Margin (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif drill_down_by == "Region":
            # Profit margin by region over time
            region_trend_data = filtered_df.copy()

            if period == "Weekly":
                region_trend_data['period'] = region_trend_data['date'].dt.to_period('W').apply(lambda r: r.start_time)
            elif period == "Monthly":
                region_trend_data['period'] = region_trend_data['date'].dt.to_period('M').apply(lambda r: r.start_time)
            elif period == "Quarterly":
                region_trend_data['period'] = region_trend_data['date'].dt.to_period('Q').apply(lambda r: r.start_time)

            region_margin_trend = region_trend_data.groupby(['period', 'region'])['profit_margin'].mean().reset_index()

            fig = px.line(region_margin_trend, x='period', y='profit_margin', color='region',
                          title=f"Profit Margin Trend by Region ({period})",
                          markers=True)
            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Profit Margin (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif drill_down_by == "Customer Segment":
            # Profit margin by customer segment over time
            segment_trend_data = filtered_df.copy()

            if period == "Weekly":
                segment_trend_data['period'] = segment_trend_data['date'].dt.to_period('W').apply(
                    lambda r: r.start_time)
            elif period == "Monthly":
                segment_trend_data['period'] = segment_trend_data['date'].dt.to_period('M').apply(
                    lambda r: r.start_time)
            elif period == "Quarterly":
                segment_trend_data['period'] = segment_trend_data['date'].dt.to_period('Q').apply(
                    lambda r: r.start_time)

            segment_margin_trend = segment_trend_data.groupby(['period', 'customer_segment'])[
                'profit_margin'].mean().reset_index()

            fig = px.line(segment_margin_trend, x='period', y='profit_margin', color='customer_segment',
                          title=f"Profit Margin Trend by Customer Segment ({period})",
                          markers=True)
            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="Profit Margin (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Performance Analysis Tab
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Sales & Profit Analysis**")

        # Dimension selection for sales & profit analysis
        dimension = st.radio(
            "View by:",
            ["Overall", "Region", "Product Category", "Customer Segment"],
            horizontal=True,
            key="sales_profit_dimension"
        )

        if dimension == "Overall":
            # Show overall metrics as a single bar for comparison
            performance_data = pd.DataFrame({
                'metric': ['Sales', 'Profit'],
                'amount': [total_sales, total_profit],
                'color': ['#1f77b4', '#2ca02c']
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data['metric'],
                y=performance_data['amount'],
                marker_color=performance_data['color'],
                text=performance_data['amount'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Amount: $%{y:,.0f}<extra></extra>'
            ))

            fig.update_layout(
                title="Overall Sales & Profit",
                xaxis_title="Metric",
                yaxis_title="Amount ($)",
                height=400,
                showlegend=False
            )

        elif dimension == "Region":
            performance_data = filtered_df.groupby('region').agg({
                'sales_amount': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            x_axis = 'region'
            title = "Sales & Profit by Region"

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['sales_amount'],
                name='Sales',
                marker_color='#1f77b4',
                text=performance_data['sales_amount'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['profit'],
                name='Profit',
                marker_color='#2ca02c',
                text=performance_data['profit'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Profit: $%{y:,.0f}<br>Margin: ' +
                              performance_data['profit_margin'].round(1).astype(str) + '%<extra></extra>'
            ))
            fig.update_layout(
                title=title,
                xaxis_title=dimension,
                yaxis_title="Amount ($)",
                barmode='group',
                height=400,
                showlegend=True
            )

        elif dimension == "Product Category":
            performance_data = filtered_df.groupby('product_category').agg({
                'sales_amount': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            x_axis = 'product_category'
            title = "Sales & Profit by Product Category"

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['sales_amount'],
                name='Sales',
                marker_color='#1f77b4',
                text=performance_data['sales_amount'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['profit'],
                name='Profit',
                marker_color='#2ca02c',
                text=performance_data['profit'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Profit: $%{y:,.0f}<br>Margin: ' +
                              performance_data['profit_margin'].round(1).astype(str) + '%<extra></extra>'
            ))
            fig.update_layout(
                title=title,
                xaxis_title=dimension,
                yaxis_title="Amount ($)",
                barmode='group',
                height=400,
                showlegend=True
            )

        elif dimension == "Customer Segment":
            performance_data = filtered_df.groupby('customer_segment').agg({
                'sales_amount': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            x_axis = 'customer_segment'
            title = "Sales & Profit by Customer Segment"

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['sales_amount'],
                name='Sales',
                marker_color='#1f77b4',
                text=performance_data['sales_amount'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['profit'],
                name='Profit',
                marker_color='#2ca02c',
                text=performance_data['profit'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Profit: $%{y:,.0f}<br>Margin: ' +
                              performance_data['profit_margin'].round(1).astype(str) + '%<extra></extra>'
            ))
            fig.update_layout(
                title=title,
                xaxis_title=dimension,
                yaxis_title="Amount ($)",
                barmode='group',
                height=400,
                showlegend=True
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Margin Distribution Analysis**")

        # Dimension selection for margin analysis
        margin_dimension = st.radio(
            "View Margin Distribution by:",
            ["Product Category", "Region", "Customer Segment"],
            horizontal=True,
            key="margin_dimension"
        )

        if margin_dimension == "Product Category":
            margin_data = filtered_df.groupby('product_category').agg({
                'profit': 'sum',
                'sales_amount': 'sum'
            }).reset_index()
            margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales_amount'] * 100).round(1)
            category_col = 'product_category'
            title = "Profit Margin Distribution by Product Category"

        elif margin_dimension == "Region":
            margin_data = filtered_df.groupby('region').agg({
                'profit': 'sum',
                'sales_amount': 'sum'
            }).reset_index()
            margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales_amount'] * 100).round(1)
            category_col = 'region'
            title = "Profit Margin Distribution by Region"

        elif margin_dimension == "Customer Segment":
            margin_data = filtered_df.groupby('customer_segment').agg({
                'profit': 'sum',
                'sales_amount': 'sum'
            }).reset_index()
            margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales_amount'] * 100).round(1)
            category_col = 'customer_segment'
            title = "Profit Margin Distribution by Customer Segment"

        # Create pie chart for margin distribution
        fig = px.pie(margin_data,
                     values='profit',
                     names=category_col,
                     title=title,
                     hover_data=['profit_margin'],
                     labels={'profit_margin': 'Margin %'},
                     hole=0.3)

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Profit: $%{value:,.0f}<br>Margin: %{customdata[0]:.1f}%<extra></extra>'
        )

        fig.update_layout(
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Data Explorer Tab
    st.write("**Interactive Data Explorer**")

    # Quick statistics
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Records Found", len(filtered_df))
    with stats_col2:
        st.metric("Date Range", f"{start_date} to {end_date}")
    with stats_col3:
        st.metric("View Period", period)
    with stats_col4:
        st.metric("Regions Active", len(regions))

    # Search and filter
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_term = st.text_input("🔍 Search in transactions...", placeholder="Enter product, region, or sales rep...")
    with search_col2:
        sort_by = st.selectbox("Sort by", ['date', 'sales_amount', 'profit', 'profit_margin'])

    # Apply search
    if search_term:
        explorer_df = filtered_df[
            filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
    else:
        explorer_df = filtered_df

    # Sort data
    explorer_df = explorer_df.sort_values(sort_by, ascending=False)

    # Display data
    st.dataframe(
        explorer_df[['date', 'transaction_id', 'region', 'product_category',
                     'customer_segment', 'sales_amount', 'cost_amount', 'profit', 'profit_margin']],
        width='stretch',
        height=400
    )

    # Export
    csv = explorer_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Current Data",
        data=csv,
        file_name=f"sales_data_{start_date}_to_{end_date}.csv",
        mime="text/csv"
    )

# QUICK INSIGHTS IN SIDEBAR
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Quick Insights")

if not filtered_df.empty:
    # Top performer
    top_region = filtered_df.groupby('region')['profit'].sum().idxmax()
    top_region_profit = filtered_df.groupby('region')['profit'].sum().max()
    st.sidebar.success(f"**Top Region**: {top_region} (${top_region_profit:,.0f})")

    # Best margin
    best_margin_product = filtered_df.groupby('product_category')['profit_margin'].mean().idxmax()
    best_margin = filtered_df.groupby('product_category')['profit_margin'].mean().max()
    st.sidebar.info(f"**Best Margin**: {best_margin_product} ({best_margin:.1f}%)")

    # Customer segment insight
    best_segment = filtered_df.groupby('customer_segment')['profit_margin'].mean().idxmax()
    best_segment_margin = filtered_df.groupby('customer_segment')['profit_margin'].mean().max()
    st.sidebar.info(f"**Best Segment**: {best_segment} ({best_segment_margin:.1f}% margin)")

    # Alert for low performance
    lowest_margin = filtered_df.groupby('product_category')['profit_margin'].mean().min()
    if lowest_margin < 10:
        st.sidebar.warning(f"**Watch**: Some products below 10% margin")
else:
    st.sidebar.info("No data available for current filters")

# Footer
st.markdown("---")
st.caption(
    f"Advanced Sales Analytics Dashboard | {period} view | {start_date} to {end_date} | {len(filtered_df)} transactions")