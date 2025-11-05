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


# Language configuration
def get_translations(language):
    translations = {
        'english': {
            'title': "🚀 Advanced Sales Analytics Dashboard",
            'subtitle': "Interactive analysis with drill-down capabilities",
            'filters': "🎛️ Filters",
            'time_settings': "Time Settings",
            'start_date': "Start Date",
            'end_date': "End Date",
            'view_period': "View Period",
            'period_help': "Select the time period for aggregation",
            'business_dims': "Business Dimensions",
            'regions': "Regions",
            'categories': "Product Categories",
            'segments': "Customer Segments",
            'metrics': "📊 Live Performance Metrics",
            'total_sales': "Total Sales",
            'total_cost': "Total Cost",
            'total_profit': "Total Profit",
            'avg_margin': "Avg Profit Margin",
            'total_transactions': "Total Transactions",
            'analytics': "📈 Interactive Analytics",
            'financial_trends': "Financial Trends",
            'performance_analysis': "Performance Analysis",
            'data_explorer': "Data Explorer",
            'financial_performance': "Financial Performance Trend",
            'sales_cost_profit': "Sales, Cost & Profit Trend",
            'profit_margin_analysis': "Profit Margin Analysis",
            'view_profit_margin_by': "View Profit Margin by:",
            'overall_trend': "Overall Trend",
            'product_category': "Product Category",
            'region': "Region",
            'customer_segment': "Customer Segment",
            'profit_margin_trend': "Profit Margin Trend",
            'sales_profit_analysis': "Sales & Profit Analysis",
            'view_by': "View by:",
            'overall': "Overall",
            'sales_profit_by_region': "Sales & Profit by Region",
            'sales_profit_by_category': "Sales & Profit by Product Category",
            'sales_profit_by_segment': "Sales & Profit by Customer Segment",
            'overall_sales_profit': "Overall Sales & Profit",
            'margin_distribution': "Margin Distribution Analysis",
            'view_margin_by': "View Margin Distribution by:",
            'margin_by_category': "Profit Margin Distribution by Product Category",
            'margin_by_region': "Profit Margin Distribution by Region",
            'margin_by_segment': "Profit Margin Distribution by Customer Segment",
            'interactive_explorer': "Interactive Data Explorer",
            'records_found': "Records Found",
            'date_range': "Date Range",
            'regions_active': "Regions Active",
            'search_transactions': "🔍 Search in transactions...",
            'search_placeholder': "Enter product, region, or sales rep...",
            'sort_by': "Sort by",
            'export_data': "📥 Export Current Data",
            'quick_insights': "💡 Quick Insights",
            'top_region': "Top Region",
            'best_margin': "Best Margin",
            'best_segment': "Best Segment",
            'watch': "Watch",
            'no_data': "No data available for current filters",
            'footer': "Advanced Sales Analytics Dashboard",
            'periods': ["Weekly", "Monthly", "Quarterly"],
            'sample_regions': ['North', 'South', 'East', 'West'],
            'sample_categories': ['Electronics', 'Clothing', 'Home Goods', 'Books'],
            'sample_segments': ['Retail', 'Wholesale', 'Online'],
            'sample_reps': ['Rep_A', 'Rep_B', 'Rep_C', 'Rep_D']
        },
        'chinese': {
            'title': "🚀 高级销售分析仪表板",
            'subtitle': "具有钻取功能的交互式分析",
            'filters': "🎛️ 筛选器",
            'time_settings': "时间设置",
            'start_date': "开始日期",
            'end_date': "结束日期",
            'view_period': "查看周期",
            'period_help': "选择数据聚合的时间周期",
            'business_dims': "业务维度",
            'regions': "区域",
            'categories': "产品类别",
            'segments': "客户细分",
            'metrics': "📊 实时绩效指标",
            'total_sales': "总销售额",
            'total_cost': "总成本",
            'total_profit': "总利润",
            'avg_margin': "平均利润率",
            'total_transactions': "总交易数",
            'analytics': "📈 交互式分析",
            'financial_trends': "财务趋势",
            'performance_analysis': "绩效分析",
            'data_explorer': "数据探索",
            'financial_performance': "财务绩效趋势",
            'sales_cost_profit': "销售额、成本和利润趋势",
            'profit_margin_analysis': "利润率分析",
            'view_profit_margin_by': "按以下维度查看利润率:",
            'overall_trend': "整体趋势",
            'product_category': "产品类别",
            'region': "区域",
            'customer_segment': "客户细分",
            'profit_margin_trend': "利润率趋势",
            'sales_profit_analysis': "销售额和利润分析",
            'view_by': "查看维度:",
            'overall': "整体",
            'sales_profit_by_region': "按区域的销售额和利润",
            'sales_profit_by_category': "按产品类别的销售额和利润",
            'sales_profit_by_segment': "按客户细分的销售额和利润",
            'overall_sales_profit': "整体销售额和利润",
            'margin_distribution': "利润率分布分析",
            'view_margin_by': "按以下维度查看利润率分布:",
            'margin_by_category': "按产品类别的利润率分布",
            'margin_by_region': "按区域的利润率分布",
            'margin_by_segment': "按客户细分的利润率分布",
            'interactive_explorer': "交互式数据探索",
            'records_found': "找到的记录",
            'date_range': "日期范围",
            'regions_active': "活跃区域",
            'search_transactions': "🔍 搜索交易...",
            'search_placeholder': "输入产品、区域或销售代表...",
            'sort_by': "排序方式",
            'export_data': "📥 导出当前数据",
            'quick_insights': "💡 快速洞察",
            'top_region': "最佳区域",
            'best_margin': "最佳利润率",
            'best_segment': "最佳细分",
            'watch': "注意",
            'no_data': "当前筛选条件下无可用数据",
            'footer': "高级销售分析仪表板",
            'periods': ["周度", "月度", "季度"],
            'sample_regions': ['北部', '南部', '东部', '西部'],
            'sample_categories': ['电子产品', '服装', '家居用品', '书籍'],
            'sample_segments': ['零售', '批发', '在线'],
            'sample_reps': ['销售员_A', '销售员_B', '销售员_C', '销售员_D']
        }
    }
    return translations[language]


# Language selector in sidebar
st.sidebar.header("🌐 Language / 语言")
language = st.sidebar.radio("Select Language / 选择语言:", ["English", "Chinese"], horizontal=True)
lang = 'english' if language == "English" else 'chinese'
t = get_translations(lang)

# Title and description
st.title(t['title'])
st.markdown(t['subtitle'])


# Create sample data
@st.cache_data
def load_data(language):
    np.random.seed(42)
    dates = pd.date_range('2023-07-01', '2023-12-31', freq='D')

    t_data = get_translations(language)

    data = pd.DataFrame({
        'date': np.random.choice(dates, size=500, replace=True),
        'transaction_id': [f'TXN{1000 + i}' for i in range(500)],
        'sales_amount': np.random.normal(1000, 300, 500),
        'cost_amount': np.random.normal(600, 150, 500),
        'quantity': np.random.randint(1, 10, 500),
        'region': np.random.choice(t_data['sample_regions'], 500),
        'product_category': np.random.choice(t_data['sample_categories'], 500),
        'customer_segment': np.random.choice(t_data['sample_segments'], 500),
        'sales_rep': np.random.choice(t_data['sample_reps'], 500)
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
df = load_data(lang)

# SIDEBAR FILTERS
st.sidebar.header(t['filters'])

# Date range and period in sidebar
st.sidebar.subheader(t['time_settings'])
start_date = st.sidebar.date_input(t['start_date'], df['date'].min())
end_date = st.sidebar.date_input(t['end_date'], df['date'].max())

period = st.sidebar.selectbox(
    t['view_period'],
    t['periods'],
    help=t['period_help']
)

# Business dimension filters in sidebar
st.sidebar.subheader(t['business_dims'])
regions = st.sidebar.multiselect(
    t['regions'],
    options=df['region'].unique(),
    default=df['region'].unique()
)

categories = st.sidebar.multiselect(
    t['categories'],
    options=df['product_category'].unique(),
    default=df['product_category'].unique()
)

segments = st.sidebar.multiselect(
    t['segments'],
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
st.subheader(t['metrics'])

# Calculate metrics
total_sales = filtered_df['sales_amount'].sum()
total_cost = filtered_df['cost_amount'].sum()
total_profit = filtered_df['profit'].sum()
avg_margin = filtered_df['profit_margin'].mean()
total_transactions = len(filtered_df)

# Display metrics
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric(t['total_sales'], f"${total_sales:,.0f}")

with metric_col2:
    st.metric(t['total_cost'], f"${total_cost:,.0f}")

with metric_col3:
    st.metric(t['total_profit'], f"${total_profit:,.0f}")

with metric_col4:
    st.metric(t['avg_margin'], f"{avg_margin:.1f}%")

with metric_col5:
    st.metric(t['total_transactions'], f"{total_transactions}")

# INTERACTIVE VISUALIZATIONS WITH DRILL-DOWN
st.markdown("---")
st.subheader(t['analytics'])

# Tabbed interface for different views
tab1, tab2, tab3 = st.tabs([t['financial_trends'], t['performance_analysis'], t['data_explorer']])

with tab1:
    # Financial Trends Tab
    st.write(f"**{t['financial_performance']} ({period}{t['view_period'][-1] if lang == 'chinese' else ''} View)**")


    # Aggregate data by selected period
    def aggregate_by_period(data, period, lang):
        data_copy = data.copy()

        period_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q'} if lang == 'english' else {'周度': 'W',
                                                                                                  '月度': 'M',
                                                                                                  '季度': 'Q'}
        period_key = period_map[period]

        data_copy['period'] = data_copy['date'].dt.to_period(period_key).apply(lambda r: r.start_time)

        aggregated = data_copy.groupby('period').agg({
            'sales_amount': 'sum',
            'cost_amount': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean',
            'transaction_id': 'count'
        }).reset_index()

        aggregated.rename(columns={'transaction_id': 'transaction_count'}, inplace=True)
        return aggregated


    agg_data = aggregate_by_period(filtered_df, period, lang)

    col1, col2 = st.columns(2)

    with col1:
        # Financial trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg_data['period'], y=agg_data['sales_amount'],
                                 mode='lines+markers', name=t['total_sales'], line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=agg_data['period'], y=agg_data['cost_amount'],
                                 mode='lines+markers', name=t['total_cost'], line=dict(color='#ff7f0e', width=3)))
        fig.add_trace(go.Scatter(x=agg_data['period'], y=agg_data['profit'],
                                 mode='lines+markers', name=t['total_profit'], line=dict(color='#2ca02c', width=3)))

        fig.update_layout(
            title=f"{t['sales_cost_profit']} ({period})",
            xaxis_title="Period" if lang == 'english' else "期间",
            yaxis_title="Amount ($)" if lang == 'english' else "金额 ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Profit margin trend with drill-down options
        st.write(f"**{t['profit_margin_analysis']}**")

        # Drill-down selection
        drill_down_by = st.radio(
            t['view_profit_margin_by'],
            [t['overall_trend'], t['product_category'], t['region'], t['customer_segment']],
            horizontal=True,
            key="drill_down"
        )

        if drill_down_by == t['overall_trend']:
            # Overall profit margin trend
            fig = go.Figure()
            fig.add_trace(go.Bar(x=agg_data['period'], y=agg_data['profit_margin'],
                                 name=t['avg_margin'], marker_color='lightgreen'))

            avg_margin_line = agg_data['profit_margin'].mean()
            fig.add_hline(y=avg_margin_line, line_dash="dash", line_color="red",
                          annotation_text=f"{'Average' if lang == 'english' else '平均'}: {avg_margin_line:.1f}%")

            fig.update_layout(
                title=f"{t['profit_margin_trend']} ({period})",
                xaxis_title="Period" if lang == 'english' else "期间",
                yaxis_title="Profit Margin (%)" if lang == 'english' else "利润率 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif drill_down_by == t['product_category']:
            # Profit margin by product category over time
            category_trend_data = filtered_df.copy()
            period_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q'} if lang == 'english' else {'周度': 'W',
                                                                                                      '月度': 'M',
                                                                                                      '季度': 'Q'}
            period_key = period_map[period]

            category_trend_data['period'] = category_trend_data['date'].dt.to_period(period_key).apply(
                lambda r: r.start_time)

            category_margin_trend = category_trend_data.groupby(['period', 'product_category'])[
                'profit_margin'].mean().reset_index()

            fig = px.line(category_margin_trend, x='period', y='profit_margin', color='product_category',
                          title=f"{t['margin_by_category']} ({period})",
                          markers=True)
            fig.update_layout(
                xaxis_title="Period" if lang == 'english' else "期间",
                yaxis_title="Profit Margin (%)" if lang == 'english' else "利润率 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif drill_down_by == t['region']:
            # Profit margin by region over time
            region_trend_data = filtered_df.copy()
            period_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q'} if lang == 'english' else {'周度': 'W',
                                                                                                      '月度': 'M',
                                                                                                      '季度': 'Q'}
            period_key = period_map[period]

            region_trend_data['period'] = region_trend_data['date'].dt.to_period(period_key).apply(
                lambda r: r.start_time)

            region_margin_trend = region_trend_data.groupby(['period', 'region'])['profit_margin'].mean().reset_index()

            fig = px.line(region_margin_trend, x='period', y='profit_margin', color='region',
                          title=f"{t['margin_by_region']} ({period})",
                          markers=True)
            fig.update_layout(
                xaxis_title="Period" if lang == 'english' else "期间",
                yaxis_title="Profit Margin (%)" if lang == 'english' else "利润率 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif drill_down_by == t['customer_segment']:
            # Profit margin by customer segment over time
            segment_trend_data = filtered_df.copy()
            period_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q'} if lang == 'english' else {'周度': 'W',
                                                                                                      '月度': 'M',
                                                                                                      '季度': 'Q'}
            period_key = period_map[period]

            segment_trend_data['period'] = segment_trend_data['date'].dt.to_period(period_key).apply(
                lambda r: r.start_time)

            segment_margin_trend = segment_trend_data.groupby(['period', 'customer_segment'])[
                'profit_margin'].mean().reset_index()

            fig = px.line(segment_margin_trend, x='period', y='profit_margin', color='customer_segment',
                          title=f"{t['margin_by_segment']} ({period})",
                          markers=True)
            fig.update_layout(
                xaxis_title="Period" if lang == 'english' else "期间",
                yaxis_title="Profit Margin (%)" if lang == 'english' else "利润率 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Performance Analysis Tab
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**{t['sales_profit_analysis']}**")

        # Dimension selection for sales & profit analysis
        dimension = st.radio(
            t['view_by'],
            [t['overall'], t['region'], t['product_category'], t['customer_segment']],
            horizontal=True,
            key="sales_profit_dimension"
        )

        if dimension == t['overall']:
            # Show overall metrics as a single bar for comparison
            performance_data = pd.DataFrame({
                'metric': [t['total_sales'], t['total_profit']],
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
                hovertemplate='<b>%{x}</b><br>' + (
                    'Amount' if lang == 'english' else '金额') + ': $%{y:,.0f}<extra></extra>'
            ))

            fig.update_layout(
                title=t['overall_sales_profit'],
                xaxis_title="Metric" if lang == 'english' else "指标",
                yaxis_title="Amount ($)" if lang == 'english' else "金额 ($)",
                height=400,
                showlegend=False
            )

        elif dimension == t['region']:
            performance_data = filtered_df.groupby('region').agg({
                'sales_amount': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            x_axis = 'region'
            title = t['sales_profit_by_region']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['sales_amount'],
                name=t['total_sales'],
                marker_color='#1f77b4',
                text=performance_data['sales_amount'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + t['total_sales'] + ': $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['profit'],
                name=t['total_profit'],
                marker_color='#2ca02c',
                text=performance_data['profit'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + t['total_profit'] + ': $%{y:,.0f}<br>' + t['avg_margin'] + ': ' +
                              performance_data['profit_margin'].round(1).astype(str) + '%<extra></extra>'
            ))
            fig.update_layout(
                title=title,
                xaxis_title=dimension,
                yaxis_title="Amount ($)" if lang == 'english' else "金额 ($)",
                barmode='group',
                height=400,
                showlegend=True
            )

        elif dimension == t['product_category']:
            performance_data = filtered_df.groupby('product_category').agg({
                'sales_amount': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            x_axis = 'product_category'
            title = t['sales_profit_by_category']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['sales_amount'],
                name=t['total_sales'],
                marker_color='#1f77b4',
                text=performance_data['sales_amount'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + t['total_sales'] + ': $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['profit'],
                name=t['total_profit'],
                marker_color='#2ca02c',
                text=performance_data['profit'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + t['total_profit'] + ': $%{y:,.0f}<br>' + t['avg_margin'] + ': ' +
                              performance_data['profit_margin'].round(1).astype(str) + '%<extra></extra>'
            ))
            fig.update_layout(
                title=title,
                xaxis_title=dimension,
                yaxis_title="Amount ($)" if lang == 'english' else "金额 ($)",
                barmode='group',
                height=400,
                showlegend=True
            )

        elif dimension == t['customer_segment']:
            performance_data = filtered_df.groupby('customer_segment').agg({
                'sales_amount': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean'
            }).reset_index()
            x_axis = 'customer_segment'
            title = t['sales_profit_by_segment']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['sales_amount'],
                name=t['total_sales'],
                marker_color='#1f77b4',
                text=performance_data['sales_amount'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + t['total_sales'] + ': $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=performance_data[x_axis],
                y=performance_data['profit'],
                name=t['total_profit'],
                marker_color='#2ca02c',
                text=performance_data['profit'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' + t['total_profit'] + ': $%{y:,.0f}<br>' + t['avg_margin'] + ': ' +
                              performance_data['profit_margin'].round(1).astype(str) + '%<extra></extra>'
            ))
            fig.update_layout(
                title=title,
                xaxis_title=dimension,
                yaxis_title="Amount ($)" if lang == 'english' else "金额 ($)",
                barmode='group',
                height=400,
                showlegend=True
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write(f"**{t['margin_distribution']}**")

        # Dimension selection for margin analysis
        margin_dimension = st.radio(
            t['view_margin_by'],
            [t['product_category'], t['region'], t['customer_segment']],
            horizontal=True,
            key="margin_dimension"
        )

        if margin_dimension == t['product_category']:
            margin_data = filtered_df.groupby('product_category').agg({
                'profit': 'sum',
                'sales_amount': 'sum'
            }).reset_index()
            margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales_amount'] * 100).round(1)
            category_col = 'product_category'
            title = t['margin_by_category']

        elif margin_dimension == t['region']:
            margin_data = filtered_df.groupby('region').agg({
                'profit': 'sum',
                'sales_amount': 'sum'
            }).reset_index()
            margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales_amount'] * 100).round(1)
            category_col = 'region'
            title = t['margin_by_region']

        elif margin_dimension == t['customer_segment']:
            margin_data = filtered_df.groupby('customer_segment').agg({
                'profit': 'sum',
                'sales_amount': 'sum'
            }).reset_index()
            margin_data['profit_margin'] = (margin_data['profit'] / margin_data['sales_amount'] * 100).round(1)
            category_col = 'customer_segment'
            title = t['margin_by_segment']

        # Create pie chart for margin distribution
        fig = px.pie(margin_data,
                     values='profit',
                     names=category_col,
                     title=title,
                     hover_data=['profit_margin'],
                     labels={'profit_margin': 'Margin %' if lang == 'english' else '利润率 %'},
                     hole=0.3)

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' + t['total_profit'] + ': $%{value:,.0f}<br>' + t[
                'avg_margin'] + ': %{customdata[0]:.1f}%<extra></extra>'
        )

        fig.update_layout(
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Data Explorer Tab
    st.write(f"**{t['interactive_explorer']}**")

    # Quick statistics
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric(t['records_found'], len(filtered_df))
    with stats_col2:
        st.metric(t['date_range'], f"{start_date} {'to' if lang == 'english' else '至'} {end_date}")
    with stats_col3:
        st.metric(t['view_period'], period)
    with stats_col4:
        st.metric(t['regions_active'], len(regions))

    # Search and filter
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_term = st.text_input(t['search_transactions'], placeholder=t['search_placeholder'])
    with search_col2:
        sort_by = st.selectbox(t['sort_by'], ['date', 'sales_amount', 'profit', 'profit_margin'])

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
        label=t['export_data'],
        data=csv,
        file_name=f"sales_data_{start_date}_to_{end_date}.csv",
        mime="text/csv"
    )

# QUICK INSIGHTS IN SIDEBAR
st.sidebar.markdown("---")
st.sidebar.subheader(t['quick_insights'])

if not filtered_df.empty:
    # Top performer
    top_region = filtered_df.groupby('region')['profit'].sum().idxmax()
    top_region_profit = filtered_df.groupby('region')['profit'].sum().max()
    st.sidebar.success(f"**{t['top_region']}**: {top_region} (${top_region_profit:,.0f})")

    # Best margin
    best_margin_product = filtered_df.groupby('product_category')['profit_margin'].mean().idxmax()
    best_margin = filtered_df.groupby('product_category')['profit_margin'].mean().max()
    st.sidebar.info(f"**{t['best_margin']}**: {best_margin_product} ({best_margin:.1f}%)")

    # Customer segment insight
    best_segment = filtered_df.groupby('customer_segment')['profit_margin'].mean().idxmax()
    best_segment_margin = filtered_df.groupby('customer_segment')['profit_margin'].mean().max()
    st.sidebar.info(
        f"**{t['best_segment']}**: {best_segment} ({best_segment_margin:.1f}% {'margin' if lang == 'english' else '利润率'})")

    # Alert for low performance
    lowest_margin = filtered_df.groupby('product_category')['profit_margin'].mean().min()
    if lowest_margin < 10:
        st.sidebar.warning(
            f"**{t['watch']}**: {'Some products below 10% margin' if lang == 'english' else '部分产品利润率低于10%'}")
else:
    st.sidebar.info(t['no_data'])

# Footer
st.markdown("---")
st.caption(
    f"{t['footer']} | {period} {'view' if lang == 'english' else '视图'} | {start_date} {'to' if lang == 'english' else '至'} {end_date} | {len(filtered_df)} {'transactions' if lang == 'english' else '笔交易'}")