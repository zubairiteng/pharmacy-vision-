import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Function to load default datasets
@st.cache_data
def load_default_data_1():
    return pd.read_csv('salesdaily.csv')


@st.cache_data
def load_default_data_2():
    return pd.read_csv('Shelf_Monitoring_and_Cheakout_Efficiency.csv'
       )


# Function to load uploaded files
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.sidebar.error("Unsupported file type! Please upload an Excel or CSV file.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()


# Sidebar for dataset selection
st.sidebar.title("Dataset Selection")
dataset_choice = st.sidebar.radio(
    "Choose a Dataset:",
    ["Sales Daily Dataset", "Shelf Monitoring Dataset"]
)

# Load the appropriate dataset and display dataset-specific radio buttons
if dataset_choice == "Sales Daily Dataset":

    # Load dataset
    data_source = st.sidebar.radio("Choose Data Source:", ("Default Dataset", "Upload Your Own Dataset"))
    if data_source == "Default Dataset":
        data = load_default_data_1()
        st.sidebar.success("Default dataset loaded successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            data = load_uploaded_file(uploaded_file)
            st.sidebar.success("Dataset uploaded successfully!")
        else:
            st.sidebar.warning("Please upload a dataset to proceed.")
            st.stop()
    st.sidebar.subheader("Pharmacy Data Dashboard")
    main_option = st.sidebar.radio(
        "Select Analysis Type",
        ["Sales Trends", "Category-Specific Analysis", "Peak Hours or Days", "Correlation Analysis"]
    )


elif dataset_choice == "Shelf Monitoring Dataset":
    # Load dataset
    data_source = st.sidebar.radio("Choose Data Source:", ("Default Dataset", "Upload Your Own Dataset"))
    if data_source == "Default Dataset":
        data = load_default_data_2()
        st.sidebar.success("Default dataset loaded successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            data = load_uploaded_file(uploaded_file)
            st.sidebar.success("Dataset uploaded successfully!")
        else:
            st.sidebar.warning("Please upload a dataset to proceed.")
            st.stop()
    st.sidebar.subheader("Shelf Monitoring Options")
    view_option = st.sidebar.radio(
        "Select View",
        ["Overall", "Shelf Analysis"]
    )


# Refresh Button
if st.button("Refresh Dashboard"):
    st.experimental_set_query_params()

# Tooltip Message
tooltip_message = (
    "The dataset is a working process. You cannot open the Excel file directly, "
    "and no modifications can be made. You can only add data to existing columns, "
    "and you cannot change the column names."
)
st.markdown(
    f'<span style="color: grey; font-size: 12px; text-decoration: underline;">{tooltip_message}</span>',
    unsafe_allow_html=True
)
# Visualization logic
if dataset_choice == "Sales Daily Dataset":
    st.header("Sales Daily Dataset")
    # Filters section (common for both datasets)
    st.sidebar.header("Filters")
    filtered_data = data.copy()  # Start with a copy of the loaded data

    # Apply Date Filters
    if "datum" in filtered_data.columns:
        filtered_data['datum'] = pd.to_datetime(filtered_data['datum'], errors='coerce')
        start_date = st.sidebar.date_input("Start Date", filtered_data['datum'].min().date())
        end_date = st.sidebar.date_input("End Date", filtered_data['datum'].max().date())
        filtered_data = filtered_data[(filtered_data['datum'] >= pd.Timestamp(start_date)) &
                                      (filtered_data['datum'] <= pd.Timestamp(end_date))]

    # Apply Numeric Filters
    for col in filtered_data.select_dtypes(include=['int64', 'float64']).columns:
        min_val, max_val = filtered_data[col].min(), filtered_data[col].max()
        selected_range = st.sidebar.slider(f"{col} Range", min_val, max_val, (min_val, max_val))
        filtered_data = filtered_data[(filtered_data[col] >= selected_range[0]) &
                                      (filtered_data[col] <= selected_range[1])]

    #
    if main_option == "Sales Trends":

        st.subheader("Sales Trends Over Time")
        # Ensure we only sum numeric columns
        numeric_data = filtered_data.select_dtypes(include=['number'])

        # Chart 1: Total Sales per Category
        category_totals = numeric_data.iloc[:, :8].sum()  # Adjust indexing based on your dataset
        fig1 = px.bar(
            x=category_totals.index,
            y=category_totals.values,
            color=category_totals.index,
            labels={"x": "Category", "y": "Total Sales"},
            title="Total Sales by Category"
        )
        st.plotly_chart(fig1)

        numeric_columns = filtered_data.select_dtypes(include=['number']).columns

        # Create Year-Month column
        filtered_data['Year'] = filtered_data['datum'].dt.year
        filtered_data['Month'] = filtered_data['datum'].dt.month
        filtered_data['Year-Month'] = pd.to_datetime(
            filtered_data[['Year', 'Month']].assign(Day=1), errors='coerce'
        )

        # Group by Year-Month
        monthly_sales = (
            filtered_data.groupby('Year-Month')[numeric_columns].sum().reset_index()
        )

        # Melt for multi-category plotting
        melted_monthly_sales = monthly_sales.melt(
            id_vars=["Year-Month"],
            value_vars=numeric_columns,
            var_name="Category",
            value_name="Sales"
        )

        # Plot the line chart
        fig1 = px.line(
            melted_monthly_sales,
            x="Year-Month",
            y="Sales",
            color="Category",
            title="Monthly Sales Trend",
            labels={"Year-Month": "Month", "Sales": "Total Sales"}
        )
        st.plotly_chart(fig1)


    elif main_option == "Category-Specific Analysis":
        st.subheader(f"Category-Specific Analysis Products")

        # Convert 'datum' to datetime
        filtered_data['datum'] = pd.to_datetime(filtered_data['datum'], format='%d/%m/%Y')

        # Summing sales for each pharmaceutical code
        pharma_codes = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
        sales_by_code = filtered_data[pharma_codes].sum().reset_index()
        sales_by_code.columns = ['Pharmaceutical Code', 'Sales']

        # Sort to get the top and low-performing products
        top_products = sales_by_code.nlargest(10, 'Sales')
        low_products = sales_by_code.nsmallest(10, 'Sales')

        # Add a radio button to select between 'Top' and 'Low'
        category = st.radio(
            "Select Category:",
            ("Top 10 Pharmaceutical Products by Sales", "Low 10 Pharmaceutical Products by Sales")
        )

        # Display the appropriate chart based on the radio button selection
        if category == "Top 10 Pharmaceutical Products by Sales":
            fig_top = px.bar(
                top_products,
                x='Pharmaceutical Code',
                y='Sales',
                title="Top 10 Pharmaceutical Products by Sales",
                color='Pharmaceutical Code',
                labels={'Sales': 'Total Sales', 'Pharmaceutical Code': 'Product Code'}
            )
            st.plotly_chart(fig_top)

        elif category == "Low 10 Pharmaceutical Products by Sales":
            fig_low = px.bar(
                low_products,
                x='Pharmaceutical Code',
                y='Sales',
                title="Low 10 Pharmaceutical Products by Sales",
                color='Pharmaceutical Code',
                labels={'Sales': 'Total Sales', 'Pharmaceutical Code': 'Product Code'}
            )
            st.plotly_chart(fig_low)
        # Pie chart showing the sales contribution of each pharmaceutical code
        fig_pie = px.pie(
            sales_by_code,
            names='Pharmaceutical Code',
            values='Sales',
            title="Sales Contribution by Pharmaceutical Codes",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie)
        # Ensure numeric columns for grouping
        numeric_columns = filtered_data.select_dtypes(include=['number']).columns

        # Convert Year and Month to numeric and filter invalid rows
        filtered_data['Year'] = pd.to_numeric(filtered_data['Year'], errors='coerce')
        filtered_data['Month'] = pd.to_numeric(filtered_data['Month'], errors='coerce')
        filtered_data = filtered_data.dropna(subset=['Year', 'Month'])

        # Create Year-Month column
        filtered_data['Year-Month'] = pd.to_datetime(
            filtered_data[['Year', 'Month']].assign(Day=1), errors='coerce'
        )

        # Group by Year-Month
        monthly_sales = (
            filtered_data.groupby('Year-Month')[numeric_columns].sum().reset_index()
        )

        # Melt for multi-category plotting
        melted_monthly_sales = monthly_sales.melt(
            id_vars=["Year-Month"],
            value_vars=numeric_columns[:8],  # Adjust to include relevant columns
            var_name="Category",
            value_name="Sales"
        )

        # Plot the line chart
        fig2 = px.line(
            melted_monthly_sales,
            x="Year-Month",
            y="Sales",
            color="Category",
            title="Monthly Sales Trend",
            labels={"Year-Month": "Month", "Sales": "Total Sales"},
            facet_col="Category",  # Create facets for each category
            facet_col_wrap=4  # Adjust layout
        )
        st.plotly_chart(fig2)
    elif main_option == "Peak Hours or Days":
        st.subheader("Peak Hours or Days Analysis")

        # Convert 'datum' to datetime and extract Hour and Weekday information
        filtered_data['datum'] = pd.to_datetime(filtered_data['datum'], format='%d/%m/%Y')
        filtered_data['Hour'] = filtered_data['datum'].dt.hour
        filtered_data['Weekday'] = filtered_data['datum'].dt.day_name()

        # Define pharmaceutical codes (make sure it's defined correctly)
        pharma_codes = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

        # Summing sales for each hour of the day
        sales_by_hour = filtered_data.groupby('Hour')[pharma_codes].sum().reset_index()

        # Summing sales for each weekday
        sales_by_weekday = filtered_data.groupby('Weekday')[pharma_codes].sum().reset_index()

        # Sort weekdays in the correct order
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sales_by_weekday['Weekday'] = pd.Categorical(sales_by_weekday['Weekday'], categories=weekday_order,
                                                     ordered=True)
        sales_by_weekday = sales_by_weekday.sort_values('Weekday')

        # Peak Hours Visualization - Line Chart for all Pharma Codes
        fig_hour_all = px.line(
            sales_by_hour,
            x='Hour',
            y=pharma_codes,
            title="Sales by Hour of the Day for All Products",
            labels={'Hour': 'Hour of the Day', 'value': 'Total Sales'},
            markers=True,
            line_group='variable'
        )
        st.plotly_chart(fig_hour_all)

        # Peak Days Visualization - Line Chart for All Pharma Codes
        fig_day_all = px.line(
            sales_by_weekday,
            x='Weekday',
            y=pharma_codes,
            title="Sales by Weekday for All Products",
            labels={'Weekday': 'Day of the Week', 'value': 'Total Sales'},
            markers=True,
            line_group='variable'
        )
        st.plotly_chart(fig_day_all)
    elif main_option == "Correlation Analysis":
        st.subheader("Correlation Analysis Between Categories")
        # Define pharmaceutical codes
        pharma_codes = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

        # -------------------- 1. Correlation Matrix (Using Plotly) --------------------
        # Calculate the correlation between the pharmaceutical codes
        corr = filtered_data[pharma_codes].corr()

        # Convert correlation matrix to a format Plotly can work with
        corr_df = corr.reset_index().melt(id_vars='index')
        corr_df.columns = ['Product1', 'Product2', 'Correlation']

        # Generate the Plotly Scatter Plot for Correlation (use color instead of size)
        fig_corr = px.scatter(
            corr_df,
            x='Product1',
            y='Product2',
            color='Correlation',  # Using color to represent correlation strength
            title="Correlation Between Product Categories",
            labels={'Correlation': 'Correlation Coefficient', 'Product1': 'Product', 'Product2': 'Product'},
            hover_data={'Correlation': True}  # Show correlation values in hover
        )
        st.plotly_chart(fig_corr)

        # -------------------- 2. Scatter Plot Between Two Products --------------------
        # Select two products (e.g., M01AB and M01AE)
        fig_pairwise = px.scatter(
            filtered_data,
            x='M01AB',  # Example product 1
            y='M01AE',  # Example product 2
            title="Scatter Plot: M01AB vs M01AE",
            labels={'M01AB': 'M01AB Sales', 'M01AE': 'M01AE Sales'},
            opacity=0.7
        )
        st.plotly_chart(fig_pairwise)

        # -------------------- 3. Bar Chart for Total Sales by Product --------------------
        # Summing sales for each pharmaceutical code
        sales_by_code = filtered_data[pharma_codes].sum().reset_index()
        sales_by_code.columns = ['Pharmaceutical Code', 'Sales']

        # Bar Chart for Top 5 Products
        fig_bar = px.bar(
            sales_by_code,
            x='Pharmaceutical Code',
            y='Sales',
            title="Total Sales by Pharmaceutical Code",
            color='Sales',
            labels={'Sales': 'Total Sales', 'Pharmaceutical Code': 'Product Code'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar)


elif dataset_choice == "Shelf Monitoring Dataset":
    shelf_data = pd.DataFrame()

    # Convert 'Timestamp' to datetime format and localize timezone if needed
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce').dt.tz_localize('UTC').dt.tz_convert(
            'UTC')

    # Time filters
    if 'Timestamp' in data.columns:
        start_time = st.sidebar.time_input("Start Time", data['Timestamp'].min().time())
        end_time = st.sidebar.time_input("End Time", data['Timestamp'].max().time())
        data.set_index('Timestamp', inplace=True)
        if start_time < end_time:
            filtered_data = data.between_time(start_time, end_time).reset_index()
        else:
            st.error("Please ensure that the start time is earlier than the end time.")
            filtered_data = pd.DataFrame()  # Empty DataFrame to prevent further errors
    else:
        st.error("Timestamp column is missing from the dataset.")
        filtered_data = data

    # Additional Filtering Options
    st.sidebar.subheader("Additional Filters")
    if st.sidebar.checkbox("Show Filters"):
        # Multi-select filters for categorical columns
        for col in filtered_data.select_dtypes(include=['object']).columns:
            unique_values = filtered_data[col].dropna().unique()
            selected_values = st.sidebar.multiselect(f"Filter by {col}", options=unique_values, default=unique_values)
            filtered_data = filtered_data[filtered_data[col].isin(selected_values)]

        # Range filters for numeric columns
        for col in filtered_data.select_dtypes(include=['number']).columns:
            min_val, max_val = filtered_data[col].min(), filtered_data[col].max()
            selected_range = st.sidebar.slider(f"Filter by {col} range", min_val, max_val, (min_val, max_val))
            filtered_data = filtered_data[
                (filtered_data[col] >= selected_range[0]) & (filtered_data[col] <= selected_range[1])]

    st.header("Shelf Monitoring Dataset")
    if view_option == "Overall":
        st.subheader("Overall Summary")

        # Define metrics and dynamically calculate their values
        metrics = {
            "Total Customers": data['Total Customers'].sum() if 'Total Customers' in data.columns else 0,
            "Total Visitors": data['Total Visitors'].sum() if 'Total Visitors' in data.columns else 0,
            "Queue Count": data['Queue Count'].sum() if 'Queue Count' in data.columns else 0,
            "Current Visitor": data['Current Visitor'].sum() if 'Current Visitor' in data.columns else 0
        }

        # Calculate total checks (row count)
        total_checks = len(data)

        # Display metrics with gauges
        st.subheader("Overall Summary Metrics")
        gauge_figures = []
        # Define a list of distinct colors for each gauge
        gauge_colors = ['#00BFFF', '#32CD32', '#FF4500', '#9370DB', '#FF6347']  # Adjusted color list

        # Create a gauge for total checks
        gauge_fig_total = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_checks,
            title={'text': "Total Checks", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, max(metrics.values(), default=0) * 1.1]},  # Set range based on max metric value
                'bar': {'color': gauge_colors[0]}  # Color for the total checks gauge
            },
            number={'font': {'size': 18, 'family': 'Arial', 'weight': 'bold', 'color': 'grey'}}
        ))
        gauge_figures.append(gauge_fig_total)  # Append total checks gauge

        # Create gauges for other metrics
        for (label, value), color in zip(metrics.items(), gauge_colors[1:]):  # Start from the second color
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': label, 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, max(metrics.values(), default=0) * 1.1]},
                    'bar': {'color': color}  # Use distinct color for each gauge
                },
                number={'font': {'size': 18, 'family': 'Arial', 'weight': 'bold', 'color': 'grey'}}
            ))
            gauge_figures.append(fig)

        # Create containers for each row of gauges
        # First row of gauges (including total checks)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(gauge_figures[0], use_container_width=True)  # Total Checks
        with col2:
            st.plotly_chart(gauge_figures[1], use_container_width=True)  # Total Customers
        with col3:
            st.plotly_chart(gauge_figures[2], use_container_width=True)  # Total Visitors

        # Second row of gauges (remaining metrics)
        col4, col5 = st.columns(2)

        with col4:
            st.plotly_chart(gauge_figures[3], use_container_width=True)  # Queue Count
        with col5:
            st.plotly_chart(gauge_figures[4], use_container_width=True)  # Current Visitor

        # Optional: Summary bar chart for all metrics
        st.subheader("Metric Summary - Bar Chart")
        bar_fig = go.Figure(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            text=list(metrics.values()),
            textposition='auto',
            marker_color=gauge_colors  # Use the distinct colors for the bar chart
        ))

        bar_fig.update_layout(
            title="Overall Metric Summary",
            xaxis_title="Metrics",
            yaxis_title="Values",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font=dict(size=24),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18)
        )
        st.plotly_chart(bar_fig)

        # Combined line chart of Customers and Visitors
        fig = go.Figure()
        if 'Timestamp' in filtered_data.columns:

            if 'Total Visitors' in filtered_data.columns:
                fig.add_trace(go.Scatter(x=filtered_data['Timestamp'], y=filtered_data['Total Visitors'],
                                         mode='lines', name='Total Visitors', line=dict(color='#FF4500')))
        fig.update_layout(
            title="Overall Visitors Over Time",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font=dict(size=24),
            xaxis_title_font=dict(size=18),
            yaxis_title_font=dict(size=18)
        )
        st.plotly_chart(fig)


    elif view_option == "Shelf Analysis":
        st.subheader("Shelf Analysis (Minute-Level)")

        # Ensure shelf columns exist in filtered_data
        shelves = [col for col in ['Shelf 1', 'Shelf 2', 'Shelf 3', 'Shelf 4'] if col in filtered_data.columns]

        if shelves:
            # Summing each shelf's usage
            shelf_totals = filtered_data[shelves].sum().reset_index()
            shelf_totals.columns = ['Shelf', 'Count']

            # 1. Total Shelf Usage (Bar Chart)
            fig1 = px.bar(shelf_totals, x='Shelf', y='Count', text='Count', color='Shelf',
                          color_discrete_sequence=['#1E90FF', '#FF6347', '#32CD32', '#FFD700'],
                          # Different colors for each shelf
                          title="Total Shelf Usage", template="plotly_dark")
            fig1.update_traces(texttemplate='%{text}', textposition='outside')
            fig1.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the chart area
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
                xaxis_title_font=dict(size=18),  # Font color removed
                yaxis_title_font=dict(size=18)  # Font color removed
            )
            st.plotly_chart(fig1)

            # 2. Minute-Level Stacked Bar Chart for Shelf Usage
            filtered_data['Minute'] = filtered_data['Timestamp'].dt.strftime('%H:%M')  # Group by minute (hour:minute)
            minute_shelf_usage = filtered_data.groupby('Minute')[shelves].sum().reset_index()

            fig2 = go.Figure()
            colors = ['#6A5ACD', '#FF4500', '#00FA9A', '#FFD700']  # Unique colors for each shelf in stacked bar chart
            for shelf, color in zip(shelves, colors):
                fig2.add_trace(go.Bar(
                    x=minute_shelf_usage['Minute'], y=minute_shelf_usage[shelf], name=shelf,
                    marker_color=color  # Different colors for each shelf
                ))
            fig2.update_layout(
                title="Minute-Level Shelf Usage (Stacked Bar Chart)",
                xaxis_title="Time (Minute)",
                yaxis_title="Usage Count",
                barmode='stack',  # Stacks bars for cumulative view
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the chart area
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
                title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
                xaxis_title_font=dict(size=18),  # Font color removed
                yaxis_title_font=dict(size=18),  # Font color removed
                xaxis=dict(tickformat="%H:%M")
            )
            st.plotly_chart(fig2)

            # 3. Line Chart for Minute-Level Shelf Usage Over Time
            fig3 = go.Figure()
            line_colors = ['#00FF7F', '#FF1493', '#1E90FF', '#FFD700']  # Professional colors for line chart
            for shelf, color in zip(shelves, line_colors):
                fig3.add_trace(go.Scatter(
                    x=filtered_data['Timestamp'], y=filtered_data[shelf],
                    mode='lines+markers',  # Added markers for better visibility
                    name=shelf,
                    line=dict(color=color, width=2)  # Thicker line width and distinct color
                ))
            fig3.update_layout(
                title="Shelf Usage Over Time (Minute-Level Line Chart)",
                xaxis_title="Time",
                yaxis_title="Usage Count",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the chart area
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
                title_font=dict(size=24, weight='bold'),  # Bold weight, color removed
                xaxis_title_font=dict(size=18),  # Font color removed
                yaxis_title_font=dict(size=18)  # Font color removed
            )
            st.plotly_chart(fig3)
