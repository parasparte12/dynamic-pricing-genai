import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from pathlib import Path

st.set_page_config(page_title="Analytics Dashboard", page_icon="📈", layout="wide")

st.title("📈 Analytics Dashboard")
st.markdown("""
Historical analysis of all pricing predictions made through the system.
Monitor model performance, usage patterns, and pricing trends.
""")

# Database connection info
BASE_DIR = Path(__file__).parent.parent.parent
try:
    from sqlalchemy import create_engine, text
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv(BASE_DIR / ".env")
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    if not DATABASE_URL:
        st.error("❌ DATABASE_URL not found in .env file")
        st.stop()
    
    # Create database connection
    engine = create_engine(DATABASE_URL)
    
except Exception as e:
    st.error(f"❌ Could not set up database connection: {e}")
    st.stop()

def get_predictions_data():
    """Fetch all predictions from database."""
    try:
        query = """
        SELECT id, timestamp, endpoint, user_id, prediction, lower_bound, upper_bound, model_version
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            data = result.fetchall()
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['id', 'timestamp', 'endpoint', 'user_id', 
                                        'prediction', 'lower_bound', 'upper_bound', 'model_version'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"❌ Error fetching predictions: {e}")
        return None

def get_predictions_by_endpoint():
    """Get count of predictions by endpoint."""
    try:
        query = """
        SELECT endpoint, COUNT(*) as count
        FROM predictions
        GROUP BY endpoint
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            data = result.fetchall()
        return pd.DataFrame(data, columns=['endpoint', 'count'])
    except Exception as e:
        st.error(f"Error fetching endpoint stats: {e}")
        return None

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Series", "Endpoint Analysis", "Raw Data"])

with tab1:
    st.subheader("System Overview")
    
    df = get_predictions_data()
    endpoint_stats = get_predictions_by_endpoint()
    
    if df is not None and len(df) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(df))
        
        with col2:
            avg_price = df['prediction'].mean()
            st.metric("Avg Predicted Price", f"${avg_price:.2f}")
        
        with col3:
            price_std = df['prediction'].std()
            st.metric("Price Std Dev", f"${price_std:.2f}")
        
        with col4:
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            st.metric("Days of Data", date_range)
        
        st.markdown("---")
        
        # Endpoint breakdown
        if endpoint_stats is not None:
            st.subheader("Predictions by Endpoint")
            
            fig = px.pie(endpoint_stats, values='count', names='endpoint',
                        title='Distribution of Predictions Across Endpoints')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show endpoint stats as table
            endpoint_detail = endpoint_stats.copy()
            endpoint_detail['percentage'] = (endpoint_detail['count'] / endpoint_detail['count'].sum() * 100).round(2)
            st.dataframe(endpoint_detail, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Price Trends Over Time")
    
    df = get_predictions_data()
    
    if df is not None and len(df) > 0:
        # Resample to daily average
        df_daily = df.set_index('timestamp').resample('D')['prediction'].agg(['mean', 'min', 'max', 'count'])
        df_daily = df_daily.reset_index()
        
        # Time series chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_daily['timestamp'],
            y=df_daily['mean'],
            name='Daily Average',
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_daily['timestamp'],
            y=df_daily['max'],
            name='Daily Max',
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=df_daily['timestamp'],
            y=df_daily['min'],
            name='Daily Min',
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.5
        ))
        
        fig.update_layout(
            title='Price Predictions Over Time',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction count over time
        st.subheader("Prediction Volume Over Time")
        
        fig_volume = px.bar(df_daily, x='timestamp', y='count',
                           title='Daily Prediction Count',
                           labels={'count': 'Number of Predictions', 'timestamp': 'Date'})
        st.plotly_chart(fig_volume, use_container_width=True)

with tab3:
    st.subheader("Endpoint Performance Analysis")
    
    df = get_predictions_data()
    
    if df is not None and len(df) > 0:
        # Price range analysis by endpoint
        endpoint_analysis = df.groupby('endpoint').agg({
            'prediction': ['mean', 'std', 'min', 'max'],
            'lower_bound': 'mean',
            'upper_bound': 'mean'
        }).round(2)
        
        st.write("**Price Statistics by Endpoint**")
        st.dataframe(endpoint_analysis, use_container_width=True)
        
        # Box plot of prices by endpoint
        fig = px.box(df, x='endpoint', y='prediction',
                    title='Price Distribution by Endpoint',
                    labels={'endpoint': 'Endpoint', 'prediction': 'Predicted Price ($)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence interval analysis
        st.subheader("Confidence Interval Analysis")
        
        df['ci_width'] = df['upper_bound'] - df['lower_bound']
        ci_by_endpoint = df.groupby('endpoint')['ci_width'].agg(['mean', 'std', 'min', 'max']).round(2)
        
        st.write("**Confidence Interval Width by Endpoint** (smaller = more confident predictions)")
        st.dataframe(ci_by_endpoint, use_container_width=True)

with tab4:
    st.subheader("Raw Prediction Data")
    
    df = get_predictions_data()
    
    if df is not None and len(df) > 0:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_endpoint = st.multiselect(
                "Filter by Endpoint",
                options=df['endpoint'].unique(),
                default=df['endpoint'].unique()
            )
        
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
                key="date_range"
            )
        
        with col3:
            model_versions = st.multiselect(
                "Filter by Model Version",
                options=df['model_version'].unique(),
                default=df['model_version'].unique()
            )
        
        # Apply filters
        filtered_df = df[
            (df['endpoint'].isin(selected_endpoint)) &
            (df['timestamp'].dt.date >= date_range[0]) &
            (df['timestamp'].dt.date <= date_range[1]) &
            (df['model_version'].isin(model_versions))
        ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} predictions")
        
        # Display table with pagination
        display_df = filtered_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['prediction'] = display_df['prediction'].apply(lambda x: f"${x:.2f}")
        display_df['lower_bound'] = display_df['lower_bound'].apply(lambda x: f"${x:.2f}")
        display_df['upper_bound'] = display_df['upper_bound'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"pricing_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("📊 No prediction data available yet. Make some predictions to populate the dashboard!")

# Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔄 Data Refresh")

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.rerun()

st.sidebar.markdown("""
This dashboard automatically refreshes every 5 minutes.
Click 'Refresh Data' to manually update statistics.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Dashboard Features")
st.sidebar.markdown("""
- **Overview**: Key metrics and distribution
- **Time Series**: Price trends and prediction volume
- **Endpoint Analysis**: Performance per endpoint
- **Raw Data**: Detailed view with filters and export
""")
