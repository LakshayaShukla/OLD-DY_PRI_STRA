import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from model import PricingModel
import seaborn as sns
import io
import base64
from PIL import Image
from api_tester import show_api_tester
import random
import time

# Page configuration
st.set_page_config(
    page_title="Dynamic Pricing Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/dynamic-pricing-strategy',
        'Report a bug': 'https://github.com/yourusername/dynamic-pricing-strategy/issues',
        'About': """
        # Dynamic Pricing Strategy Dashboard
        
        This dashboard provides powerful visualization and simulation tools for dynamic pricing optimization.
        
        © 2025 - All rights reserved
        """
    }
)

# CSS styling
def local_css():
    st.markdown("""
    <style>
        /* Main Headers */
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #0891b2; /* Teal color */
            margin-bottom: 1rem;
            letter-spacing: -0.5px;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 500;
            color: #0e7490;
            margin-bottom: 1rem;
        }
        
        /* Card Components */
        .card {
            border-radius: 8px;
            background-color: #FFFFFF;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            padding: 24px;
            margin-bottom: 24px;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 1px solid #f0f0f0;
        }
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        }
        
        /* Metric Containers */
        .metric-container {
            background-color: #f8fafc;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            padding: 18px;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(0,0,0,0.03);
        }
        .metric-container:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            background-color: #f1f5f9;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0891b2; /* Teal color */
            transition: color 0.2s;
        }
        .metric-container:hover .metric-value {
            color: #0e7490; /* Darker teal on hover */
        }
        .metric-label {
            font-size: 0.9rem;
            color: #475569;
            margin-top: 6px;
            font-weight: 500;
        }
        
        /* KPI Boxes */
        .kpi-box {
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            border: 1px solid #e2e8f0;
            transition: all 0.3s;
        }
        .kpi-box:hover {
            background: linear-gradient(135deg, #e0f2fe 0%, #cffafe 100%);
            transform: scale(1.03);
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0e7490;
        }
        .kpi-label {
            font-size: 0.85rem;
            color: #475569;
            margin-top: 5px;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8fafc;
            padding: 8px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f5f9;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e0f2fe;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0891b2 !important;
            color: white !important;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #0891b2;
            color: white;
            border-radius: 6px;
            padding: 4px 20px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0e7490;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:active {
            transform: translateY(0px);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f8fafc;
        }
        
        /* Inputs */
        .stSlider>div>div {
            color: #0891b2;
        }
        .stSlider>div>div>div>div {
            background-color: #0891b2;
        }
        
        /* Other Elements */
        .fullwidth {
            width: 100%;
        }
        
        /* Custom Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #334155;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load the sample data"""
    try:
        df = pd.read_csv("pricing_data.csv", parse_dates=["timestamp"])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please generate the data first using data_generator.py")
        return None

@st.cache_resource
def load_model():
    """Load the trained pricing model"""
    try:
        model = PricingModel()
        model.load_model()
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Using a new model instead.")
        model = PricingModel(model_type="xgboost")
        return model

def create_plotly_figure(fig_type, df, x, y, color=None, title="", xaxis_title="", yaxis_title=""):
    """Create a Plotly figure based on the specified type"""
    if fig_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, opacity=0.7, 
                        title=title)
    elif fig_type == "line":
        fig = px.line(df, x=x, y=y, color=color, 
                     title=title)
    elif fig_type == "bar":
        fig = px.bar(df, x=x, y=y, color=color, 
                    title=title)
    elif fig_type == "box":
        fig = px.box(df, x=x, y=y, color=color, 
                    title=title)
    else:
        fig = px.scatter(df, x=x, y=y, color=color, 
                        title=title)
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title_text="",
        height=500,
    )
    
    return fig

def get_top_similar_products(df, target_features, n=5):
    """Get top similar products based on feature similarity"""
    # Filter features for similarity calculation
    feature_cols = ['demand_score', 'inventory_level', 'competitor_price']
    
    # Extract target values
    target_values = np.array([target_features[col] for col in feature_cols]).reshape(1, -1)
    
    # Create a dataframe with just the feature columns
    df_features = df[feature_cols].values
    
    # Calculate Euclidean distance (simple similarity)
    distances = np.sqrt(np.sum((df_features - target_values)**2, axis=1))
    
    # Get top N similar product indices
    similar_indices = np.argsort(distances)[:n]
    
    # Return similar products
    return df.iloc[similar_indices]

def main():
    local_css()
    
    # Header with API status check
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.markdown('<div class="main-header">Dynamic Pricing Dashboard</div>', unsafe_allow_html=True)
    
    with header_col2:
        # Check API status
        try:
            import requests
            try:
                response = requests.get("http://localhost:8000/status", timeout=2)
                if response.status_code == 200:
                    st.markdown(
                        '<div style="background-color: #4CAF50; color: white; padding: 8px 15px; '
                        'border-radius: 20px; display: inline-block; float: right;">'
                        '<span style="font-size: 10px; margin-right: 5px;">●</span> API Connected</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="background-color: #FFA726; color: white; padding: 8px 15px; '
                        'border-radius: 20px; display: inline-block; float: right;">'
                        '<span style="font-size: 10px; margin-right: 5px;">●</span> API Issue</div>',
                        unsafe_allow_html=True
                    )
            except:
                st.markdown(
                    '<div style="background-color: #F44336; color: white; padding: 8px 15px; '
                    'border-radius: 20px; display: inline-block; float: right;">'
                    '<span style="font-size: 10px; margin-right: 5px;">●</span> API Offline</div>',
                    unsafe_allow_html=True
                )
        except ImportError:
            st.markdown(
                '<div style="background-color: #9E9E9E; color: white; padding: 8px 15px; '
                'border-radius: 20px; display: inline-block; float: right;">'
                '<span style="font-size: 10px; margin-right: 5px;">●</span> Status Unknown</div>',
                unsafe_allow_html=True
            )
            
    # Notification system
    if random.random() < 0.7:  # Show notification 70% of the time
        notification_type = random.choice(["info", "success", "warning"])
        
        if notification_type == "info":
            notification = {
                "title": "Did you know?",
                "message": random.choice([
                    "You can export any chart by clicking the menu in the top-right corner.",
                    "Try different simulation parameters to optimize your pricing strategy.",
                    "The API provides detailed explanations for each price recommendation."
                ]),
                "color": "#2196F3"
            }
        elif notification_type == "success":
            notification = {
                "title": "Pro Tip!",
                "message": random.choice([
                    "Setting prices 5-10% above competitors can maximize revenue for premium users.",
                    "Low inventory levels generally support higher price points.",
                    "Seasonal pricing strategies can increase overall profitability by 15-20%."
                ]),
                "color": "#4CAF50"
            }
        else:
            notification = {
                "title": "Pricing Alert",
                "message": random.choice([
                    "Competitor prices have changed significantly in the last period.",
                    "Demand patterns show unusual fluctuations. Consider reviewing your strategy.",
                    "Price elasticity varies greatly by user segment. Have you optimized per segment?"
                ]),
                "color": "#FF9800"
            }
            
        # Display notification
        with st.container():
            st.markdown(
                f"""
                <div style="
                    position: relative;
                    padding: 10px 15px;
                    background-color: white;
                    border-left: 5px solid {notification['color']};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    border-radius: 4px;
                ">
                    <div style="position: absolute; top: 10px; right: 10px; cursor: pointer; color: #9E9E9E;">✕</div>
                    <h4 style="margin: 0; color: {notification['color']};">{notification['title']}</h4>
                    <p style="margin: 5px 0 0 0;">{notification['message']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Sidebar with custom styling
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="background: linear-gradient(120deg, #0891b2, #0e7490); border-radius: 12px; padding: 15px; margin-bottom: 10px; box-shadow: 0 4px 12px rgba(14, 116, 144, 0.2);">
                <img src="https://img.icons8.com/fluency/96/000000/price-tag.png" width="60" style="filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));">
                <h2 style="color: white; margin: 10px 0 0 0; font-size: 1.5rem;">Dynamic Pricing</h2>
                <p style="color: rgba(255, 255, 255, 0.9); margin: 5px 0 0 0; font-size: 0.9rem;">Intelligence Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # User Profile Section
    st.sidebar.markdown("""
        <div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background-color: #0891b2; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                    <span style="color: white; font-size: 18px;">👤</span>
                </div>
                <div>
                    <div style="font-weight: 600; color: #334155;">Guest User</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Demo Access</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Add a theme selector with icons
    st.sidebar.markdown("<h3 style='margin-bottom: 12px; color: #334155; font-size: 1.1rem;'>🎨 Appearance</h3>", unsafe_allow_html=True)
    theme_options = ["Default", "Light", "Dark", "Blue", "Green"]
    theme_icons = ["🌟", "☀️", "🌙", "🔵", "🌿"]
    theme_display = [f"{icon} {theme}" for icon, theme in zip(theme_icons, theme_options)]
    selected_theme_display = st.sidebar.selectbox("Choose Theme", theme_display, index=0)
    selected_theme = theme_options[theme_display.index(selected_theme_display)]
    
    # Apply theme
    if selected_theme == "Dark":
        st.markdown("""
        <style>
        .main {background-color: #0E1117; color: white;}
        .css-1d391kg {background-color: #1E1E1E;}
        .card {background-color: #262730 !important; border: 1px solid #383B42 !important;}
        .metric-container {background-color: #262730 !important; color: white !important;}
        .metric-value {color: #00A6FB !important;}
        </style>
        """, unsafe_allow_html=True)
    elif selected_theme == "Light":
        st.markdown("""
        <style>
        .main {background-color: #F5F5F5; color: black;}
        .css-1d391kg {background-color: #EEEEEE;}
        .card {background-color: white !important; border: 1px solid #DDDDDD !important;}
        .metric-value {color: #0068C9 !important;}
        </style>
        """, unsafe_allow_html=True)
    elif selected_theme == "Blue":
        st.markdown("""
        <style>
        .main {background-color: #EFF6FF; color: #1E293B;}
        .css-1d391kg {background-color: #DBEAFE;}
        .card {background-color: white !important; border: 1px solid #BFDBFE !important;}
        .metric-value {color: #1D4ED8 !important;}
        </style>
        """, unsafe_allow_html=True)
    elif selected_theme == "Green":
        st.markdown("""
        <style>
        .main {background-color: #F0FDF4; color: #1E293B;}
        .css-1d391kg {background-color: #DCFCE7;}
        .card {background-color: white !important; border: 1px solid #BBF7D0 !important;}
        .metric-value {color: #16A34A !important;}
        </style>
        """, unsafe_allow_html=True)
    
    # Add sidebar options
    st.sidebar.markdown("### Quick Options")
    data_options = st.sidebar.expander("Data Options", expanded=False)
    with data_options:
        refresh_data = st.button("Refresh Data")
        if refresh_data:
            st.cache_data.clear()
            st.success("Data cache cleared!")
            
        regenerate_data = st.button("Regenerate Sample Data")
        if regenerate_data:
            try:
                st.info("Generating new data...")
                import subprocess
                subprocess.run(["python", "data_generator.py"], check=True)
                st.success("New data generated! Refresh the page to load it.")
            except Exception as e:
                st.error(f"Error generating data: {e}")
                
    # Model options    
    model_options = st.sidebar.expander("Model Options", expanded=False)
    with model_options:
        retrain_model = st.button("Retrain Model")
        if retrain_model:
            try:
                st.info("Retraining model...")
                import subprocess
                subprocess.run(["python", "model.py"], check=True)
                st.success("Model retrained successfully!")
            except Exception as e:
                st.error(f"Error retraining model: {e}")
                
    # Help section
    help_section = st.sidebar.expander("Help & Resources", expanded=False)
    with help_section:
        st.markdown("""
        - **Home**: Overview of pricing data
        - **Model Output**: Test price predictions
        - **Simulation**: Run pricing simulations
        - **Insights**: Advanced data analysis
        
        [View Documentation](https://github.com/yourusername/dynamic-pricing-strategy)
        """)
        
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="text-align: center; color: #888888; font-size: 0.8em;">Powered by Dynamic Pricing Engine<br>v1.0.0</div>', 
        unsafe_allow_html=True
    )
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Load model
    model = load_model()
    
    # Enhanced tabs with icons
    tab_icons = ["🏠", "🔮", "📈", "🔌", "📊"]
    tab_titles = ["Home", "Simulation", "Forecast", "API Tester", "Insights"]
    
    # Create tabs with icons
    tabs = st.tabs([f"{icon} {title}" for icon, title in zip(tab_icons, tab_titles)])
    
    # Welcome banner
    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #0891b2, #0e7490); padding: 16px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(14, 116, 144, 0.2);">
            <h2 style="color: white; margin: 0; font-weight: 600;">Welcome to Dynamic Pricing Intelligence</h2>
            <p style="color: rgba(255, 255, 255, 0.9); margin-top: 8px; font-size: 1.1rem;">
                Optimize your pricing strategy with data-driven insights and machine learning
            </p>
            <div style="display: flex; gap: 12px; margin-top: 16px;">
                <div style="background: rgba(255, 255, 255, 0.2); padding: 8px 12px; border-radius: 6px; backdrop-filter: blur(4px);">
                    <span style="color: white; font-size: 0.9rem;">💰 Revenue Optimization</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 8px 12px; border-radius: 6px; backdrop-filter: blur(4px);">
                    <span style="color: white; font-size: 0.9rem;">📊 Demand Analysis</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.2); padding: 8px 12px; border-radius: 6px; backdrop-filter: blur(4px);">
                    <span style="color: white; font-size: 0.9rem;">🔮 Price Forecasting</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Home Tab
    with tabs[0]:
        st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">'
                      f'<div class="metric-value">${df["current_price"].mean():.2f}</div>'
                      '<div class="metric-label">Average Price</div>'
                      '</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-container">'
                      f'<div class="metric-value">${df["competitor_price"].mean():.2f}</div>'
                      '<div class="metric-label">Avg. Competitor Price</div>'
                      '</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-container">'
                      f'<div class="metric-value">{df["demand_score"].mean():.2f}</div>'
                      '<div class="metric-label">Avg. Demand Score</div>'
                      '</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-container">'
                      f'<div class="metric-value">{len(df["product_id"].unique())}</div>'
                      '<div class="metric-label">Total Products</div>'
                      '</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data upload option
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload Your Own Data")
        uploaded_file = st.file_uploader("Upload a CSV file with pricing data", type="csv")
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
                st.success("Data uploaded successfully!")
                st.dataframe(user_df.head(5), use_container_width=True)
                
                # Option to replace current data
                if st.button("Use this data for analysis"):
                    df = user_df
                    st.success("Data replaced successfully!")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Overview visualizations
        st.markdown('<div class="sub-header">Key Visualizations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = create_plotly_figure(
                "scatter", df, "demand_score", "current_price", "user_type",
                "Price vs Demand by User Type", 
                "Demand Score", "Price ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = create_plotly_figure(
                "box", df, "season", "current_price", None,
                "Price Distribution by Season", 
                "Season", "Price ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Simulation Tab (formerly Model Output)
    with tabs[1]:
        st.markdown('<div class="sub-header">Pricing Simulation</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Simulation Input")
            
            # Feature inputs for prediction
            demand = st.slider("Demand Score", 0.1, 1.0, 0.5, 0.05, 
                              help="Current market demand level (0.1 = very low, 1.0 = very high)")
            inventory = st.slider("Inventory Level", 0.1, 1.0, 0.5, 0.05,
                                 help="Current inventory as percentage of maximum capacity")
            comp_price = st.slider("Competitor Price ($)", 50.0, 500.0, 200.0, 10.0,
                                  help="Current average competitor price")
            
            user_segment = st.selectbox("Customer Segment", ["New", "Returning", "Loyal"], 
                                      help="Select customer segment to target")
            user_type = user_segment.lower()  # Convert to lowercase for model compatibility
            if user_type == "returning":
                user_type = "loyal"  # Map returning to loyal for model compatibility
                
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"],
                                 help="Current or target season")
            base_price = st.slider("Base Price ($)", 50.0, 500.0, 200.0, 10.0,
                                  help="Standard base price before adjustments")
            
            # Current datetime
            now = datetime.now()
            
            # Create feature dict
            features = {
                'demand_score': demand,
                'inventory_level': inventory,
                'competitor_price': comp_price,
                'hour': now.hour,
                'day': now.day,
                'month': now.month,
                'day_of_week_num': now.weekday(),
                'user_type': user_type.lower(),
                'season': season.lower(),
                'base_price': base_price
            }
            
            # Add API option
            prediction_method = st.radio("Calculation Method", ["Local Engine", "API Service"],
                                        help="Choose calculation method: Local uses the local model, API uses the pricing service")
            
            # Make prediction
            if st.button("Calculate Optimal Price", key="predict_btn", use_container_width=True):
                with st.spinner("Analyzing market conditions..."):
                    try:
                        if prediction_method == "Local Engine":
                            # Use local model
                            predicted_price = model.predict(pd.DataFrame([features]))
                            
                            # Create simulated revenue_gain and confidence interval
                            avg_price = df["current_price"].mean()
                            revenue_gain = ((predicted_price / avg_price) - 1) * 100
                            
                            # Create simulated result structure similar to API
                            result = {
                                "recommended_price": float(predicted_price),
                                "revenue_gain_estimate": float(revenue_gain),
                                "confidence_interval": [float(predicted_price * 0.95), float(predicted_price * 1.05)],
                                "explanation": {
                                    "factors": {
                                        "demand": features['demand_score'],
                                        "inventory": features['inventory_level'],
                                        "competitor_price": features['competitor_price'],
                                        "user_type": features['user_type'],
                                        "season": features['season']
                                    }
                                }
                            }
                        else:
                            # Use API
                            import requests
                            import json
                            
                            # Prepare API request
                            api_features = {
                                "demand_score": features["demand_score"],
                                "inventory_level": features["inventory_level"],
                                "competitor_price": features["competitor_price"],
                                "user_type": features["user_type"],
                                "season": features["season"],
                                "base_price": features["base_price"],
                                "timestamp": now.isoformat()
                            }
                            
                            try:
                                response = requests.post(
                                    "http://localhost:8000/recommend_price",
                                    json=api_features,
                                    timeout=5
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    predicted_price = result["recommended_price"]
                                    
                                    # Add revenue gain and confidence interval if not present
                                    if "revenue_gain_estimate" not in result:
                                        avg_price = df["current_price"].mean()
                                        result["revenue_gain_estimate"] = float(((predicted_price / avg_price) - 1) * 100)
                                        
                                    if "confidence_interval" not in result:
                                        result["confidence_interval"] = [float(predicted_price * 0.95), float(predicted_price * 1.05)]
                                else:
                                    st.error(f"API Error: Status {response.status_code}")
                                    st.error(response.text)
                                    return
                            except requests.exceptions.RequestException as e:
                                st.error(f"API Connection Error: {e}")
                                st.warning("Falling back to local model")
                                predicted_price = model.predict(pd.DataFrame([features]))
                                
                                # Create simulated result with revenue gain
                                avg_price = df["current_price"].mean()
                                revenue_gain = ((predicted_price / avg_price) - 1) * 100
                                
                                result = {
                                    "recommended_price": float(predicted_price),
                                    "revenue_gain_estimate": float(revenue_gain),
                                    "confidence_interval": [float(predicted_price * 0.95), float(predicted_price * 1.05)],
                                    "explanation": {
                                        "factors": features
                                    }
                                }
                        
                        # Display results with animation
                        result_placeholder = st.empty()
                        
                        # Animated countdown effect
                        for i in range(3):
                            result_placeholder.markdown(
                                f'<div class="metric-container" style="text-align: center;">'
                                f'<div style="font-size: 1.2rem;">Finalizing recommendation...</div>'
                                f'<div class="metric-value" style="font-size: 3rem;">${predicted_price - 2 + i:.2f}</div>'
                                '</div>', 
                                unsafe_allow_html=True
                            )
                            time.sleep(0.3)
                        
                        # Final price
                        result_placeholder.markdown(
                            f'<div class="metric-container" style="text-align: center;">'
                            f'<div class="metric-value" style="font-size: 3rem;">${predicted_price:.2f}</div>'
                            '<div class="metric-label">Recommended Price</div>'
                            '</div>', 
                            unsafe_allow_html=True
                        )
                        
                        # KPI boxes for metrics
                        col1a, col1b, col1c = st.columns(3)
                        
                        with col1a:
                            price_diff = predicted_price - comp_price
                            price_diff_pct = (price_diff / comp_price) * 100
                            color = "#16a34a" if price_diff_pct > 0 else "#ef4444"
                            
                            st.markdown(
                                f"""
                                <div class="kpi-box">
                                    <div class="kpi-label">vs Competition</div>
                                    <div class="kpi-value" style="color: {color};">{price_diff_pct:.1f}%</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        with col1b:
                            revenue_gain = result["revenue_gain_estimate"]
                            color = "#16a34a" if revenue_gain > 0 else "#ef4444"
                            
                            st.markdown(
                                f"""
                                <div class="kpi-box">
                                    <div class="kpi-label">Revenue Gain</div>
                                    <div class="kpi-value" style="color: {color};">{revenue_gain:.1f}%</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        with col1c:
                            # Calculate price elasticity (simplified)
                            elasticity = 0.8 + (1 - demand) * 0.5
                            st.markdown(
                                f"""
                                <div class="kpi-box">
                                    <div class="kpi-label">Price Elasticity</div>
                                    <div class="kpi-value">{elasticity:.2f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Confidence interval
                        low, high = result["confidence_interval"]
                        st.markdown(
                            f"""
                            <div style="text-align: center; margin: 15px 0;">
                                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 5px;">Price Range (95% Confidence)</div>
                                <div style="display: flex; align-items: center; justify-content: center;">
                                    <div style="color: #64748b; font-weight: 500;">${low:.2f}</div>
                                    <div style="height: 4px; background: linear-gradient(90deg, #0891b2 0%, #0e7490 100%); 
                                               width: 100px; margin: 0 10px; border-radius: 2px;"></div>
                                    <div style="color: #64748b; font-weight: 500;">${high:.2f}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Add recommendation explanation
                        if price_diff_pct > 10:
                            explanation = "💰 **Premium pricing** recommended due to high demand and favorable conditions."
                            recommendation_details = "Set higher prices to maximize revenue from less price-sensitive customers."
                        elif price_diff_pct > 0:
                            explanation = "✅ **Value pricing** recommended based on current market position."
                            recommendation_details = "Slight price advantage lets you capture more market share while maintaining margins."
                        elif price_diff_pct > -10:
                            explanation = "🏷️ **Competitive pricing** suggested to maintain market share."
                            recommendation_details = "Keep prices close to competitors to retain customers and stay competitive."
                        else:
                            explanation = "🔥 **Discount pricing** advised to increase demand and competitiveness."
                            recommendation_details = "Lower prices to stimulate demand and improve inventory turnover."
                        
                        st.markdown(f"### {explanation}")
                        st.markdown(recommendation_details)
                        
                        # View Details Expander
                        with st.expander("View Calculation Details", expanded=False):
                            st.json(result)
                    
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.error("Try training the model first")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Market Analysis")
            
            # Create tabs for different visualizations
            viz_tabs = st.tabs(["Price Positioning", "Customer Segments", "Price Elasticity"])
            
            with viz_tabs[0]:
                # Find similar products
                similar_products = get_top_similar_products(df, features)
                
                # Visualize similar products
                fig = px.scatter(
                    df.sample(min(500, len(df))), x="demand_score", y="current_price", 
                    opacity=0.4, color_discrete_sequence=["lightgrey"],
                    labels={"demand_score": "Demand Score", "current_price": "Price ($)"}
                )
                
                fig.add_scatter(
                    x=similar_products["demand_score"], y=similar_products["current_price"],
                    mode="markers", marker=dict(size=12, color="#0891b2"),
                    name="Similar Products"
                )
                
                fig.add_scatter(
                    x=[demand], y=[comp_price],
                    mode="markers", marker=dict(size=18, color="#f97316", symbol="star"),
                    name="Current Product"
                )
                
                fig.update_layout(
                    title="Market Positioning Map",
                    xaxis_title="Demand Score",
                    yaxis_title="Price ($)",
                    height=450,
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display similar products in a clean table
                st.markdown("#### Products with Similar Characteristics")
                similar_df = similar_products[['product_id', 'current_price', 'demand_score', 'inventory_level', 'competitor_price', 'user_type']]
                similar_df.columns = ['Product ID', 'Price ($)', 'Demand', 'Inventory', 'Comp. Price ($)', 'Customer Type']
                st.dataframe(similar_df, use_container_width=True, hide_index=True)
            
            with viz_tabs[1]:
                # User segment analysis
                segment_data = df.groupby('user_type').agg({
                    'current_price': ['mean', 'std'],
                    'demand_score': 'mean',
                    'product_id': 'count'
                }).reset_index()
                
                segment_data.columns = ['Segment', 'Avg Price', 'Price Std', 'Avg Demand', 'Count']
                
                # Map user types to more readable names
                segment_map = {'new': 'New', 'loyal': 'Loyal', 'premium': 'Premium'}
                segment_data['Segment'] = segment_data['Segment'].map(segment_map)
                
                # Calculate price index
                segment_data['Price Index'] = segment_data['Avg Price'] / segment_data['Avg Price'].mean()
                
                # Create bar chart for price by segment
                fig = px.bar(
                    segment_data, x='Segment', y='Avg Price',
                    color='Segment', 
                    text_auto='.2f',
                    color_discrete_map={'New': '#94a3b8', 'Loyal': '#0891b2', 'Premium': '#0e7490'}
                )
                
                fig.update_layout(
                    title="Average Price by Customer Segment",
                    xaxis_title="Customer Segment",
                    yaxis_title="Average Price ($)",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a scatter plot for demand vs price by segment
                fig2 = px.scatter(
                    df, x='demand_score', y='current_price', color='user_type',
                    color_discrete_map={'new': '#94a3b8', 'loyal': '#0891b2', 'premium': '#0e7490'},
                    opacity=0.7,
                    labels={
                        'demand_score': 'Demand Score',
                        'current_price': 'Price ($)',
                        'user_type': 'Customer Segment'
                    }
                )
                
                # Add trend lines
                fig2.update_layout(
                    title="Price vs Demand by Customer Segment",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with viz_tabs[2]:
                # Price elasticity visualization
                # Group data by demand ranges
                df['demand_bin'] = pd.cut(df['demand_score'], bins=10)
                elasticity_data = df.groupby(['demand_bin', 'user_type']).agg({
                    'current_price': 'mean',
                    'demand_score': 'mean',
                    'product_id': 'count'
                }).reset_index()
                
                # Add a smooth curve for each user type
                elasticity_types = elasticity_data['user_type'].unique()
                
                fig = go.Figure()
                
                colors = {'new': '#94a3b8', 'loyal': '#0891b2', 'premium': '#0e7490'}
                
                for user_type in elasticity_types:
                    data = elasticity_data[elasticity_data['user_type'] == user_type]
                    
                    # Sort by demand score for smooth line
                    data = data.sort_values('demand_score')
                    
                    fig.add_trace(go.Scatter(
                        x=data['demand_score'], 
                        y=data['current_price'],
                        mode='lines+markers',
                        name=user_type.capitalize(),
                        line=dict(color=colors[user_type], width=3),
                        marker=dict(size=8, color=colors[user_type])
                    ))
                
                # Add marker for current demand
                fig.add_trace(go.Scatter(
                    x=[demand],
                    y=[comp_price],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='#f97316',
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    name='Current Position'
                ))
                
                fig.update_layout(
                    title="Price Elasticity by Customer Segment",
                    xaxis_title="Demand Score",
                    yaxis_title="Price ($)",
                    height=450,
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Elasticity explanation
                st.markdown("""
                    #### Understanding Price Elasticity
                    
                    Price elasticity measures how sensitive demand is to price changes:
                    
                    - **High elasticity** (steep curve): Small price changes cause large demand changes
                    - **Low elasticity** (flat curve): Demand is less affected by price changes
                    
                    Premium customers typically show lower elasticity than new customers.
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Forecast Tab
    with tabs[2]:
        st.markdown('<div class="sub-header">Price Forecasting</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Revenue & Price Forecast</h3>
            <p>Predict future pricing trends and revenue impact based on current data and market conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns
        forecast_col1, forecast_col2 = st.columns([1, 2])
        
        with forecast_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Forecast Settings")
            
            # Forecast parameters
            forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30,
                                     help="Number of days to forecast into the future")
            
            forecast_product = st.selectbox(
                "Select Product",
                df["product_id"].unique().tolist()[:10],  # Limit to first 10 products
                help="Product to create forecast for"
            )
            
            # Seasonality options
            st.markdown("#### Seasonality Factors")
            weekly_seasonality = st.checkbox("Weekly Patterns", value=True,
                                           help="Include day-of-week patterns in forecast")
            monthly_seasonality = st.checkbox("Monthly Patterns", value=True,
                                            help="Include monthly patterns in forecast")
            
            # Market conditions
            st.markdown("#### Market Conditions")
            market_trend = st.select_slider(
                "Market Trend",
                options=["Strong Decline", "Slight Decline", "Stable", "Slight Growth", "Strong Growth"],
                value="Stable",
                help="Expected overall market trend during forecast period"
            )
            
            # Map trend to numerical values
            trend_values = {
                "Strong Decline": -0.2,
                "Slight Decline": -0.05,
                "Stable": 0.0,
                "Slight Growth": 0.05,
                "Strong Growth": 0.15
            }
            trend_value = trend_values[market_trend]
            
            # Competition intensity
            competition_intensity = st.select_slider(
                "Competition Intensity",
                options=["Very Low", "Low", "Moderate", "High", "Very High"],
                value="Moderate",
                help="Expected competitive pressure during forecast period"
            )
            
            # Map competition to numerical values
            competition_values = {
                "Very Low": 0.1,
                "Low": 0.05,
                "Moderate": 0.0,
                "High": -0.05,
                "Very High": -0.1
            }
            competition_value = competition_values[competition_intensity]
            
            # Generate forecast button
            generate_forecast = st.button("Generate Forecast", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with forecast_col2:
            if generate_forecast:
                with st.spinner("Generating forecast..."):
                    # Create placeholder for forecast visualization
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # Filter data for selected product
                    product_data = df[df["product_id"] == forecast_product].copy()
                    
                    # If not enough data, sample randomly
                    if len(product_data) < 30:
                        # Use all available data and add synthetic points
                        existing_count = len(product_data)
                        needed_count = 30 - existing_count
                        
                        if existing_count > 0:
                            # Use existing product data if available
                            latest_data = product_data.iloc[-1]
                            base_price = latest_data["current_price"]
                            base_demand = latest_data["demand_score"]
                        else:
                            # Use average values if no data for selected product
                            base_price = df["current_price"].mean()
                            base_demand = df["demand_score"].mean()
                    
                    # Sort data by timestamp
                    product_data = product_data.sort_values("timestamp")
                    
                    # Create date range for forecast
                    last_date = pd.to_datetime(product_data["timestamp"].max() if len(product_data) > 0 else pd.Timestamp.now())
                    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
                    
                    # Create synthetic forecast data
                    import numpy as np
                    
                    # Get latest values or use defaults
                    if len(product_data) > 0:
                        last_price = product_data["current_price"].iloc[-1]
                        last_demand = product_data["demand_score"].iloc[-1]
                    else:
                        last_price = df["current_price"].mean()
                        last_demand = df["demand_score"].mean()
                    
                    # Generate synthetic forecast
                    forecast_prices = []
                    forecast_demand = []
                    forecast_revenue = []
                    
                    # Weekly patterns if enabled
                    weekly_factors = {
                        0: 0.98,  # Monday
                        1: 0.97,  # Tuesday
                        2: 0.99,  # Wednesday
                        3: 1.01,  # Thursday
                        4: 1.03,  # Friday
                        5: 1.05,  # Saturday
                        6: 1.02   # Sunday
                    }
                    
                    # Monthly patterns if enabled
                    monthly_factors = {
                        1: 0.95,  # Start of month
                        10: 0.98, 
                        20: 1.05  # End of month
                    }
                    
                    # Generate forecast
                    for i, date in enumerate(future_dates):
                        # Base trend component (slight random walk)
                        day_trend = trend_value * (i / forecast_days)
                        
                        # Apply seasonality if enabled
                        seasonality_factor = 1.0
                        if weekly_seasonality:
                            seasonality_factor *= weekly_factors[date.weekday()]
                        if monthly_seasonality:
                            day_of_month = date.day
                            month_factor = 1.0
                            for threshold, factor in monthly_factors.items():
                                if day_of_month >= threshold:
                                    month_factor = factor
                            seasonality_factor *= month_factor
                        
                        # Competition effect
                        competition_effect = competition_value * (1 + 0.5 * np.sin(i / 10))
                        
                        # Calculate price
                        price = last_price * (1 + day_trend + 0.01 * np.random.randn() + competition_effect)
                        price *= seasonality_factor
                        
                        # Ensure price doesn't change too drastically
                        price = max(price, last_price * 0.8)
                        price = min(price, last_price * 1.2)
                        
                        # Calculate demand (inversely related to price changes)
                        demand_elasticity = 0.8  # Price elasticity
                        price_change_pct = (price / last_price) - 1
                        demand = last_demand * (1 - demand_elasticity * price_change_pct)
                        demand = max(0.1, min(1.0, demand))  # Keep within bounds
                        
                        # Estimated units sold
                        units = 100 * demand * seasonality_factor
                        revenue = price * units
                        
                        forecast_prices.append(price)
                        forecast_demand.append(demand)
                        forecast_revenue.append(revenue)
                    
                    # Create forecast dataframe
                    forecast_df = pd.DataFrame({
                        'date': future_dates,
                        'price': forecast_prices,
                        'demand': forecast_demand,
                        'revenue': forecast_revenue
                    })
                    
                    # Calculate historical prices for comparison if data exists
                    if len(product_data) > 0:
                        historical_dates = product_data["timestamp"].tolist()
                        historical_prices = product_data["current_price"].tolist()
                    else:
                        # Generate synthetic historical data
                        historical_dates = [last_date - pd.Timedelta(days=i) for i in range(min(30, forecast_days), 0, -1)]
                        historical_prices = [last_price * (1 + 0.02 * np.random.randn()) for _ in range(len(historical_dates))]
                    
                    # Create figure for price forecast
                    fig1 = go.Figure()
                    
                    # Add historical prices
                    fig1.add_trace(go.Scatter(
                        x=historical_dates,
                        y=historical_prices,
                        name='Historical Price',
                        line=dict(color='#94a3b8', width=3)
                    ))
                    
                    # Add forecasted prices
                    fig1.add_trace(go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['price'],
                        name='Forecasted Price',
                        line=dict(color='#0891b2', width=3)
                    ))
                    
                    # Add confidence interval (simple +/- 10%)
                    fig1.add_trace(go.Scatter(
                        x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                        y=[p * 1.1 for p in forecast_df['price']] + [p * 0.9 for p in forecast_df['price']][::-1],
                        fill='toself',
                        fillcolor='rgba(8, 145, 178, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                    
                    # Add vertical line to separate historical from forecast
                    fig1.add_shape(
                        type="line",
                        x0=last_date,
                        y0=min(min(historical_prices), min(forecast_df['price']) * 0.9),
                        x1=last_date,
                        y1=max(max(historical_prices), max(forecast_df['price']) * 1.1),
                        line=dict(color="#64748b", width=2, dash="dash"),
                    )
                    
                    # Add annotation for forecast start
                    fig1.add_annotation(
                        x=last_date,
                        y=min(min(historical_prices), min(forecast_df['price']) * 0.9),
                        text="Forecast Start",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=30
                    )
                    
                    fig1.update_layout(
                        title=f"Price Forecast for {forecast_product}",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Create figure for revenue forecast
                    fig2 = go.Figure()
                    
                    # Add revenue forecast
                    fig2.add_trace(go.Bar(
                        x=forecast_df['date'],
                        y=forecast_df['revenue'],
                        name='Forecasted Revenue',
                        marker_color='#0891b2'
                    ))
                    
                    # Add demand line
                    fig2.add_trace(go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['demand'] * max(forecast_df['revenue']) / max(forecast_df['demand']),
                        name='Demand Index',
                        line=dict(color='#f97316', width=3),
                        yaxis='y2'
                    ))
                    
                    fig2.update_layout(
                        title="Revenue & Demand Forecast",
                        xaxis_title="Date",
                        yaxis_title="Revenue ($)",
                        height=350,
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        yaxis2=dict(
                            title="Demand Index",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Add forecast summary metrics
                    avg_price = np.mean(forecast_df['price'])
                    avg_revenue = np.mean(forecast_df['revenue'])
                    max_revenue_idx = np.argmax(forecast_df['revenue'])
                    optimal_price = forecast_df['price'][max_revenue_idx]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div class="kpi-box">
                                <div class="kpi-label">Avg. Forecasted Price</div>
                                <div class="kpi-value">${avg_price:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div class="kpi-box">
                                <div class="kpi-label">Optimal Price Point</div>
                                <div class="kpi-value">${optimal_price:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            f"""
                            <div class="kpi-box">
                                <div class="kpi-label">Avg. Daily Revenue</div>
                                <div class="kpi-value">${avg_revenue:.2f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Forecast insights
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Forecast Insights")
                    
                    # Generate insights based on forecast
                    price_trend = "increasing" if forecast_df['price'].iloc[-1] > forecast_df['price'].iloc[0] else "decreasing"
                    price_change_pct = abs(forecast_df['price'].iloc[-1] / forecast_df['price'].iloc[0] - 1) * 100
                    
                    st.markdown(f"""
                    #### Key Takeaways
                    
                    - Prices are forecasted to be **{price_trend}** by **{price_change_pct:.1f}%** over the next {forecast_days} days.
                    - The optimal price point for maximum revenue is **${optimal_price:.2f}**.
                    - {market_trend} market conditions are the primary driver of the forecast.
                    - {competition_intensity} competition intensity affects pricing flexibility.
                    """)
                    
                    # Recommendations based on forecast
                    st.markdown("#### Recommendations")
                    
                    if price_trend == "increasing" and price_change_pct > 5:
                        st.markdown("""
                        - Consider gradual price increases to maximize revenue
                        - Monitor competitor responses closely
                        - Focus marketing on value proposition to justify higher prices
                        """)
                    elif price_trend == "decreasing" and price_change_pct > 5:
                        st.markdown("""
                        - Prepare for potential revenue impact from price decreases
                        - Consider cost reduction initiatives or volume-based strategies
                        - Explore promotional offers rather than permanent price cuts
                        """)
                    else:
                        st.markdown("""
                        - Maintain current pricing strategy with minor adjustments
                        - Focus on non-price value propositions
                        - Monitor market conditions for potential changes
                        """)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Display placeholder content
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("""
                    <div style="text-align: center; padding: 50px 20px;">
                        <img src="https://img.icons8.com/fluency/96/000000/line-chart.png" width="60" style="opacity: 0.5; margin-bottom: 15px;">
                        <h3 style="color: #64748b; font-weight: 500; margin-bottom: 10px;">Configure Forecast Settings</h3>
                        <p style="color: #94a3b8;">Adjust forecast parameters on the left panel and click "Generate Forecast" to see price and revenue predictions.</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Pricing Strategy Simulator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Simulation parameters
            st.markdown("#### Parameters")
            base_price = st.number_input("Base Product Price ($)", 50.0, 1000.0, 200.0, 10.0)
            elasticity = st.slider("Price Elasticity (sensitivity)", 0.1, 2.0, 1.0, 0.1)
            start_demand = st.slider("Initial Demand Score", 0.3, 0.9, 0.6, 0.05)
            
            # Competitor strategy
            st.markdown("#### Competitor Strategy")
            comp_strategy = st.selectbox(
                "Competitor Pricing Strategy",
                ["Stable", "Gradual Increase", "Gradual Decrease", "Fluctuating"]
            )
            comp_base = st.number_input("Competitor Base Price ($)", 50.0, 1000.0, base_price, 10.0)
            
            # Simulation settings
            st.markdown("#### Settings")
            days = st.slider("Simulation Days", 7, 90, 30)
            
            # Run simulation button
            sim_clicked = st.button("Run Simulation")
        
        with col2:
            # Generate simulation data on button click
            if sim_clicked:
                st.markdown("#### Simulation Results")
                
                # Create date range
                start_date = datetime.now()
                dates = [start_date + timedelta(days=i) for i in range(days)]
                
                # Initialize data
                sim_data = []
                current_demand = start_demand
                
                # Generate competitor prices based on strategy
                if comp_strategy == "Stable":
                    comp_prices = [comp_base] * days
                elif comp_strategy == "Gradual Increase":
                    comp_prices = [comp_base * (1 + 0.01 * i) for i in range(days)]
                elif comp_strategy == "Gradual Decrease":
                    comp_prices = [comp_base * (1 - 0.005 * i) for i in range(days)]
                elif comp_strategy == "Fluctuating":
                    comp_prices = [comp_base * (1 + 0.1 * np.sin(i / 5)) for i in range(days)]
                
                # Initialize prices with base price
                our_prices = [base_price] * days
                optimal_prices = []
                
                # Simulate for each day
                for i in range(days):
                    date = dates[i]
                    comp_price = comp_prices[i]
                    
                    # For the optimal price simulation, we need to create feature inputs
                    features = {
                        'demand_score': current_demand,
                        'inventory_level': 0.5,  # Assuming stable inventory for simulation
                        'competitor_price': comp_price,
                        'hour': date.hour,
                        'day': date.day,
                        'month': date.month,
                        'day_of_week_num': date.weekday(),
                        'user_type': 'loyal',  # Assuming loyal user
                        'season': "winter" if date.month < 3 else "spring" if date.month < 6 else "summer",
                        'base_price': base_price
                    }
                    
                    # Try to get optimal price from model
                    try:
                        optimal_price = model.predict(pd.DataFrame([features]))
                        optimal_prices.append(optimal_price)
                    except:
                        # If model not trained, use a simple formula
                        optimal_price = base_price * (1 + 0.3 * current_demand - 0.2 * (1 - 0.5))
                        optimal_price = max(min(optimal_price, comp_price * 1.2), comp_price * 0.8)
                        optimal_prices.append(optimal_price)
                    
                    # After first day, update our prices based on optimal
                    if i > 0:
                        our_prices[i] = optimal_prices[i-1]
                    
                    # Update demand based on our price vs competitor (simple elasticity model)
                    price_ratio = our_prices[i] / comp_price
                    demand_change = -elasticity * (price_ratio - 1)
                    
                    current_demand = max(0.1, min(0.9, current_demand * (1 + demand_change * 0.1)))
                    
                    # Store simulation data
                    sim_data.append({
                        'date': date,
                        'our_price': our_prices[i],
                        'competitor_price': comp_price,
                        'optimal_price': optimal_prices[i] if i < len(optimal_prices) else our_prices[i],
                        'demand_score': current_demand
                    })
                
                # Convert to dataframe
                sim_df = pd.DataFrame(sim_data)
                
                # Calculate revenue (simplified)
                sim_df['estimated_sales'] = sim_df['demand_score'] * 100
                sim_df['our_revenue'] = sim_df['our_price'] * sim_df['estimated_sales']
                sim_df['competitor_revenue'] = sim_df['competitor_price'] * sim_df['estimated_sales']
                
                # Plot price comparison
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=sim_df['date'], y=sim_df['our_price'],
                    mode='lines+markers', name='Our Dynamic Price',
                    line=dict(color='#1E88E5', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=sim_df['date'], y=sim_df['competitor_price'],
                    mode='lines+markers', name='Competitor Price',
                    line=dict(color='#FFC107', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=sim_df['date'], y=sim_df['optimal_price'],
                    mode='lines', name='Optimal Price',
                    line=dict(color='#4CAF50', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Price Comparison Over Time',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=400,
                    template='plotly_white',
                    legend=dict(orientation='h', y=1.1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Revenue and demand chart
                fig2 = go.Figure()
                
                fig2.add_trace(go.Bar(
                    x=sim_df['date'], y=sim_df['our_revenue'],
                    name='Our Revenue',
                    marker_color='#1E88E5'
                ))
                
                fig2.add_trace(go.Bar(
                    x=sim_df['date'], y=sim_df['competitor_revenue'],
                    name='Competitor Est. Revenue',
                    marker_color='#FFC107'
                ))
                
                fig2.add_trace(go.Scatter(
                    x=sim_df['date'], y=sim_df['demand_score'] * max(sim_df['our_revenue']) / max(sim_df['demand_score']),
                    mode='lines', name='Demand Score (scaled)',
                    line=dict(color='#4CAF50', width=2),
                    yaxis='y2'
                ))
                
                fig2.update_layout(
                    title='Estimated Revenue & Demand',
                    xaxis_title='Date',
                    yaxis_title='Revenue ($)',
                    height=350,
                    template='plotly_white',
                    legend=dict(orientation='h', y=1.1),
                    yaxis2=dict(
                        title='Demand Score',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    barmode='group'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Summary stats
                total_our_revenue = sim_df['our_revenue'].sum()
                total_comp_revenue = sim_df['competitor_revenue'].sum()
                revenue_diff = ((total_our_revenue - total_comp_revenue) / total_comp_revenue) * 100
                
                st.markdown("#### Simulation Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Our Total Revenue", f"${total_our_revenue:,.2f}", 
                              f"{revenue_diff:.1f}% vs competitor")
                with col2:
                    st.metric("Competitor Revenue", f"${total_comp_revenue:,.2f}")
                with col3:
                    st.metric("Avg. Price Difference", 
                             f"${(sim_df['our_price'] - sim_df['competitor_price']).mean():.2f}")
                
                # Show data table
                with st.expander("View Simulation Data"):
                    st.dataframe(sim_df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # API Tester Tab
    with tabs[3]:
        show_api_tester()
        
    # Insights Tab
    with tabs[4]:
        st.markdown('<div class="sub-header">Data Insights</div>', unsafe_allow_html=True)
        
        # Filter data
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_filter = st.multiselect("User Type", df['user_type'].unique(), default=df['user_type'].unique())
        with col2:
            season_filter = st.multiselect("Season", df['season'].unique(), default=df['season'].unique())
        with col3:
            price_range = st.slider("Price Range ($)", 
                                   float(df['current_price'].min()), 
                                   float(df['current_price'].max()), 
                                   (float(df['current_price'].min()), float(df['current_price'].max())))
        
        # Apply filters
        filtered_df = df[
            (df['user_type'].isin(user_filter)) &
            (df['season'].isin(season_filter)) &
            (df['current_price'] >= price_range[0]) &
            (df['current_price'] <= price_range[1])
        ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} records")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced visualizations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Advanced Analysis")
        
        visualization_type = st.selectbox(
            "Select Visualization",
            ["Price Heatmap by Day and Hour", "Price Elasticity", "Competitor Price Comparison", 
             "User Segment Analysis", "Seasonal Trends"]
        )
        
        if visualization_type == "Price Heatmap by Day and Hour":
            # Create hour and day aggregation
            if len(filtered_df) > 0:
                pivot_df = filtered_df.pivot_table(
                    values='current_price', 
                    index=filtered_df['timestamp'].dt.hour, 
                    columns=filtered_df['day_of_week'],
                    aggfunc='mean'
                ).fillna(0)
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df,
                    labels=dict(x="Day of Week", y="Hour of Day", color="Price"),
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    title='Average Price by Day and Hour',
                    height=600,
                    coloraxis_colorbar=dict(title="Price ($)")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("This heatmap shows how prices vary by day of the week and hour of the day. "
                           "Darker colors represent higher prices.")
            else:
                st.warning("Not enough data with current filters for this visualization.")
        
        elif visualization_type == "Price Elasticity":
            # Group by demand ranges and calculate average price
            if len(filtered_df) > 0:
                # Create demand bins
                filtered_df['demand_bin'] = pd.cut(
                    filtered_df['demand_score'], 
                    bins=10, 
                    labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(0, 10, 1)]
                )
                
                # Group by demand bin and user type
                elasticity_df = filtered_df.groupby(['demand_bin', 'user_type']).agg({
                    'current_price': 'mean',
                    'demand_score': 'mean',
                    'product_id': 'count'
                }).reset_index()
                
                elasticity_df.rename(columns={'product_id': 'count'}, inplace=True)
                
                # Create elasticity chart
                fig = px.line(
                    elasticity_df,
                    x='demand_score',
                    y='current_price',
                    color='user_type',
                    markers=True,
                    size='count',
                    hover_data=['demand_bin', 'count'],
                    labels={
                        'current_price': 'Average Price ($)',
                        'demand_score': 'Demand Score',
                        'user_type': 'User Type',
                        'count': 'Number of Products'
                    }
                )
                
                fig.update_layout(
                    title='Price Elasticity by User Type',
                    height=500,
                    template='plotly_white',
                    xaxis_title='Demand Score (Higher = More Demand)',
                    yaxis_title='Average Price ($)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("This chart shows the relationship between price and demand for different user types. "
                           "The slope represents price elasticity - how sensitive demand is to price changes.")
            else:
                st.warning("Not enough data with current filters for this visualization.")
        
        elif visualization_type == "Competitor Price Comparison":
            # Analyze our pricing versus competitor pricing
            if len(filtered_df) > 0:
                # Create price difference column
                filtered_df['price_diff'] = filtered_df['current_price'] - filtered_df['competitor_price']
                filtered_df['price_diff_pct'] = (filtered_df['price_diff'] / filtered_df['competitor_price']) * 100
                
                # Group by product
                product_df = filtered_df.groupby('product_id').agg({
                    'current_price': 'mean',
                    'competitor_price': 'mean',
                    'price_diff': 'mean',
                    'price_diff_pct': 'mean'
                }).sort_values('price_diff_pct').reset_index()
                
                # Create scatter plot
                fig = px.scatter(
                    product_df,
                    x='competitor_price',
                    y='current_price',
                    hover_data=['product_id', 'price_diff', 'price_diff_pct'],
                    color='price_diff_pct',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,
                    labels={
                        'current_price': 'Our Price ($)',
                        'competitor_price': 'Competitor Price ($)',
                        'price_diff_pct': 'Price Difference (%)'
                    }
                )
                
                # Add reference line
                fig.add_trace(
                    go.Scatter(
                        x=[product_df['competitor_price'].min(), product_df['competitor_price'].max()],
                        y=[product_df['competitor_price'].min(), product_df['competitor_price'].max()],
                        mode='lines',
                        line=dict(color='black', width=1, dash='dash'),
                        name='Equal Price'
                    )
                )
                
                fig.update_layout(
                    title='Our Prices vs Competitor Prices',
                    height=600,
                    template='plotly_white',
                    xaxis_title='Competitor Price ($)',
                    yaxis_title='Our Price ($)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price difference distribution
                fig2 = px.histogram(
                    product_df,
                    x='price_diff_pct',
                    nbins=20,
                    labels={'price_diff_pct': 'Price Difference (%)'},
                    color_discrete_sequence=['#1E88E5']
                )
                
                fig2.add_vline(
                    x=0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Equal Price",
                    annotation_position="top"
                )
                
                fig2.update_layout(
                    title='Distribution of Price Differences from Competitor',
                    height=400,
                    template='plotly_white',
                    xaxis_title='Price Difference from Competitor (%)',
                    yaxis_title='Number of Products'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("These charts compare our pricing strategy to our competitors. Points above the line indicate products "
                          "where our price is higher than the competitor's price.")
            else:
                st.warning("Not enough data with current filters for this visualization.")
        
        elif visualization_type == "User Segment Analysis":
            # Analyze pricing strategies for different user segments
            if len(filtered_df) > 0:
                # Group by user type and calculate metrics
                user_df = filtered_df.groupby('user_type').agg({
                    'current_price': ['mean', 'median', 'std', 'min', 'max', 'count'],
                    'demand_score': 'mean',
                    'inventory_level': 'mean',
                    'competitor_price': 'mean'
                }).reset_index()
                
                user_df.columns = ['user_type', 'avg_price', 'median_price', 'price_std', 'min_price', 
                                 'max_price', 'count', 'avg_demand', 'avg_inventory', 'avg_competitor_price']
                
                # Calculate price premium over competitor
                user_df['price_premium'] = ((user_df['avg_price'] / user_df['avg_competitor_price']) - 1) * 100
                
                # Create segmentation chart
                fig = px.scatter(
                    user_df,
                    x='avg_demand',
                    y='price_premium',
                    color='user_type',
                    size='count',
                    hover_data=['avg_price', 'avg_competitor_price', 'count'],
                    text='user_type',
                    labels={
                        'avg_demand': 'Average Demand Score',
                        'price_premium': 'Price Premium over Competitor (%)',
                        'count': 'Number of Products'
                    }
                )
                
                fig.update_traces(
                    textposition='top center',
                    marker=dict(sizemin=15)
                )
                
                fig.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="No Premium",
                    annotation_position="left"
                )
                
                fig.update_layout(
                    title='User Segment Pricing Analysis',
                    height=500,
                    template='plotly_white',
                    xaxis_title='Average Demand (Higher = More Demand)',
                    yaxis_title='Price Premium over Competitor (%)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show user segment table
                st.dataframe(user_df, use_container_width=True)
                
                st.markdown("This analysis shows how pricing strategy differs across user segments, "
                          "including the price premium we charge compared to competitors for each user type.")
            else:
                st.warning("Not enough data with current filters for this visualization.")
        
        elif visualization_type == "Seasonal Trends":
            # Analyze seasonal pricing patterns
            if len(filtered_df) > 0:
                # Group by season and day of week
                season_df = filtered_df.groupby(['season', 'day_of_week']).agg({
                    'current_price': 'mean',
                    'demand_score': 'mean',
                    'competitor_price': 'mean',
                    'product_id': 'count'
                }).reset_index()
                
                # Create seasonal trends chart
                fig = px.line(
                    season_df,
                    x='day_of_week',
                    y='current_price',
                    color='season',
                    markers=True,
                    labels={
                        'current_price': 'Average Price ($)',
                        'day_of_week': 'Day of Week',
                        'season': 'Season'
                    },
                    category_orders={
                        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    }
                )
                
                fig.update_layout(
                    title='Seasonal Price Trends by Day of Week',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Demand by season
                fig2 = px.bar(
                    season_df,
                    x='season',
                    y='demand_score',
                    color='season',
                    barmode='group',
                    labels={
                        'demand_score': 'Average Demand Score',
                        'season': 'Season'
                    }
                )
                
                fig2.update_layout(
                    title='Demand by Season',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("These charts show how prices and demand vary across seasons and days of the week.")
            else:
                st.warning("Not enough data with current filters for this visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data download section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Filtered Data to CSV"):
                csv = filtered_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="filtered_pricing_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("Export Current Visualization"):
                # Create a placeholder for the figure
                st.markdown("Right-click on the visualization and select 'Save as Image' to download")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()