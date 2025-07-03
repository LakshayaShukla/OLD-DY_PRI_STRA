"""
Script to update dashboard.py to include API tester tab
"""
import fileinput
import re
import os

# Check if api_tester.py exists, if not create it
if not os.path.exists("api_tester.py"):
    print("Creating api_tester.py...")
    with open("api_tester.py", "w") as f:
        f.write('''import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import json
import time
import random

def show_api_tester():
    """Display the API Tester tab content"""
    st.markdown('<div class="sub-header">API Testing Console</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>API Connection Tester</h3>
        <p>Test your dynamic pricing API endpoints and visualize responses in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    api_col1, api_col2 = st.columns([1, 2])
    
    with api_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Request Builder")
        
        # API Settings
        api_base_url = st.text_input("API Base URL", "http://localhost:8000", 
                                     help="Base URL of your dynamic pricing API")
        
        # Endpoint selection
        endpoint = st.selectbox(
            "Endpoint",
            ["/recommend_price", "/status", "/batch_recommendations", "/pricing_history"],
            help="API endpoint to call"
        )
        
        # HTTP Method
        http_method = st.selectbox(
            "HTTP Method",
            ["POST", "GET"],
            help="HTTP method to use for the request"
        )
        
        # Parameters based on endpoint selection
        st.markdown("#### Request Parameters")
        
        if endpoint == "/recommend_price":
            # Product parameters
            demand = st.slider("Demand Score", 0.1, 1.0, 0.5, 0.05)
            inventory = st.slider("Inventory Level", 0.1, 1.0, 0.5, 0.05)
            comp_price = st.slider("Competitor Price ($)", 50.0, 500.0, 200.0, 10.0)
            
            user_segment = st.selectbox("Customer Segment", ["New", "Returning", "Loyal"])
            user_type = user_segment.lower()  # Convert to lowercase for model compatibility
            if user_type == "returning":
                user_type = "loyal"  # Map returning to loyal for model compatibility
                
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
            base_price = st.slider("Base Price ($)", 50.0, 500.0, 200.0, 10.0)
            
            # Current datetime
            now = datetime.now()
            
            # Create request body
            request_body = {
                "demand_score": demand,
                "inventory_level": inventory,
                "competitor_price": comp_price,
                "user_type": user_type.lower(),
                "season": season.lower(),
                "base_price": base_price,
                "timestamp": now.isoformat()
            }
            
        elif endpoint == "/batch_recommendations":
            # For batch endpoint, simulate multiple products
            num_products = st.slider("Number of Products", 1, 10, 3)
            
            # Show an example structure
            st.code("""
{
  "products": [
    {
      "product_id": "product1",
      "demand_score": 0.8,
      "inventory_level": 0.5,
      "competitor_price": 199.99,
      "user_type": "loyal",
      "season": "summer",
      "base_price": 180.0
    },
    {
      "product_id": "product2",
      "demand_score": 0.6,
      "inventory_level": 0.3,
      "competitor_price": 149.99,
      "user_type": "new",
      "season": "summer",
      "base_price": 120.0
    }
  ]
}
            """, language="json")
            
            # Create batch request
            products = []
            for i in range(num_products):
                product = {
                    "product_id": f"product{i+1}",
                    "demand_score": round(random.uniform(0.3, 0.9), 2),
                    "inventory_level": round(random.uniform(0.2, 0.8), 2),
                    "competitor_price": round(random.uniform(100, 300), 2),
                    "user_type": random.choice(["new", "loyal", "premium"]),
                    "season": random.choice(["winter", "spring", "summer", "fall"]),
                    "base_price": round(random.uniform(80, 250), 2)
                }
                products.append(product)
            
            request_body = {"products": products}
            
        elif endpoint == "/pricing_history":
            # For history endpoint
            product_id = st.text_input("Product ID", "product1")
            days = st.slider("Days of History", 1, 30, 7)
            
            request_body = {
                "product_id": product_id,
                "days": days
            }
            
        else:
            # For status and other simple endpoints
            request_body = {}
        
        # Headers
        st.markdown("#### Headers")
        include_auth = st.checkbox("Include Authentication", value=False)
        
        headers = {}
        if include_auth:
            auth_type = st.selectbox("Auth Type", ["API Key", "Bearer Token"])
            if auth_type == "API Key":
                api_key = st.text_input("API Key", "your_api_key_here", type="password")
                headers["X-API-Key"] = api_key
            else:
                token = st.text_input("Bearer Token", "your_token_here", type="password")
                headers["Authorization"] = f"Bearer {token}"
        
        # Display request body
        with st.expander("View Request Body"):
            st.json(request_body)
        
        # Make API call
        call_api = st.button("Send Request", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with api_col2:
        # Request/Response area
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        if call_api:
            # Status indicator
            with st.status("Making API request...", expanded=True) as status:
                try:
                    # Prepare request
                    full_url = f"{api_base_url.rstrip('/')}{endpoint}"
                    st.markdown(f"**URL:** `{full_url}`")
                    
                    if http_method == "POST":
                        st.markdown("**Request Body:**")
                        st.json(request_body)
                        
                        # Make the request with animation
                        st.markdown("**Sending request...**")
                        time.sleep(0.5)  # Simulate network delay
                        
                        response = requests.post(
                            full_url,
                            json=request_body,
                            headers=headers,
                            timeout=5
                        )
                    else:
                        # For GET requests
                        params = request_body if endpoint != "/status" else {}
                        
                        st.markdown("**Request Parameters:**")
                        st.json(params)
                        
                        # Make the request with animation
                        st.markdown("**Sending request...**")
                        time.sleep(0.5)  # Simulate network delay
                        
                        response = requests.get(
                            full_url,
                            params=params,
                            headers=headers,
                            timeout=5
                        )
                    
                    # Response headers
                    st.markdown("**Response Headers:**")
                    headers_dict = dict(response.headers)
                    st.json(headers_dict)
                    
                    # Response status and timing
                    st.markdown(f"**Status Code:** `{response.status_code} - {response.reason}`")
                    st.markdown(f"**Response Time:** `{response.elapsed.total_seconds():.4f} seconds`")
                    
                    # Format and display the JSON response
                    try:
                        response_json = response.json()
                        st.markdown("**Response Body:**")
                        st.json(response_json)
                        
                        # Store the response for visualization
                        if response.status_code == 200:
                            status.update(label="✅ Request completed successfully", state="complete")
                            
                            # For price recommendation response, show visualization
                            if endpoint == "/recommend_price" and "recommended_price" in response_json:
                                recommended_price = response_json["recommended_price"]
                                competitor_price = request_body["competitor_price"]
                                base_price = request_body["base_price"]
                                
                                # Price comparison chart
                                price_data = pd.DataFrame({
                                    'Price Type': ['Base Price', 'Competitor Price', 'Recommended Price'],
                                    'Price': [base_price, competitor_price, recommended_price]
                                })
                                
                                fig = px.bar(
                                    price_data, 
                                    x='Price Type', 
                                    y='Price',
                                    color='Price Type',
                                    color_discrete_map={
                                        'Base Price': '#94a3b8',
                                        'Competitor Price': '#f97316',
                                        'Recommended Price': '#0891b2'
                                    },
                                    text_auto='.2f'
                                )
                                
                                fig.update_layout(
                                    title="Price Comparison",
                                    yaxis_title="Price ($)",
                                    template="plotly_white",
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show explanation if available
                                if "explanation" in response_json:
                                    st.markdown("#### API Explanation")
                                    st.json(response_json["explanation"])
                            
                            # For batch recommendations, show table
                            elif endpoint == "/batch_recommendations" and "recommendations" in response_json:
                                st.markdown("#### Batch Recommendations")
                                
                                # Convert recommendations to DataFrame for display
                                recommendations = response_json["recommendations"]
                                recs_df = pd.json_normalize(recommendations)
                                
                                st.dataframe(recs_df, use_container_width=True)
                                
                                # Generate simple bar chart of recommended prices
                                if len(recommendations) > 0 and "product_id" in recommendations[0] and "recommended_price" in recommendations[0]:
                                    fig = px.bar(
                                        recs_df,
                                        x='product_id',
                                        y='recommended_price',
                                        color='recommended_price',
                                        labels={
                                            'product_id': 'Product ID',
                                            'recommended_price': 'Recommended Price ($)'
                                        },
                                        color_continuous_scale='Viridis'
                                    )
                                    
                                    fig.update_layout(
                                        title="Recommended Prices by Product",
                                        template="plotly_white"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # For pricing history, show timeline
                            elif endpoint == "/pricing_history" and "history" in response_json:
                                st.markdown("#### Price History")
                                
                                # Convert history to DataFrame
                                history = response_json["history"]
                                if len(history) > 0:
                                    history_df = pd.json_normalize(history)
                                    if "date" in history_df.columns and "price" in history_df.columns:
                                        # Convert date strings to datetime
                                        history_df["date"] = pd.to_datetime(history_df["date"])
                                        
                                        fig = px.line(
                                            history_df,
                                            x="date",
                                            y="price",
                                            labels={
                                                "date": "Date",
                                                "price": "Price ($)"
                                            }
                                        )
                                        
                                        fig.update_layout(
                                            title=f"Price History for {request_body['product_id']}",
                                            template="plotly_white"
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show the data table
                                        st.dataframe(history_df, use_container_width=True)
                            
                        else:
                            status.update(label=f"⚠️ Request failed with status {response.status_code}", state="error")
                            
                    except ValueError:
                        # Non-JSON response
                        st.markdown("**Response Body:**")
                        st.text(response.text)
                        
                        if response.status_code == 200:
                            status.update(label="✅ Request completed with non-JSON response", state="complete")
                        else:
                            status.update(label=f"⚠️ Request failed with status {response.status_code}", state="error")
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Error making request: {e}")
                    status.update(label="❌ Request failed", state="error")
        else:
            # When no API call has been made
            st.markdown(
                """
                <div style="text-align: center; padding: 50px 0;">
                    <img src="https://img.icons8.com/fluency/96/000000/api.png" width="60" style="opacity: 0.5; margin-bottom: 15px;">
                    <h3 style="color: #64748b; font-weight: 500; margin-bottom: 10px;">API Response</h3>
                    <p style="color: #94a3b8;">Configure your request parameters and click "Send Request" to see the response here.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Documentation section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### API Documentation")
    
    api_docs_tabs = st.tabs(["Endpoints", "Authentication", "Request Examples", "Response Format"])
    
    with api_docs_tabs[0]:
        st.markdown("""
        #### Available Endpoints
        
        | Endpoint | Method | Description |
        | --- | --- | --- |
        | `/recommend_price` | POST | Get optimal price recommendation for a single product |
        | `/batch_recommendations` | POST | Get price recommendations for multiple products |
        | `/pricing_history` | GET | Retrieve historical pricing data for a product |
        | `/status` | GET | Check API status and health |
        """)
        
    with api_docs_tabs[1]:
        st.markdown("""
        #### Authentication Methods
        
        The API supports two authentication methods:
        
        1. **API Key Authentication**
           - Add your API key to the `X-API-Key` header
           - Example: `X-API-Key: your_api_key_here`
           
        2. **Bearer Token Authentication**
           - Add your token to the `Authorization` header
           - Example: `Authorization: Bearer your_token_here`
        """)
        
    with api_docs_tabs[2]:
        st.markdown("""
        #### Request Examples
        
        ##### Single Price Recommendation
        ```python
        import requests
        
        url = "http://localhost:8000/recommend_price"
        
        payload = {
            "demand_score": 0.7,
            "inventory_level": 0.5,
            "competitor_price": 199.99,
            "user_type": "loyal",
            "season": "summer",
            "base_price": 180.0
        }
        
        response = requests.post(url, json=payload)
        print(response.json())
        ```
        """)
        
    with api_docs_tabs[3]:
        st.markdown("""
        #### Response Format
        
        ##### Single Price Recommendation Response
        ```json
        {
            "recommended_price": 215.50,
            "revenue_gain_estimate": 12.5,
            "confidence_interval": [205.23, 225.77],
            "explanation": {
                "factors": {
                    "demand": 0.7,
                    "inventory": 0.5,
                    "competitor_price": 199.99,
                    "user_type": "loyal",
                    "season": "summer"
                }
            }
        }
        ```
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)''')

# Function to add import line
def add_import_line(dashboard_file="dashboard.py"):
    # Read the file
    with open(dashboard_file, 'r') as file:
        content = file.read()
    
    # Check if import already exists
    if "from api_tester import show_api_tester" not in content:
        # Find the import section
        import_section_end = content.find("# Page configuration")
        if import_section_end == -1:
            import_section_end = content.find("import")
            # Find the last import line
            lines = content[:import_section_end].split('\n')
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith('import') or lines[i].startswith('from'):
                    import_section_end = content.find(lines[i]) + len(lines[i])
                    break
        
        # Insert the import line
        new_content = content[:import_section_end] + "\nfrom api_tester import show_api_tester" + content[import_section_end:]
        
        # Write the modified content back to the file
        with open(dashboard_file, 'w') as file:
            file.write(new_content)
        
        print("Added import line to dashboard.py")
    else:
        print("Import line already exists")

# Function to update tabs section to include API Tester
def update_tabs_section(dashboard_file="dashboard.py"):
    # Read the file
    with open(dashboard_file, 'r') as file:
        content = file.readlines()
    
    # Find the tabs section
    tab_icons_line = -1
    for i, line in enumerate(content):
        if "tab_icons" in line and "[" in line and "]" in line:
            tab_icons_line = i
            break
    
    if tab_icons_line != -1:
        # Check if we already have the right tabs
        if "API Tester" in content[tab_icons_line + 1] or "API Tester" in content[tab_icons_line]:
            print("Tabs already updated")
        else:
            # Update tab icons and titles
            content[tab_icons_line] = '    tab_icons = ["🏠", "🔮", "📈", "🔌", "📊"]\n'
            content[tab_icons_line + 1] = '    tab_titles = ["Home", "Simulation", "Forecast", "API Tester", "Insights"]\n'
            
            # Write the modified content back to the file
            with open(dashboard_file, 'w') as file:
                file.writelines(content)
            
            print("Updated tab icons and titles")
    else:
        print("Could not find tab_icons line")

# Function to add API Tester tab implementation
def add_api_tester_tab(dashboard_file="dashboard.py"):
    # Read the file
    with open(dashboard_file, 'r') as file:
        content = file.readlines()
    
    # Find the Insights Tab section
    insights_tab_line = -1
    for i, line in enumerate(content):
        if "# Insights Tab" in line:
            insights_tab_line = i
            break
    
    if insights_tab_line != -1:
        # Check if we already have the API Tester tab
        for i in range(max(0, insights_tab_line-20), insights_tab_line):
            if "# API Tester Tab" in content[i]:
                print("API Tester tab already added")
                return
        
        # Add API Tester tab before Insights tab
        content.insert(insights_tab_line, "    # API Tester Tab\n")
        content.insert(insights_tab_line + 1, "    with tabs[3]:\n")
        content.insert(insights_tab_line + 2, "        show_api_tester()\n")
        content.insert(insights_tab_line + 3, "\n")
        
        # Update the Insights tab index
        content[insights_tab_line + 4] = "    # Insights Tab\n"
        content[insights_tab_line + 5] = "    with tabs[4]:\n"
        
        # Write the modified content back to the file
        with open(dashboard_file, 'w') as file:
            file.writelines(content)
        
        print("Added API Tester tab implementation")
    else:
        print("Could not find Insights Tab section")

# Main execution
if __name__ == "__main__":
    add_import_line()
    update_tabs_section()
    add_api_tester_tab()
    print("Dashboard.py updated successfully with API Tester tab") 