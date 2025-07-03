import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
        api_base_url = st.text_input("API Base URL", "http://localhost:8000")
        
        # Endpoint selection
        endpoint = st.selectbox(
            "Endpoint",
            ["/recommend_price", "/status", "/batch_recommendations"]
        )
        
        # HTTP Method
        http_method = st.selectbox(
            "HTTP Method",
            ["POST", "GET"]
        )
        
        # Parameters based on endpoint selection
        st.markdown("#### Request Parameters")
        
        if endpoint == "/recommend_price":
            # Product parameters
            demand = st.slider("Demand Score", 0.1, 1.0, 0.5, 0.05)
            inventory = st.slider("Inventory Level", 0.1, 1.0, 0.5, 0.05)
            comp_price = st.slider("Competitor Price ($)", 50.0, 500.0, 200.0, 10.0)
            user_type = st.selectbox("User Type", ["new", "loyal", "premium"])
            season = st.selectbox("Season", ["winter", "spring", "summer", "fall"])
            base_price = st.slider("Base Price ($)", 50.0, 500.0, 200.0, 10.0)
            
            # Current datetime
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            
            # Create request body
            request_body = {
                "demand_score": demand,
                "inventory_level": inventory,
                "competitor_price": comp_price,
                "user_type": user_type,
                "season": season.lower(),
                "base_price": base_price,
                "timestamp": now
            }
        else:
            # For other endpoints
            request_body = {}
        
        # Headers
        st.markdown("#### Headers")
        include_auth = st.checkbox("Include Authentication", value=False)
        
        headers = {}
        if include_auth:
            auth_type = st.selectbox("Auth Type", ["API Key", "Bearer Token"])
            if auth_type == "API Key":
                api_key = st.text_input("API Key", "your_api_key_here")
                headers["X-API-Key"] = api_key
            else:
                token = st.text_input("Bearer Token", "your_token_here")
                headers["Authorization"] = f"Bearer {token}"
        
        # Display request body
        with st.expander("View Request Body"):
            st.json(request_body)
        
        # Make API call
        call_api = st.button("Send Request")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with api_col2:
        # Request/Response area
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        if call_api:
            try:
                # Prepare request
                full_url = f"{api_base_url.rstrip('/')}{endpoint}"
                st.markdown(f"**URL:** `{full_url}`")
                
                if http_method == "POST":
                    st.markdown("**Request Body:**")
                    st.json(request_body)
                    
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
                    
                    response = requests.get(
                        full_url,
                        params=params,
                        headers=headers,
                        timeout=5
                    )
                
                # Response status
                st.markdown(f"**Status Code:** `{response.status_code} - {response.reason}`")
                
                # Format and display the JSON response
                try:
                    response_json = response.json()
                    st.markdown("**Response Body:**")
                    st.json(response_json)
                    
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
                            text='Price'
                        )
                        
                        fig.update_layout(
                            title="Price Comparison",
                            yaxis_title="Price ($)",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                except ValueError:
                    # Non-JSON response
                    st.markdown("**Response Body:**")
                    st.text(response.text)
            
            except requests.exceptions.RequestException as e:
                st.error(f"Error making request: {e}")
        else:
            # When no API call has been made
            st.markdown(
                """
                <div style="text-align: center; padding: 50px 0;">
                    <h3 style="color: #64748b;">API Response</h3>
                    <p style="color: #94a3b8;">Configure your request parameters and click "Send Request"</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # API Documentation Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### API Documentation")
    
    api_docs_tabs = st.tabs(["Endpoints", "Authentication", "Examples"])
    
    with api_docs_tabs[0]:
        st.markdown("""
        #### Available Endpoints
        
        | Endpoint | Method | Description |
        | --- | --- | --- |
        | `/recommend_price` | POST | Get price recommendation |
        | `/status` | GET | Check API status |
        """)
        
    with api_docs_tabs[1]:
        st.markdown("""
        #### Authentication Methods
        
        The API supports two authentication methods:
        
        1. **API Key Authentication**
           - Add your API key to the `X-API-Key` header
           
        2. **Bearer Token Authentication**
           - Add your token to the `Authorization` header
        """)
        
    with api_docs_tabs[2]:
        st.markdown("""
        #### Request Examples
        
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
    
    st.markdown('</div>', unsafe_allow_html=True) 