# Dynamic Pricing Strategy

A comprehensive system for dynamic pricing based on demand, time, user behavior, seasonality, inventory, and competitor pricing.

## 🚀 Overview

This project provides an intelligent pricing engine that can dynamically adjust product or service prices based on multiple factors. It's ideal for e-commerce platforms, travel sites, or any business that can benefit from dynamic pricing strategies.

## ✨ Features

- **Data Generation**: Simulates realistic pricing data with variations based on multiple factors
- **Exploratory Data Analysis**: Visualizes relationships between demand, price, and other factors
- **Machine Learning Model**: Uses XGBoost to predict optimal pricing
- **Interactive Dashboard**: Beautiful Streamlit UI for data visualization and price simulation
- **API Endpoint**: FastAPI backend for price recommendations

## 📋 Project Structure

- `data_generator.py`: Generates synthetic pricing data
- `model.py`: Handles data preprocessing, model training, and predictions
- `dashboard.py`: Streamlit app with interactive visualizations
- `api.py`: FastAPI server for price recommendations

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/dynamic-pricing-strategy.git
cd dynamic-pricing-strategy
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Generate sample data**
```bash
python data_generator.py
```

2. **Train the pricing model**
```bash
python model.py
```

3. **Launch the dashboard**
```bash
streamlit run dashboard.py
```

4. **Start the API server**
```bash
python api.py
```

## 📊 Dashboard Features

The Streamlit dashboard includes:

- **Home**: Overview of data and key metrics
- **Model Output**: Test the price prediction model with different inputs
- **Simulation**: Simulate pricing strategies and compare with competitors
- **Insights**: Advanced data analysis and visualizations

## 🔄 API Endpoints

- `GET /`: API welcome message
- `GET /status`: Check API and model status
- `POST /recommend_price`: Get price recommendation for a single product
- `POST /batch_recommend`: Get price recommendations for multiple products

Example API request:
```json
{
  "demand_score": 0.75,
  "inventory_level": 0.5,
  "competitor_price": 199.99,
  "user_type": "loyal",
  "timestamp": "2023-07-15T12:00:00",
  "season": "summer",
  "base_price": 189.99,
  "product_id": "P001"
}
```

## 📈 Example Usage Workflow

1. Generate synthetic data or upload your own through the dashboard
2. Explore data patterns and relationships in the dashboard
3. Train a pricing model on your data
4. Use the simulation tab to test different pricing strategies
5. Integrate with the API for automated price recommendations

## 📝 License

[MIT](LICENSE)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 