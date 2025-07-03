import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pickle


class PricingModel:
    def __init__(self, model_type="xgboost"):
        """
        Initialize the pricing model
        
        Args:
            model_type: Type of model to use ("xgboost" or "random_forest")
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_importance = None
        
    def preprocess_data(self, df):
        """
        Preprocess the data for model training
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            X: Features DataFrame
            y: Target variable Series
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Extract datetime features
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['day_of_week_num'] = data['timestamp'].dt.dayofweek
        
        # Define target variable (trying to predict optimal price)
        # Here we'll use current_price as our target, assuming it's optimal
        # In real scenarios, you might use a more complex target like revenue-optimized price
        y = data['current_price']
        
        # Features to use for prediction
        numeric_features = [
            'demand_score', 
            'inventory_level', 
            'competitor_price', 
            'hour', 
            'day', 
            'month', 
            'day_of_week_num',
            'base_price'  # Including base price as a reference point
        ]
        
        categorical_features = [
            'user_type', 
            'season',
        ]
        
        # Feature columns
        feature_columns = numeric_features + categorical_features
        
        # Create feature matrix
        X = data[feature_columns]
        
        return X, y
    
    def build_preprocessor(self, X):
        """
        Build a preprocessing pipeline for the data
        
        Args:
            X: Feature DataFrame
            
        Returns:
            preprocessor: Sklearn ColumnTransformer for preprocessing
        """
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessor
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train(self, df, test_size=0.2, random_state=42):
        """
        Train the pricing model
        
        Args:
            df: DataFrame with training data
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            metrics: Dictionary with model performance metrics
        """
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Build preprocessor
        self.preprocessor = self.build_preprocessor(X)
        
        # Choose model
        if self.model_type == "xgboost":
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=5,
                random_state=random_state
            )
        else:  # random_forest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=random_state
            )
        
        # Create pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        if hasattr(self.model['regressor'], 'feature_importances_'):
            self.feature_importance = self.model['regressor'].feature_importances_
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Return metrics
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        return metrics
    
    def predict(self, features):
        """
        Predict price based on input features
        
        Args:
            features: Dictionary or DataFrame with input features
            
        Returns:
            price: Predicted price
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame if dictionary
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Preprocess features
        if 'timestamp' in features.columns:
            features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
            features['day'] = pd.to_datetime(features['timestamp']).dt.day
            features['month'] = pd.to_datetime(features['timestamp']).dt.month
            features['day_of_week_num'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        
        # Make prediction
        price = self.model.predict(features)
        
        # Return price
        return price[0] if len(price) == 1 else price
    
    def save_model(self, filepath="pricing_model.pkl"):
        """
        Save the trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath="pricing_model.pkl"):
        """
        Load a trained model from file
        
        Args:
            filepath: Path to load the model from
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


def perform_eda(df):
    """
    Perform exploratory data analysis on pricing data
    
    Args:
        df: DataFrame with pricing data
        
    Returns:
        figs: Dictionary of matplotlib figures with plots
    """
    figs = {}
    
    # Price distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['current_price'], kde=True, ax=ax1)
    ax1.set_title('Distribution of Current Prices')
    ax1.set_xlabel('Price ($)')
    ax1.set_ylabel('Frequency')
    figs['price_distribution'] = fig1
    
    # Price by user type
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='user_type', y='current_price', data=df, ax=ax2)
    ax2.set_title('Price Distribution by User Type')
    ax2.set_xlabel('User Type')
    ax2.set_ylabel('Price ($)')
    figs['price_by_user_type'] = fig2
    
    # Price by season
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='season', y='current_price', data=df, ax=ax3)
    ax3.set_title('Price Distribution by Season')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Price ($)')
    figs['price_by_season'] = fig3
    
    # Price vs demand
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='demand_score', y='current_price', data=df, alpha=0.5, ax=ax4)
    ax4.set_title('Price vs Demand')
    ax4.set_xlabel('Demand Score')
    ax4.set_ylabel('Price ($)')
    figs['price_vs_demand'] = fig4
    
    # Price vs inventory
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='inventory_level', y='current_price', data=df, alpha=0.5, ax=ax5)
    ax5.set_title('Price vs Inventory Level')
    ax5.set_xlabel('Inventory Level (% of max)')
    ax5.set_ylabel('Price ($)')
    figs['price_vs_inventory'] = fig5
    
    # Price vs competitor price
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='competitor_price', y='current_price', data=df, alpha=0.5, ax=ax6)
    ax6.set_title('Price vs Competitor Price')
    ax6.set_xlabel('Competitor Price ($)')
    ax6.set_ylabel('Current Price ($)')
    figs['price_vs_competitor'] = fig6
    
    # Day of week patterns
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.boxplot(x='day_of_week', y='current_price', data=df, order=day_order, ax=ax7)
    ax7.set_title('Price by Day of Week')
    ax7.set_xlabel('Day of Week')
    ax7.set_ylabel('Price ($)')
    plt.xticks(rotation=45)
    figs['price_by_day'] = fig7
    
    return figs


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv("pricing_data.csv", parse_dates=['timestamp'])
    
    # Perform EDA
    print("Performing EDA...")
    eda_figs = perform_eda(df)
    
    # Save EDA figures
    for name, fig in eda_figs.items():
        fig.savefig(f"eda_{name}.png")
        plt.close(fig)
    
    # Train model
    print("Training model...")
    model = PricingModel(model_type="xgboost")
    metrics = model.train(df)
    
    # Print metrics
    print("\nModel performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    model.save_model() 