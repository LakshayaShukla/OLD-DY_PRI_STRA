import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_products=100, n_days=90):
    """
    Generate synthetic e-commerce/travel pricing data
    
    Args:
        n_products: Number of unique products
        n_days: Number of days to generate data for
        
    Returns:
        DataFrame with synthetic pricing data
    """
    
    # Create empty list to hold data rows
    data = []
    
    # Define user types
    user_types = ['new', 'loyal', 'premium']
    
    # Define seasons (assuming a 90-day period covering multiple seasons)
    seasons = {
        0: 'winter', 
        30: 'spring', 
        60: 'summer'
    }
    
    # Generate base price for each product (between $50 and $500)
    product_base_prices = {
        f"P{i:03d}": round(random.uniform(50, 500), 2) 
        for i in range(1, n_products + 1)
    }
    
    # Create timestamps for the past n_days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Weekly demand patterns (0 = Monday, 6 = Sunday)
    weekday_demand_factor = {
        0: 0.9,  # Monday
        1: 0.8,  # Tuesday
        2: 0.7,  # Wednesday
        3: 0.8,  # Thursday
        4: 1.1,  # Friday
        5: 1.3,  # Saturday
        6: 1.2   # Sunday
    }
    
    # Generate data for each product and day
    for product_id, base_price in product_base_prices.items():
        # Add some product-specific randomness
        product_popularity = random.uniform(0.7, 1.3)
        
        # Initial inventory for the product (between 50 and 200 units)
        initial_inventory = int(random.uniform(50, 200))
        current_inventory = initial_inventory
        
        for ts in timestamps:
            # Determine season
            day_of_year = ts.timetuple().tm_yday
            season = seasons.get((day_of_year // 30) * 30, 'winter')
            
            # Season factor influences demand and prices
            season_factor = 1.0
            if season == 'winter':
                season_factor = 0.8
            elif season == 'spring':
                season_factor = 1.1
            elif season == 'summer':
                season_factor = 1.3
            
            # Day of week factor
            day_factor = weekday_demand_factor[ts.weekday()]
            
            # Random demand score (influenced by season and day of week)
            demand_score = round(
                random.uniform(0.3, 1.0) * day_factor * season_factor * product_popularity,
                2
            )
            
            # Calculate inventory change based on demand (higher demand = more sales)
            inventory_change = int(demand_score * 10)
            if inventory_change < current_inventory:
                current_inventory -= inventory_change
            else:
                # If not enough inventory, sell what's available and restock
                current_inventory = max(0, current_inventory)
                if current_inventory < 20:  # Restock threshold
                    current_inventory += int(random.uniform(30, 80))
            
            # Inventory level as percentage of initial inventory
            inventory_level = round(current_inventory / initial_inventory, 2)
            
            # Competitor price (random variation around base price)
            competitor_factor = random.uniform(0.85, 1.15)
            competitor_price = round(base_price * competitor_factor, 2)
            
            # Current price (influenced by demand, season, inventory)
            # Higher demand and lower inventory = higher price
            price_factor = 1.0
            price_factor += (demand_score - 0.5) * 0.3  # Demand influence
            price_factor += (1 - inventory_level) * 0.2  # Inventory influence
            price_factor *= season_factor  # Season influence
            
            current_price = round(base_price * price_factor, 2)
            
            # Randomly assign user type for this data point
            user_type = random.choice(user_types)
            
            # Create data row
            data.append({
                'product_id': product_id,
                'timestamp': ts,
                'demand_score': demand_score,
                'inventory_level': inventory_level,
                'competitor_price': competitor_price,
                'current_price': current_price,
                'user_type': user_type,
                'day_of_week': ts.strftime('%A'),
                'season': season,
                'base_price': base_price
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(n_products=50, n_days=90)
    
    # Save to CSV
    df.to_csv("pricing_data.csv", index=False)
    print(f"Data generated with {len(df)} records and saved to pricing_data.csv")
    
    # Display sample
    print("\nSample data:")
    print(df.head()) 