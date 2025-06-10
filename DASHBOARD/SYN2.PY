import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define key parameters
start_date = "2021-01-01"
end_date = "2023-12-31"
total_units = 70000

# Define SKU details
sku_list = {
    'MIC1': {'Category': 'Microphone', 'Domestic_Price': 250},
    'MIC2': {'Category': 'Microphone', 'Domestic_Price': 450},
    'MIC3': {'Category': 'Microphone', 'Domestic_Price': 700},
    'MIC4': {'Category': 'Microphone', 'Domestic_Price': 550},
    'HP1': {'Category': 'Headphone', 'Domestic_Price': 150},
    'HP2': {'Category': 'Headphone', 'Domestic_Price': 250},
    'HP3': {'Category': 'Headphone', 'Domestic_Price': 350},
    'HP4': {'Category': 'Headphone', 'Domestic_Price': 500},
    'HP5': {'Category': 'Headphone', 'Domestic_Price': 180},
    'SPK1': {'Category': 'Speaker', 'Domestic_Price': 600},
    'SPK2': {'Category': 'Speaker', 'Domestic_Price': 400},
    'SPK3': {'Category': 'Speaker', 'Domestic_Price': 300},
}

# Generate dataset
dates = pd.date_range(start=start_date, end=end_date, freq='D')
df = pd.DataFrame({
    'Date': np.random.choice(dates, size=total_units),
    'Product_ID': np.random.choice(list(sku_list.keys()), size=total_units)
})

# Add product details
df['Category'] = df['Product_ID'].map(lambda x: sku_list[x]['Category'])
df['Domestic_Price'] = df['Product_ID'].map(lambda x: sku_list[x]['Domestic_Price'])

# Assign sales channel
df['Sales_Channel'] = np.random.choice(
    ['Online', 'Offline', 'Exports', 'Quick Commerce'],
    size=total_units,
    p=[0.35, 0.35, 0.2, 0.1]
)

# Assign region
def assign_region(channel):
    if channel == 'Exports':
        return np.random.choice(['USA', 'Germany', 'UAE', 'Australia'], p=[0.4, 0.3, 0.2, 0.1])
    else:
        return np.random.choice(['North', 'South', 'East', 'West'], p=[0.3, 0.3, 0.2, 0.2])

df['Region'] = df['Sales_Channel'].apply(assign_region)

# Adjust price for exports
df['Price'] = df.apply(
    lambda x: x['Domestic_Price'] * 1.3 if x['Sales_Channel'] == 'Exports' else x['Domestic_Price'],
    axis=1
)

# Festival multipliers
festival_multipliers = {
    'Diwali': (10, [4, 24, 12], 1.4),
    'Dussehra': (10, [15, 5, 20], 1.3),
    'Prime Day': (7, [26, 13, 15], 1.3),
    'Eid': (4, [13, 2, 21], 1.2),
    'Wedding Season': (11, [1, 1, 1], 1.2)
}

def assign_festival_boost(date):
    for festival, (month, days, mult) in festival_multipliers.items():
        if date.month == month and (festival == 'Wedding Season' or date.day in days):
            return mult
    return 1.0

df['Festival_Multiplier'] = df['Date'].apply(assign_festival_boost)

# Quick Commerce boost
def apply_quick_commerce_boost(row):
    if row['Sales_Channel'] == 'Quick Commerce' and row['Region'] in ['North', 'South']:
        return 1.05
    return 1.0

df['Quick_Commerce_Boost'] = df.apply(apply_quick_commerce_boost, axis=1)

# Units Sold Calculation
def calculate_units(row):
    base_units = np.random.randint(15, 40)
    year_growth = {2021: 1.0, 2022: 1.02, 2023: 1.04}[row['Date'].year]
    units = base_units * year_growth * row['Festival_Multiplier'] * row['Quick_Commerce_Boost']
    return int(units + np.random.normal(0, 5))

df['Units_Sold'] = df.apply(lambda x: calculate_units(x), axis=1)

# Financials
df['Revenue'] = df['Price'] * df['Units_Sold']
df['COGS'] = df.apply(lambda x: x['Revenue'] * 0.75 if x['Sales_Channel'] == 'Exports' else x['Revenue'] * 0.6, axis=1)
df['OpEx'] = df['Revenue'] * 0.2
df['Profit'] = df['Revenue'] - (df['COGS'] + df['OpEx'])

# EBITDA
df['EBITDA'] = (df['Revenue'] - df['COGS'] - df['OpEx']) / df['Revenue']
df['EBITDA'] = df['EBITDA'].clip(lower=0.04, upper=0.12)

# Export data
df.to_csv('final_audio_sales_data_with_profit.csv', index=False)

# Print Column Details and Unique Values
column_details = {
    "Date": "Date of sale",
    "Product_ID": "Unique identifier for each product",
    "Category": "Product category (Microphone, Headphone, Speaker)",
    "Domestic_Price": "Base price in domestic market (in INR)",
    "Sales_Channel": "Sales channel (Online, Offline, Exports, Quick Commerce)",
    "Region": "Sales region (India or export markets)",
    "Price": "Final selling price",
    "Festival_Multiplier": "Multiplier applied during festive sales",
    "Quick_Commerce_Boost": "Boost applied for quick commerce channel",
    "Units_Sold": "Total units sold per transaction",
    "Revenue": "Total revenue generated (Price × Units Sold)",
    "COGS": "Cost of Goods Sold (60-75% of revenue)",
    "OpEx": "Operating expenses (20% of revenue)",
    "Profit": "Total profit (Revenue - COGS - OpEx)",
    "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization"
}

print("\n🔹 Column Name & Description:")
for col, desc in column_details.items():
    print(f"✔ {col}: {desc}")
    unique_values = df[col].unique()
    print(f"   🔹 Unique Values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}\n")

# Print Summary
total_revenue = df['Revenue'].sum()
total_units_sold = df['Units_Sold'].sum()
total_profit = df['Profit'].sum()

print("\n🔹 Dataset Summary:")
print(f"✔ Total Revenue: ₹{total_revenue:,.2f}")
print(f"✔ Total Units Sold: {total_units_sold}")
print(f"✔ Total Profit: ₹{total_profit:,.2f}")
