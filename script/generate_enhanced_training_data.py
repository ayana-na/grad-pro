import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" Advanced Synthetic Training Data Generator")
print("="*70)

np.random.seed(42)

print("\n[1/4] Generating 2000 realistic properties...")

locations = {
    'Los Angeles': {'neighborhoods': 15, 'avg_price': 850000},
    'San Francisco': {'neighborhoods': 12, 'avg_price': 1200000},
    'San Diego': {'neighborhoods': 10, 'avg_price': 700000},
    'Sacramento': {'neighborhoods': 8, 'avg_price': 450000},
    'San Jose': {'neighborhoods': 10, 'avg_price': 950000}
}

property_types = ['APARTMENT', 'VILLA', 'HOUSE', 'STORE', 'HALL']
type_distributions = {
    'APARTMENT': 0.35,
    'HOUSE': 0.30,
    'VILLA': 0.20,
    'HALL': 0.10,
    'STORE': 0.05
}

properties = []
property_id = 1

for location, data in locations.items():
    num_properties = 400
    
    for _ in range(num_properties):
        ptype = np.random.choice(
            list(type_distributions.keys()),
            p=list(type_distributions.values())
        )
        

        price_modifiers = {
            'APARTMENT': np.random.normal(1.0, 0.3),
            'HOUSE': np.random.normal(1.2, 0.35),
            'VILLA': np.random.normal(1.5, 0.4),
            'HALL': np.random.normal(0.8, 0.25),
            'STORE': np.random.normal(0.9, 0.3)
        }
        
        base_price = data['avg_price']
        price = max(150000, base_price * max(0.5, price_modifiers[ptype]))
        
        sqft_by_type = {
            'APARTMENT': np.random.randint(600, 1500),
            'HOUSE': np.random.randint(1500, 4000),
            'VILLA': np.random.randint(3000, 6000),
            'HALL': np.random.randint(2000, 8000),
            'STORE': np.random.randint(800, 3000)
        }
        
        bedrooms_by_type = {
            'APARTMENT': np.random.randint(1, 4),
            'HOUSE': np.random.randint(3, 6),
            'VILLA': np.random.randint(4, 7),
            'HALL': 0,
            'STORE': 0
        }
        
        bathrooms_by_type = {
            'APARTMENT': np.random.randint(1, 3),
            'HOUSE': np.random.randint(2, 4),
            'VILLA': np.random.randint(3, 5),
            'HALL': 1,
            'STORE': 1
        }
        
        listing_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        days_on_market = (datetime.now() - listing_date).days
        
        properties.append({
            'property_id': property_id,
            'location': location,
            'neighborhood': np.random.randint(1, data['neighborhoods'] + 1),
            'price': round(price, 2),
            'sqft': sqft_by_type[ptype],
            'bedrooms': bedrooms_by_type[ptype],
            'bathrooms': bathrooms_by_type[ptype],
            'type': ptype,
            'listing_date': listing_date.strftime('%Y-%m-%d'),
            'days_on_market': max(0, days_on_market)
        })
        property_id += 1

properties_df = pd.DataFrame(properties)
print(f"✓ Generated {len(properties_df)} properties")
print(f"  Property types: {properties_df['type'].value_counts().to_dict()}")
print(f"  Locations: {properties_df['location'].value_counts().to_dict()}")

print("\n[2/4] Generating 80 specialized employees...")

employee_specializations = {
    'Luxury Properties': {'locations': ['San Francisco', 'Los Angeles'], 'types': ['VILLA', 'HOUSE']},
    'Family Homes': {'locations': ['Los Angeles', 'San Diego', 'San Jose'], 'types': ['HOUSE', 'APARTMENT']},
    'Commercial': {'locations': ['Sacramento', 'San Jose'], 'types': ['STORE', 'HALL']},
    'Budget Properties': {'locations': ['Sacramento', 'San Diego'], 'types': ['APARTMENT', 'STORE']},
    'Premium Markets': {'locations': ['San Francisco'], 'types': ['VILLA', 'HOUSE']}
}

employees = []

for emp_id in range(1, 81):
    specialization = np.random.choice(list(employee_specializations.keys()))
    spec_data = employee_specializations[specialization]
    
    
    productivity = np.random.normal(0.85, 0.10)
    productivity = np.clip(productivity, 0.6, 0.98)
    
    years_experience = np.random.randint(1, 15)
    performance_boost = years_experience * 0.02
    productivity = min(0.98, productivity + performance_boost)
    

    total_deals = int(np.random.normal(20 + years_experience * 2, 5))
    deals_won = int(total_deals * (0.5 + productivity * 0.3))
    deals_lost = total_deals - deals_won
    
    response_time = max(1, np.random.normal(12, 5))  
    
    employees.append({
        'employee_id': emp_id,
        'name': f'Employee_{emp_id}',
        'type': 'SALES',
        'preferred_location': spec_data['locations'][0],
        'specialization': specialization,
        'preferred_property_type': spec_data['types'][0],
        'productivity': round(productivity, 3),
        'average_response_time': round(response_time, 2),
        'deals_won': deals_won,
        'deals_lost': deals_lost,
        'total_deals_closed': total_deals,
        'years_experience': years_experience
    })

employees_df = pd.DataFrame(employees)
print(f"✓ Generated {len(employees_df)} employees")
print(f"  Specializations: {employees_df['specialization'].value_counts().to_dict()}")
print(f"  Productivity range: {employees_df['productivity'].min():.2f} - {employees_df['productivity'].max():.2f}")
print(f"  Avg deals per employee: {employees_df['deals_won'].mean():.1f} won, {employees_df['deals_lost'].mean():.1f} lost")

print("\n[3/4] Generating 3000 realistic deals...")

deals = []
deal_id = 1

for _ in range(3000):
    
    client_location = np.random.choice(list(locations.keys()))
    client_budget_min = np.random.randint(200000, 800000)
    client_budget_max = client_budget_min + np.random.randint(100000, 1000000)
    client_property_type = np.random.choice(property_types, p=[0.35, 0.30, 0.20, 0.10, 0.05])
    

    property_idx = np.random.randint(0, len(properties_df))
    property_data = properties_df.iloc[property_idx]
    
    
    match_prob = np.random.random()
    
    if match_prob < 0.40:

        candidate_employees = employees_df[
            (employees_df['preferred_location'] == client_location) &
            (employees_df['preferred_property_type'] == client_property_type)
        ]
        if len(candidate_employees) > 0:
            selected_employee = candidate_employees.sample(1).iloc[0]
        else:
            selected_employee = employees_df.sample(1).iloc[0]
    elif match_prob < 0.70:
        
        candidate_employees = employees_df[
            employees_df['preferred_location'] == client_location
        ]
        if len(candidate_employees) > 0:
            selected_employee = candidate_employees.sample(1).iloc[0]
        else:
            selected_employee = employees_df.sample(1).iloc[0]
    else:

        selected_employee = employees_df.sample(1).iloc[0]
    
    success_prob = 0.4  
    
    if selected_employee['preferred_location'] == client_location:
        success_prob += 0.15
    

    if selected_employee['preferred_property_type'] == client_property_type:
        success_prob += 0.15
    
    
    if client_budget_min <= property_data['price'] <= client_budget_max:
        success_prob += 0.10
    
    
    success_prob += selected_employee['productivity'] * 0.15
    
    
    success_prob += selected_employee['years_experience'] * 0.01
    
    
    success_prob = np.clip(success_prob + np.random.normal(0, 0.05), 0, 1)
    
    
    status = 'WON' if np.random.random() < success_prob else 'LOST'
    
    deals.append({
        'deal_id': deal_id,
        'client_id': np.random.randint(1, 1000),
        'property_id': property_data['property_id'],
        'employee_id': selected_employee['employee_id'],
        'assigned_employee_id': selected_employee['employee_id'],
        'status': status,
        'client_preferred_budget_min': client_budget_min,
        'client_preferred_budget_max': client_budget_max,
        'client_preferred_location': client_location,
        'client_preferred_property_type': client_property_type,
        'property_price': property_data['price'],
        'property_sqft': property_data['sqft'],
        'property_bedrooms': property_data['bedrooms'],
        'property_bathrooms': property_data['bathrooms'],
        'property_type': property_data['type'],
        'property_location': property_data['location'],
        'property_neighborhood': property_data['neighborhood'],
        'days_on_market': property_data['days_on_market'],
        'employee_productivity': selected_employee['productivity'],
        'employee_response_time': selected_employee['average_response_time'],
        'employee_deals_won': selected_employee['deals_won'],
        'employee_deals_lost': selected_employee['deals_lost'],
        'employee_total_deals': selected_employee['total_deals_closed'],
        'employee_specialization': selected_employee['specialization']
    })
    
    deal_id += 1

deals_df = pd.DataFrame(deals)
win_rate = (deals_df['status'] == 'WON').sum() / len(deals_df)

print(f"✓ Generated {len(deals_df)} deals")
print(f"  Win rate: {win_rate * 100:.1f}%")
print(f"  Deals by status: {deals_df['status'].value_counts().to_dict()}")

print("\n[4/4] Saving datasets...")

properties_df.to_csv('enhanced_properties_v2.csv', index=False)
employees_df.to_csv('enhanced_employees_v2.csv', index=False)
deals_df.to_csv('enhanced_training_data_v2.csv', index=False)

print("✓ Files saved successfully")

print("\n" + "="*70)
print(" Data Generation Complete!")
print("="*70)

print(f"\n Dataset Summary:")
print(f"  • Properties: {len(properties_df)} (5 locations, {len(property_types)} types)")
print(f"  • Employees: {len(employees_df)} (5 specializations)")
print(f"  • Training samples: {len(deals_df)} deals")
print(f"  • Win rate: {win_rate * 100:.1f}%")

print(f"\n Key Features:")
print(f"  • Location-based matching")
print(f"  • Property type specialization")
print(f"  • Employee experience levels")
print(f"  • Budget alignment")
print(f"  • Days on market analysis")
print(f"  • Response time tracking")

print(f"\n Output Files:")
print(f"  • enhanced_properties_v2.csv")
print(f"  • enhanced_employees_v2.csv")
print(f"  • enhanced_training_data_v2.csv")
