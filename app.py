from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import os
from geopy.geocoders import Nominatim
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Currency mapping by country code with exchange rates (USD base)
CURRENCY_MAP = {
    'US': {'code': 'USD', 'symbol': '$', 'name': 'US Dollar', 'rate': 1.0},
    'IN': {'code': 'INR', 'symbol': '₹', 'name': 'Indian Rupee', 'rate': 83.15},
    'GB': {'code': 'GBP', 'symbol': '£', 'name': 'British Pound', 'rate': 0.79},
    'CA': {'code': 'CAD', 'symbol': 'C$', 'name': 'Canadian Dollar', 'rate': 1.36},
    'AU': {'code': 'AUD', 'symbol': 'A$', 'name': 'Australian Dollar', 'rate': 1.52},
    'DE': {'code': 'EUR', 'symbol': '€', 'name': 'Euro', 'rate': 0.92},
    'FR': {'code': 'EUR', 'symbol': '€', 'name': 'Euro', 'rate': 0.92},
    'JP': {'code': 'JPY', 'symbol': '¥', 'name': 'Japanese Yen', 'rate': 148.50},
    'CN': {'code': 'CNY', 'symbol': '¥', 'name': 'Chinese Yuan', 'rate': 7.25},
    'SG': {'code': 'SGD', 'symbol': 'S$', 'name': 'Singapore Dollar', 'rate': 1.35},
    'AE': {'code': 'AED', 'symbol': 'د.إ', 'name': 'UAE Dirham', 'rate': 3.67},
    'MX': {'code': 'MXN', 'symbol': '$', 'name': 'Mexican Peso', 'rate': 17.95},
    'BR': {'code': 'BRL', 'symbol': 'R$', 'name': 'Brazilian Real', 'rate': 4.97},
    'ZA': {'code': 'ZAR', 'symbol': 'R', 'name': 'South African Rand', 'rate': 18.75},
}

DEFAULT_CURRENCY = {'code': 'USD', 'symbol': '$', 'name': 'US Dollar', 'rate': 1.0}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(500))
    area_sqft = db.Column(db.Float)
    bedrooms = db.Column(db.Integer)
    bathrooms = db.Column(db.Integer)
    house_age = db.Column(db.Integer)
    country_code = db.Column(db.String(10))
    currency_code = db.Column(db.String(10))
    currency_symbol = db.Column(db.String(10))
    linear_pred = db.Column(db.Float)
    rf_pred = db.Column(db.Float)
    avg_pred = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Load dataset (using California housing as example)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Train models
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location = data.get('location')
    area_sqft = data.get('area_sqft')
    bedrooms = data.get('bedrooms')
    bathrooms = data.get('bathrooms')
    house_age = data.get('house_age')
    
    # Get address, location type, coordinates, and currency
    address, location_type, lat, lng, country_code, currency = get_location_info(location)
    
    # Create feature vector
    # Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    features = np.array([[
        5.0,  # MedInc - default median income
        house_age,
        bedrooms + bathrooms + 1,  # AveRooms - estimate total rooms
        bedrooms,
        1000,  # Population - default
        3.0,   # AveOccup - default
        lat,
        lng
    ]])
    
    predictions = {}
    for name, model in trained_models.items():
        pred = model.predict(features)[0]
        # Convert from USD to local currency
        converted_price = pred * currency.get('rate', 1.0)
        predictions[name] = converted_price
    
    # Save to database
    avg_price = sum(predictions.values()) / len(predictions)
    new_prediction = Prediction(
        location=location,
        address=address,
        area_sqft=area_sqft,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        house_age=house_age,
        country_code=country_code,
        currency_code=currency['code'],
        currency_symbol=currency['symbol'],
        linear_pred=predictions.get('Linear Regression', 0),
        rf_pred=predictions.get('Random Forest', 0),
        avg_pred=avg_price
    )
    db.session.add(new_prediction)
    db.session.commit()
    
    return jsonify({
        'address': address,
        'location_type': location_type,
        'predictions': predictions,
        'country_code': country_code,
        'currency': currency
    })

def get_location_info(location):
    try:
        # Use Nominatim for geocoding (free)
        geolocator = Nominatim(user_agent="real_estate_app")
        location_data = geolocator.geocode(location, addressdetails=True)
        if location_data:
            address = location_data.address
            lat = location_data.latitude
            lng = location_data.longitude
            
            # Extract country code from address details
            country_code = 'US'  # default
            if hasattr(location_data, 'raw') and 'address' in location_data.raw:
                address_details = location_data.raw['address']
                country_code = address_details.get('country_code', 'US').upper()
            
            # Get currency for country
            currency = CURRENCY_MAP.get(country_code, DEFAULT_CURRENCY)
            
            # Simple classification based on address
            if 'city' in address.lower():
                loc_type = 'City'
            elif 'rural' in address.lower() or 'village' in address.lower():
                loc_type = 'Rural'
            else:
                loc_type = 'Urban'
            return address, loc_type, lat, lng, country_code, currency
        else:
            return "Location not found", "Unknown", 0, 0, 'US', DEFAULT_CURRENCY
    except:
        return "Error getting location", "Unknown", 0, 0, 'US', DEFAULT_CURRENCY

@app.route('/dashboard')
def dashboard():
    # Create some visualizations
    fig1 = px.scatter(df, x='MedInc', y='PRICE', title='Median Income vs Price')
    fig2 = px.histogram(df, x='PRICE', title='Price Distribution')
    
    graphs = [fig1, fig2]
    graphJSON = json.dumps(graphs, cls=PlotlyJSONEncoder)
    
    return render_template('dashboard.html', graphJSON=graphJSON)

@app.route('/compare')
def compare():
    # Compare models
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
    
    return render_template('compare.html', results=results)

@app.route('/history')
def history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)