# House Price Predictor

House Price Prediction using Machine Learning predicts property prices based on features like area, bedrooms, and bathrooms. It uses Linear Regression and Random Forest algorithms to analyze data and estimate prices. The models are trained on historical housing data and compared to find the most accurate prediction method.

## Features

- Location-based price prediction using Google Maps integration
- Automatic classification of rural/urban/city areas
- Data visualization dashboard
- Comparison of multiple machine learning algorithms
- Web interface built with Flask

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up Google Maps API key:
   ```
   export GOOGLE_MAPS_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```
   python app.py
   ```

## Usage

- Go to the home page and enter:
  - Location (address or city)
  - House area in square feet
  - Number of bedrooms
  - Number of bathrooms
  - House age in years
- The app will show the address and location type
- Get price predictions from different models
- View the dashboard for data visualizations

## Enhancements

- Add more features to the prediction model
- Improve location classification accuracy
- Add more visualization options
- Implement user authentication
