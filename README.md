# CO2 Emissions Analysis Web Application

A modern web application for analyzing CO2 emissions data and predicting regional patterns using machine learning.

## Features

### 🌍 Data Visualization
- Interactive charts and graphs using Plotly
- Country-wise CO2 emissions analysis
- Sector-based emission breakdowns
- GDP vs emissions scatter plots
- Renewable energy usage comparisons
- Correlation heatmaps

### 🧠 AI Prediction Engine
- Machine learning model trained on 7 different algorithms
- Best performing model automatically selected
- Real-time predictions for Asia vs Europe classification
- Confidence scores and probability distributions
- User-friendly input forms with validation

### 🎨 Modern UI/UX
- Responsive design with Bootstrap 5
- Dark theme with gradient backgrounds
- Interactive elements and animations
- Mobile-friendly interface
- Professional dashboard layout

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Data Visualization**: Plotly.js
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Icons**: Font Awesome

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone/Download the Project
Make sure you have all the following files in your project directory:
```
├── app.py
├── model.py
├── requirements.txt
├── Co2_Emissions_by_Sectors_Europe-Asia.csv
├── best_model.pkl
├── scaler.pkl
├── model_info.pkl
└── templates/
    ├── base.html
    ├── index.html
    ├── visualization.html
    ├── prediction.html
    └── error.html
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Model Files (if not already present)
If you don't have the pickle files, run the model training first:
```bash
python model.py
```

This will generate:
- `best_model.pkl` - The trained machine learning model
- `scaler.pkl` - The feature scaler
- `model_info.pkl` - Model metadata and feature names

### 4. Run the Web Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

### Homepage
- Overview of the application features
- Navigation to visualization and prediction sections
- Dataset statistics and technology stack information

### Data Visualization (`/visualization`)
- **CO2 Emissions by Country**: Bar chart showing total emissions per country
- **Emissions by Sector**: Pie chart breaking down emissions by industrial sectors
- **GDP vs CO2 Emissions**: Scatter plot showing relationship between economic output and emissions
- **Renewable Energy Usage**: Bar chart comparing renewable energy adoption
- **Emissions Trend Over Time**: Line chart showing temporal patterns (if available)
- **Correlation Heatmap**: Shows relationships between all numerical features

### AI Prediction (`/prediction`)
- Input form for environmental and economic indicators
- Real-time prediction results
- Confidence scores and probability distributions
- Sample data loading for testing
- Feature explanations and help text

## Model Information

The application uses a machine learning pipeline that:

1. **Trains 7 different algorithms**:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Naive Bayes
   - Gradient Boosting

2. **Selects the best performing model** based on accuracy

3. **Uses the top 5 most important features** for prediction

4. **Provides predictions** for whether a region belongs to Asia or Europe

## Dataset

The application uses the `Co2_Emissions_by_Sectors_Europe-Asia.csv` dataset containing:

- **Countries**: Multiple countries from Asia and Europe
- **Emission Data**: Total CO2 emissions and sector-wise breakdowns
- **Economic Indicators**: GDP, population, energy consumption
- **Environmental Metrics**: Renewable energy percentage, urbanization
- **Industry Data**: Industrial growth and transport patterns

## API Endpoints

- `GET /` - Homepage
- `GET /visualization` - Data visualization dashboard
- `GET /prediction` - AI prediction interface
- `POST /predict` - Make predictions via JSON API

### Prediction API Example
```javascript
// POST /predict
{
  "Co2_Emissions_MetricTons": 500.0,
  "Energy_Consumption_TWh": 1200.0,
  "GDP_Billion_USD": 800.0,
  "Population_Millions": 50.0,
  "Renewable_Energy_Percentage": 35.0
}

// Response
{
  "prediction": "Europe",
  "confidence": 87.3,
  "probabilities": {
    "Asia": 12.7,
    "Europe": 87.3
  }
}
```

## Customization

### Adding New Visualizations
1. Add new chart creation logic in `create_visualizations()` function in `app.py`
2. Add corresponding HTML containers in `visualization.html`
3. Add JavaScript rendering code in the template

### Modifying the Model
1. Edit `model.py` to change algorithms or features
2. Re-run training: `python model.py`
3. Restart the web application

### Styling Changes
- Modify CSS variables in `base.html` for color scheme changes
- Update Bootstrap classes for layout modifications
- Add custom CSS in individual templates for specific styling

## Troubleshooting

### Common Issues

1. **Model files not found**
   - Run `python model.py` to generate the pickle files

2. **Dataset not found**
   - Ensure `Co2_Emissions_by_Sectors_Europe-Asia.csv` is in the project directory

3. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`

4. **Port already in use**
   - Change the port in `app.py`: `app.run(debug=True, port=5001)`

### Development Mode
For development with auto-reload:
```bash
export FLASK_ENV=development
python app.py
```

## License

This project is for educational and research purposes.

## Support

For issues or questions, please check:
1. All files are in the correct locations
2. Dependencies are properly installed
3. Python version is 3.8 or higher
4. Model files have been generated successfully 