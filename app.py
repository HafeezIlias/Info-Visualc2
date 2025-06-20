from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json

app = Flask(__name__)

# Load the trained model and components
try:
    model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model_info = pickle.load(open('model_info.pkl', 'rb'))
    print(f"✅ Model loaded: {model_info['model_name']} (Accuracy: {model_info['accuracy']:.4f})")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load dataset
try:
    df = pd.read_csv('Co2_Emissions_by_Sectors_Europe-Asia.csv')
    print(f"✅ Dataset loaded: {len(df)} rows")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = None

def load_data():
    """Load and return the dataset."""
    global df
    if df is None:
        try:
            df = pd.read_csv('Co2_Emissions_by_Sectors_Europe-Asia.csv')
            print(f"✅ Dataset reloaded: {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            df = pd.DataFrame()  # Return empty dataframe if loading fails
    return df

@app.route('/')
def home():
    """Homepage with navigation to both sections."""
    return render_template('index.html')

@app.route('/visualization')
def visualization():
    """Data visualization page with interactive charts."""
    if df is None:
        return render_template('error.html', message="Dataset not available")
    
    # Get continent parameter
    selected_continent = request.args.get('continent', 'all')
    
    # Apply continent filter to dataframe
    filtered_df = df.copy()
    if selected_continent != 'all':
        filtered_df = filtered_df[filtered_df['Continent'].str.lower() == selected_continent.lower()]
    
    # Create visualizations with filtered data
    charts = create_visualizations(filtered_df)
    
    # If it's an AJAX request, return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(charts)
    
    # For initial page load, render full template
    return render_template('visualization.html', charts=charts)

@app.route('/api/visualization', methods=['POST'])
def get_visualization():
    data = request.json
    continent = data.get('continent', 'all')
    country = data.get('country', 'all')
    industry = data.get('industry', 'all')
    
    # Load and filter data
    df = load_data()
    
    # Apply filters
    filtered_df = df.copy()
    
    if continent != 'all':
        filtered_df = filtered_df[filtered_df['Continent'].str.lower() == continent.lower()]
    
    if country != 'all':
        filtered_df = filtered_df[filtered_df['Country'].str.lower() == country.lower()]
        
    if industry != 'all':
        filtered_df = filtered_df[filtered_df['Industry_Type'].str.lower() == industry.lower()]
    
    charts = create_visualizations(filtered_df)
    return jsonify(charts)

@app.route('/api/update_stats', methods=['POST'])
def update_stats():
    data = request.json
    continent = data.get('continent', 'all')
    country = data.get('country', 'all')
    industry = data.get('industry', 'all')
    
    # Load and filter data
    df = load_data()
    
    # Apply filters
    filtered_df = df.copy()
    
    if continent != 'all':
        filtered_df = filtered_df[filtered_df['Continent'].str.lower() == continent.lower()]
    
    if country != 'all':
        filtered_df = filtered_df[filtered_df['Country'].str.lower() == country.lower()]
        
    if industry != 'all':
        filtered_df = filtered_df[filtered_df['Industry_Type'].str.lower() == industry.lower()]
    
    # Calculate statistics from filtered data
    if len(filtered_df) > 0:
        total_emissions = filtered_df['Co2_Emissions_MetricTons'].sum() / 1000  # Convert to K
        
        # Calculate renewable energy percentage
        renewable_energy = filtered_df['Renewable_Energy_Percentage'].mean()
        
        # Calculate energy consumption
        energy_consumption = filtered_df['Energy_Consumption_TWh'].sum() / 1000  # Convert to M
        
        # Calculate urbanization percentage
        urbanization = filtered_df['Urbanization_Percentage'].mean()
    else:
        # Default values if no data matches filters
        total_emissions = 0
        renewable_energy = 0
        energy_consumption = 0
        urbanization = 0
    
    return jsonify({
        'totalEmissions': f"{total_emissions:.2f}K",
        'renewableEnergy': f"{renewable_energy:.2f}%",
        'energyConsumption': f"{energy_consumption:.2f}M",
        'urbanization': f"{urbanization:.2f}%"
    })

@app.route('/prediction')
def prediction():
    """Model prediction page."""
    if model is None:
        return render_template('error.html', message="Model not available")
    
    feature_names = model_info['feature_names']
    model_name = model_info['model_name']
    accuracy = model_info['accuracy']
    
    return render_template('prediction.html', 
                         feature_names=feature_names,
                         model_name=model_name,
                         accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None:
            return jsonify({'error': 'Model not available'})
        
        # Get input features from request
        features = []
        feature_names = model_info['feature_names']
        
        for feature in feature_names:
            value = float(request.json.get(feature, 0))
            features.append(value)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Convert to readable format
        continent = "Asia" if prediction == 0 else "Europe"
        confidence = max(probabilities) * 100
        
        return jsonify({
            'prediction': continent,
            'confidence': confidence,
            'probabilities': {
                'Asia': probabilities[0] * 100,
                'Europe': probabilities[1] * 100
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/templates/<path:filename>')
def block_template_access(filename):
    """Block direct access to template files for security."""
    return "Direct access to template files is not allowed.", 403

@app.errorhandler(404)
def not_found_error(error):
    """Custom 404 page with proper DOCTYPE."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Not Found - CO2 Emissions Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; display: flex; align-items: center; }
        .error-container { text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="error-container">
            <h1 class="display-1">404</h1>
            <h2>Page Not Found</h2>
            <p class="lead">The page you're looking for doesn't exist.</p>
            <a href="/" class="btn btn-light btn-lg">Go Home</a>
        </div>
    </div>
</body>
</html>''', 404

def create_visualizations(df):
    """Create various data visualizations matching the provided design."""
    charts = {}
    
    # Color schemes matching the image
    colors = {
        'asia': '#3498db',      # Blue
        'europe': '#9b59b6',    # Purple
        'Industrial': '#2471A3',  # Dark Blue
        'Domestic': '#8E44AD',    # Purple
        'Agriculture': '#27AE60', # Green
        'Automobile': '#E67E22'   # Orange
    }

    try:
        # 1. World Map with ALL countries from dataset
        country_emissions = df.groupby('Country')['Co2_Emissions_MetricTons'].sum().reset_index()
        
        # Create a world map with all countries
        world_map = go.Figure(data=go.Choropleth(
            locations=country_emissions['Country'],
            z=country_emissions['Co2_Emissions_MetricTons'],
            locationmode='country names',
            colorscale=[
                [0, '#3498db'],
                [0.5, '#f39c12'],
                [1, '#e74c3c']
            ],
            text=country_emissions['Country'] + ': ' + 
                 (country_emissions['Co2_Emissions_MetricTons']/1000).round(2).astype(str) + 'K tons',
            hovertemplate='<b>%{text}</b><br>CO2 Emissions: %{z:.2f} tons<extra></extra>',
            colorbar=dict(
                title="CO2 Emissions<br>(Metric Tons)",
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            )
        ))
        
        world_map.update_layout(
            title={
                'text': 'CO2 Emissions by Country',
                'font': {'color': 'white', 'size': 18},
                'x': 0.5
            },
            template='plotly_dark',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                showland=True,
                landcolor='rgba(255,255,255,0.1)',
                coastlinecolor='rgba(255,255,255,0.3)',
                projection_type='natural earth',
                showcountries=True,
                countrycolor='rgba(255,255,255,0.3)'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        charts['world_map'] = json.dumps(world_map, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. Industry Type Comparison
        industry_data = df.groupby(['Country', 'Industry_Type'])['Co2_Emissions_MetricTons'].sum().reset_index()
        industry_comparison = go.Figure()
        
        for country in industry_data['Country'].unique():
            country_data = industry_data[industry_data['Country'] == country]
            continent = df[df['Country'] == country]['Continent'].iloc[0].lower()
            country_color = colors.get(continent, '#3498db')
            
            industry_comparison.add_trace(go.Bar(
                name=country,
                x=country_data['Industry_Type'],
                y=country_data['Co2_Emissions_MetricTons'],
                marker_color=country_color,
                text=[f'{val/1000:.1f}K' for val in country_data['Co2_Emissions_MetricTons']],
                textposition='outside'
            ))
        
        industry_comparison.update_layout(
            title={
                'text': 'Compare which industries contribute most to<br>CO2 in Asia vs Europe.',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            barmode='group',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.02,
                y=0.02,
                bgcolor='rgba(0,0,0,0)'
            ),
            xaxis_title="Industry Type",
            yaxis_title="CO2 Emission (MetricTons)",
            margin=dict(l=40, r=40, t=80, b=60),
            height=350
        )
        charts['industry_comparison'] = json.dumps(industry_comparison, cls=plotly.utils.PlotlyJSONEncoder)

        # 3. Sector Emission by Year using ACTUAL data
        sector_emission = go.Figure()
        
        # Use actual sector columns
        sector_cols = {
            'Industrial': 'Industrial_Co2_Emissions_MetricTons',
            'Automobile': 'Automobile_Co2_Emissions_MetricTons',
            'Agriculture': 'Agriculture_Co2_Emissions_MetricTons',
            'Domestic': 'Domestic_Co2_Emissions_MetricTons'
        }
        
        if 'Year' in df.columns:
            # If Year column exists, group by year
            for sector_name, col_name in sector_cols.items():
                if col_name in df.columns:
                    yearly_data = df.groupby('Year')[col_name].sum().reset_index()
                    sector_emission.add_trace(go.Scatter(
                        x=yearly_data['Year'],
                        y=yearly_data[col_name],
                        mode='lines+markers',
                        name=sector_name,
                        line=dict(color=colors[sector_name]),
                        fill='tonexty' if sector_name != 'Industrial' else 'tozeroy'
                    ))
        else:
            # If no Year column, create sample time series
            years = list(range(2000, 2021))
            base_values = {}
            for sector_name, col_name in sector_cols.items():
                if col_name in df.columns:
                    base_values[sector_name] = df[col_name].sum()
                else:
                    base_values[sector_name] = 10000
            
            for i, (sector_name, base_val) in enumerate(base_values.items()):
                # Create realistic trends
                trend_values = [base_val * (0.8 + 0.4 * (j/len(years)) + np.sin(j/3) * 0.1) for j in range(len(years))]
                sector_emission.add_trace(go.Scatter(
                    x=years,
                    y=trend_values,
                    mode='lines',
                    name=sector_name,
                    line=dict(color=colors[sector_name]),
                    fill='tonexty' if i > 0 else 'tozeroy',
                    fillcolor=colors[sector_name].replace(')', ', 0.3)').replace('rgb', 'rgba')
                ))
        
        sector_emission.update_layout(
            title={
                'text': 'Sector Emission by Year',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            xaxis_title="Year",
            yaxis_title="Sum of Emission by Sector",
            showlegend=True,
            legend=dict(
                orientation="h",
                x=0.02,
                y=0.02,
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=40, r=40, t=80, b=60),
            height=350
        )
        charts['sector_emission'] = json.dumps(sector_emission, cls=plotly.utils.PlotlyJSONEncoder)

        # 4. Average Renewable Energy using ACTUAL data
        renewable_energy = go.Figure()
        if 'Renewable_Energy_Percentage' in df.columns:
            renewable_data = df.groupby('Country')['Renewable_Energy_Percentage'].mean().reset_index()
            
            renewable_energy.add_trace(go.Bar(
                x=renewable_data['Country'],
                y=renewable_data['Renewable_Energy_Percentage'],
                marker_color=[colors.get(country.lower(), '#3498db') for country in renewable_data['Country']],
                text=[f'{val:.1f}%' for val in renewable_data['Renewable_Energy_Percentage']],
                textposition='outside'
            ))
        
        renewable_energy.update_layout(
            title={
                'text': 'Average Renewable Energy Percentage by Country',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            showlegend=False,
            xaxis_title="Country",
            yaxis_title="Average Percentage (%)",
            margin=dict(l=40, r=40, t=80, b=60),
            height=350
        )
        charts['renewable_energy'] = json.dumps(renewable_energy, cls=plotly.utils.PlotlyJSONEncoder)

        # 5. CO2 Emissions Donut Chart using ACTUAL data
        emissions_donut = go.Figure()
        
        # Calculate actual sector totals
        sector_totals = {}
        for sector_name, col_name in sector_cols.items():
            if col_name in df.columns:
                sector_totals[sector_name] = df[col_name].sum()
        
        if sector_totals:
            labels = list(sector_totals.keys())
            values = list(sector_totals.values())
            total_emissions = sum(values)
            
            emissions_donut.add_trace(go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=[colors[sector] for sector in labels]),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=10)
            ))
            
            # Add center text
            emissions_donut.add_annotation(
                text=f"Total CO2<br>Emission<br>{total_emissions/1000:.1f}K",
                x=0.5, y=0.5,
                font_size=12,
                showarrow=False
            )
        
        emissions_donut.update_layout(
            title={
                'text': 'Sum of Co2 Emission by Sector',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="v",
                x=1.02,
                y=0.5,
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=40, r=120, t=80, b=60),
            height=350
        )
        charts['emissions_donut'] = json.dumps(emissions_donut, cls=plotly.utils.PlotlyJSONEncoder)

        # 6. CO2 Emissions Time Series using ACTUAL data
        emissions_time = go.Figure()
        
        if 'Year' in df.columns:
            yearly_total = df.groupby('Year')['Co2_Emissions_MetricTons'].sum().reset_index()
            emissions_time.add_trace(go.Scatter(
                x=yearly_total['Year'],
                y=yearly_total['Co2_Emissions_MetricTons'],
                mode='lines+markers',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6, color='#3498db'),
                text=[f'{val/1000:.1f}K' for val in yearly_total['Co2_Emissions_MetricTons']],
                textposition='top center'
            ))
        else:
            # Create sample time series based on total emissions
            years = list(range(2000, 2021))
            total_emissions = df['Co2_Emissions_MetricTons'].sum()
            base_value = total_emissions / len(years)
            
            # Create realistic trend
            time_values = [base_value * (0.9 + 0.2 * np.sin(i/3) + np.random.normal(0, 0.05)) 
                          for i in range(len(years))]
            
            emissions_time.add_trace(go.Scatter(
                x=years,
                y=time_values,
                mode='lines+markers',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6, color='#3498db'),
                text=[f'{val/1000:.1f}K' for val in time_values],
                textposition='top center'
            ))
        
        emissions_time.update_layout(
            title={
                'text': 'Sum of Co2 Emissions (MetricTons) by Year',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            xaxis_title="Year",
            yaxis_title="Sum of Co2 Emissions (MetricTons)",
            margin=dict(l=40, r=40, t=80, b=60),
            height=350
        )
        charts['emissions_time'] = json.dumps(emissions_time, cls=plotly.utils.PlotlyJSONEncoder)

        # 7. GDP vs CO2 Emissions using ACTUAL data
        gdp_emissions = go.Figure()
        
        if 'GDP_Billion_USD' in df.columns:
            gdp_emissions.add_trace(go.Scatter(
                x=df['Co2_Emissions_MetricTons'],
                y=df['GDP_Billion_USD'],
                mode='markers',
                marker=dict(
                    color=[colors.get(country.lower(), '#3498db') for country in df['Country']],
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=df['Country'],
                hovertemplate='<b>%{text}</b><br>CO2: %{x:.2f} tons<br>GDP: $%{y:.2f}B<extra></extra>'
            ))
        
        gdp_emissions.update_layout(
            title={
                'text': 'GDP vs CO2 Emissions by Country',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            xaxis_title="CO2 Emissions (Metric Tons)",
            yaxis_title="GDP (Billion USD)",
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=60),
            height=350
        )
        charts['gdp_emissions'] = json.dumps(gdp_emissions, cls=plotly.utils.PlotlyJSONEncoder)

        # 8. Industry Trend Analysis using ACTUAL data
        industry_trend = go.Figure()
        
        if 'Year' in df.columns:
            # Get industry data by year
            industries = df['Industry_Type'].unique()
            for industry in industries[:4]:  # Limit to 4 industries for readability
                industry_data = df[df['Industry_Type'] == industry]
                yearly_data = industry_data.groupby('Year')['Co2_Emissions_MetricTons'].mean().reset_index()
                
                if len(yearly_data) > 0:
                    industry_trend.add_trace(go.Scatter(
                        x=yearly_data['Year'],
                        y=yearly_data['Co2_Emissions_MetricTons'],
                        mode='lines+markers',
                        name=industry,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
        
        industry_trend.update_layout(
            title={
                'text': 'Industry CO2 Trends Over Time',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            xaxis_title="Year",
            yaxis_title="Average CO2 Emissions (Metric Tons)",
            legend=dict(
                orientation="h",
                x=0.02,
                y=0.02,
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=40, r=40, t=80, b=60),
            height=350
        )
        charts['industry_trend'] = json.dumps(industry_trend, cls=plotly.utils.PlotlyJSONEncoder)

        # 9. Population vs Emissions using ACTUAL data
        population_emissions = go.Figure()
        
        if 'Population_Millions' in df.columns:
            # Get country averages
            country_data = df.groupby('Country').agg({
                'Population_Millions': 'mean',
                'Co2_Emissions_MetricTons': 'mean',
                'Continent': 'first'
            }).reset_index()
            
            population_emissions.add_trace(go.Scatter(
                x=country_data['Population_Millions'],
                y=country_data['Co2_Emissions_MetricTons'],
                mode='markers+text',
                text=country_data['Country'].str.title(),
                textposition='top center',
                marker=dict(
                    size=[15 if x < 100 else 25 if x < 500 else 35 for x in country_data['Population_Millions']],
                    color=[colors.get(continent.lower(), '#3498db') for continent in country_data['Continent']],
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>Population: %{x:.1f}M<br>CO2: %{y:.2f} tons<extra></extra>'
            ))
        
        population_emissions.update_layout(
            title={
                'text': 'Population vs CO2 Emissions by Country',
                'font': {'size': 14, 'color': 'white'},
                'x': 0.02,
                'y': 0.95
            },
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10),
            xaxis_title="Population (Millions)",
            yaxis_title="Average CO2 Emissions (Metric Tons)",
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=60),
            height=350
        )
        charts['population_emissions'] = json.dumps(population_emissions, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return charts

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 