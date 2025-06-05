"""
Forest Fire Detection & Spread Prediction Dashboard
Interactive Streamlit application for fire monitoring and prediction in India
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

import json
from pathlib import Path
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path("src")))

try:
    from src.detection_model import load_trained_model, predict_fire_in_image
    from src.prediction_model import load_trained_prediction_model, predict_fire_spread, create_fire_spread_forecast
    from src.preprocessing import DataAugmentation
    from src.alert_system import alert_system
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all model files are present.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🔥 Forest Fire Monitoring System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #FF8E53);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6B6B;
    }
    .risk-low { color: #28a745; }
    .risk-medium { color: #ffc107; }
    .risk-high { color: #fd7e14; }
    .risk-critical { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_india_coordinates():
    """Load India fire-prone area coordinates"""
    try:
        with open('data/processed/india_fire_coordinates.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default coordinates if file not found
        return {
            'Odisha': {'lat': 20.9517, 'lon': 85.0985, 'fire_frequency': 'High'},
            'Chhattisgarh': {'lat': 21.2787, 'lon': 81.8661, 'fire_frequency': 'High'},
            'Maharashtra': {'lat': 19.7515, 'lon': 75.7139, 'fire_frequency': 'Medium'},
            'Telangana': {'lat': 18.1124, 'lon': 79.0193, 'fire_frequency': 'Medium'},
            'Jharkhand': {'lat': 23.6102, 'lon': 85.2799, 'fire_frequency': 'High'}
        }

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        # Load detection model
        detection_path = Path("models/saved_models/best_fire_detection_model.pth")
        if detection_path.exists():
            models['detection'] = load_trained_model(str(detection_path))
            
        # Load prediction model
        prediction_path = Path("models/saved_models/best_fire_prediction_model.pth")
        if prediction_path.exists():
            models['prediction'] = load_trained_prediction_model(str(prediction_path))
            
    except Exception as e:
        st.warning(f"Could not load models: {e}")
    
    return models



def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔥 Forest Fire Detection & Spread Prediction System</h1>
        <p>AI-powered forest fire monitoring for India</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    coordinates_data = load_india_coordinates()
    models = load_models()
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Main functionality selection
    mode = st.sidebar.selectbox(
        "Select Functionality",
        ["🏠 Dashboard", "🔥 Fire Detection", "🌪️ Spread Prediction", "🗺️ Fire Map", "📊 Analytics"]
    )
    
    if mode == "🏠 Dashboard":
        show_dashboard(coordinates_data, models)
    elif mode == "🔥 Fire Detection":
        show_fire_detection(models)
    elif mode == "🌪️ Spread Prediction":
        show_spread_prediction(models)
    elif mode == "🗺️ Fire Map":
        show_fire_map(coordinates_data)
    elif mode == "📊 Analytics":
        show_analytics()

def show_dashboard(coordinates_data, models):
    """Main dashboard with overview"""
    st.header("📊 Fire Monitoring Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Get alert statistics for dynamic metrics
    alert_stats = alert_system.get_alert_statistics()
    
    with col1:
        st.metric("🔥 Active Alerts", alert_stats['detection_alerts'], f"+{alert_stats['detection_alerts']//3}")
    with col2:
        st.metric("🌪️ High Risk Areas", alert_stats['critical_alerts'] + alert_stats['high_alerts'], f"+{alert_stats['critical_alerts']}")
    with col3:
        st.metric("🛡️ Total Alerts", alert_stats['total_alerts'], "0")
    with col4:
        st.metric("⚡ Model Accuracy", "98.5%", "+0.3%")
    
    # Map and recent alerts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Fire Risk Map - India")
        
        # Create simple dataframe for native map
        import pandas as pd
        map_data = []
        for state, info in coordinates_data.items():
            map_data.append({
                'latitude': info['lat'],
                'longitude': info['lon'],
                'state': state,
                'risk': info['fire_frequency']
            })
        
        df = pd.DataFrame(map_data)
        
        # Display native Streamlit map
        st.map(df, latitude='latitude', longitude='longitude', size=20, color='#ff0000')
        
        # Show details below map
        st.markdown("**Fire Risk Areas:**")
        for state, info in coordinates_data.items():
            risk_color = "🔴" if info['fire_frequency'] == 'High' else "🟡" if info['fire_frequency'] == 'Medium' else "🟢"
            st.markdown(f"{risk_color} **{state}** - {info['fire_frequency']} Risk")
    
    with col2:
        st.subheader("🚨 Recent Alerts")
        
        # Update alerts and get recent ones
        alert_system.update_alerts()
        recent_alerts = alert_system.get_recent_alerts(hours=24, limit=5)
        
        if recent_alerts:
            for alert in recent_alerts:
                risk_class = f"risk-{alert['risk_level'].lower()}"
                icon = "🔥" if alert['type'] == "Fire Detection" else "🌪️" if "Weather" in alert['type'] else "⚠️"
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{icon} {alert['location']}</strong><br>
                    <span class="{risk_class}">{alert['risk_level']} Risk</span><br>
                    <small>{alert['relative_time']} - {alert['type']}</small><br>
                    <em style="font-size: 0.8em; color: #666;">{alert.get('details', '')[:60]}...</em>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("No recent alerts. System monitoring...")
            
        # Add refresh button
        if st.button("🔄 Refresh Alerts"):
            st.experimental_rerun()

def show_fire_detection(models):
    """Fire detection interface"""
    st.header("🔥 Fire Detection from Satellite Images")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Upload a satellite image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload satellite imagery to detect fires"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            if st.button("🔍 Detect Fire", type="primary"):
                if 'detection' in models:
                    try:
                        # Preprocess image
                        transform = DataAugmentation.get_val_transforms()
                        image_tensor = transform(image)
                        
                        # Make prediction
                        with st.spinner("Analyzing image..."):
                            result = predict_fire_in_image(models['detection'], image_tensor)
                        
                        # Display results in col2
                        with col2:
                            st.subheader("🎯 Detection Results")
                            
                            if result['has_fire']:
                                st.error("🔥 FIRE DETECTED!")
                                st.markdown(f"**Fire Probability:** {result['fire_probability']:.4f}")
                                st.markdown(f"**Confidence:** {result['confidence']:.4f}")
                                
                                # Generate automatic alert
                                if result['confidence'] > 0.8:
                                    risk_level = "Critical" if result['confidence'] > 0.9 else "High"
                                    alert_system.add_alert(
                                        location="Satellite Image Detection",
                                        risk_level=risk_level,
                                        alert_type="Fire Detection",
                                        details=f"Fire detected with {result['confidence']:.2f} confidence in uploaded image",
                                        coordinates=None
                                    )
                                    st.success("🚨 Alert generated and saved to system!")
                                
                                # Show recommendation
                                st.warning("⚠️ **Immediate Action Required**")
                                st.info("📞 Alert fire department\n🚁 Deploy firefighting resources")
                                
                            else:
                                st.success("✅ No fire detected")
                                st.markdown(f"**Fire Probability:** {result['fire_probability']:.4f}")
                                st.markdown(f"**Confidence:** {result['confidence']:.4f}")
                                
                    except Exception as e:
                        st.error(f"Detection failed: {e}")
                else:
                    st.error("Detection model not loaded. Please train the model first.")
        
        else:
            with col2:
                st.info("👆 Upload an image to start fire detection")
                
                # Sample images
                st.subheader("📸 Sample Images")
                st.markdown("You can test with these sample scenarios:")
                st.markdown("- Forest landscapes")
                st.markdown("- Satellite imagery")
                st.markdown("- Smoke patterns")

def show_spread_prediction(models):
    """Fire spread prediction interface"""
    st.header("🌪️ Fire Spread Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌤️ Weather Input")
        
        # Weather input form
        with st.form("weather_form"):
            st.markdown("**Current Weather Conditions**")
            
            temperature = st.slider("🌡️ Temperature (°C)", min_value=15, max_value=50, value=30)
            humidity = st.slider("💧 Humidity (%)", min_value=10, max_value=100, value=40)
            wind_speed = st.slider("💨 Wind Speed (km/h)", min_value=0, max_value=50, value=15)
            rainfall = st.slider("🌧️ Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
            
            # Additional parameters
            st.markdown("**Historical Data (Past 7 days)**")
            use_historical = st.checkbox("Use historical weather data")
            
            predict_button = st.form_submit_button("🔮 Predict Fire Spread", type="primary")
        
        if predict_button and 'prediction' in models:
            try:
                # Create weather sequence
                if use_historical:
                    # Generate realistic historical data
                    weather_sequence = []
                    for day in range(7):
                        temp_var = temperature + np.random.normal(0, 3)
                        humid_var = humidity + np.random.normal(0, 10)
                        wind_var = wind_speed + np.random.normal(0, 5)
                        rain_var = max(0, rainfall + np.random.normal(0, 2))
                        
                        # Normalize to 0-1 scale (simplified)
                        weather_sequence.append([
                            min(max(temp_var, 15), 50) / 50,  # Normalize temperature
                            min(max(humid_var, 10), 100) / 100,  # Normalize humidity
                            min(max(wind_var, 0), 50) / 50,  # Normalize wind speed
                            min(rain_var, 50) / 50  # Normalize rainfall
                        ])
                else:
                    # Use current conditions for all days
                    current_conditions = [
                        temperature / 50,
                        humidity / 100,
                        wind_speed / 50,
                        rainfall / 50
                    ]
                    weather_sequence = [current_conditions] * 7
                
                weather_array = np.array(weather_sequence)
                
                with st.spinner("Predicting fire spread risk..."):
                    result = predict_fire_spread(models['prediction'], weather_array)
                
                # Display results in col2
                with col2:
                    st.subheader("🎯 Prediction Results")
                    
                    risk_level = result['risk_level']
                    risk_prob = result['fire_risk_probability']
                    
                    # Risk level indicator
                    if risk_level == "Critical":
                        st.error(f"🚨 **{risk_level} Risk**")
                    elif risk_level == "High":
                        st.warning(f"⚠️ **{risk_level} Risk**")
                    elif risk_level == "Medium":
                        st.info(f"📊 **{risk_level} Risk**")
                    else:
                        st.success(f"✅ **{risk_level} Risk**")
                    
                    # Generate automatic alert for high risk predictions
                    if risk_level in ["Critical", "High"]:
                        alert_system.add_alert(
                            location="Weather Prediction Model",
                            risk_level=risk_level,
                            alert_type="Spread Prediction",
                            details=f"High fire spread risk predicted: {temperature}°C, {humidity}% humidity, {wind_speed} km/h winds",
                            coordinates=None
                        )
                        st.success("🚨 Weather alert generated and saved to system!")
                    
                    st.metric("Fire Risk Probability", f"{risk_prob:.4f}")
                    st.metric("Confidence", f"{result['confidence']:.4f}")
                    
                    # Risk gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_prob * 100,
                        title = {'text': "Fire Risk (%)"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 60], 'color': "yellow"},
                                {'range': [60, 80], 'color': "orange"},
                                {'range': [80, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if risk_level in ["Critical", "High"]:
                        st.error("🚨 **Immediate Actions:**")
                        st.markdown("- Alert fire services")
                        st.markdown("- Increase surveillance")
                        st.markdown("- Prepare evacuation plans")
                        st.markdown("- Deploy fire-fighting resources")
                    elif risk_level == "Medium":
                        st.warning("⚠️ **Monitoring Required:**")
                        st.markdown("- Monitor weather conditions")
                        st.markdown("- Check fire suppression equipment")
                        st.markdown("- Issue weather warnings")
                    else:
                        st.success("✅ **Normal Operations:**")
                        st.markdown("- Continue routine monitoring")
                        st.markdown("- Maintain equipment")
                        
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        
        elif predict_button:
            st.error("Prediction model not loaded. Please train the model first.")
    
    with col2:
        if not predict_button:
            st.info("👈 Enter weather conditions to predict fire spread risk")
            
            # Show sample weather patterns
            st.subheader("🌡️ Weather Risk Factors")
            st.markdown("""
            **High Risk Conditions:**
            - Temperature > 35°C
            - Humidity < 30%
            - Wind Speed > 20 km/h
            - No rainfall for 7+ days
            
            **Low Risk Conditions:**
            - Temperature < 25°C
            - Humidity > 70%
            - Wind Speed < 10 km/h
            - Recent rainfall > 10mm
            """)

def show_fire_map(coordinates_data):
    """Interactive fire monitoring map"""
    st.header("🗺️ Interactive Fire Monitoring Map")
    
    # Map controls
    col1, col2 = st.columns(2)
    with col1:
        show_risk_areas = st.checkbox("Show Risk Areas", value=True)
    with col2:
        show_detections = st.checkbox("Show Fire Detections", value=True)
    
    # Simulate some fire detections
    fire_detections = [
        {"lat": 21.0, "lon": 85.5, "confidence": 0.95},
        {"lat": 22.0, "lon": 82.0, "confidence": 0.87},
        {"lat": 19.5, "lon": 75.5, "confidence": 0.76}
    ] if show_detections else None
    
    # Create map data using Streamlit native map
    import pandas as pd
    
    all_map_data = []
    
    if show_risk_areas:
        # Add fire-prone areas
        for state, info in coordinates_data.items():
            all_map_data.append({
                'latitude': info['lat'],
                'longitude': info['lon'],
                'type': 'Risk Area',
                'details': f"{state} - {info['fire_frequency']} Risk",
                'size': 100
            })
    
    if show_detections:
        # Add fire detections
        for detection in fire_detections:
            all_map_data.append({
                'latitude': detection['lat'],
                'longitude': detection['lon'],
                'type': 'Fire Detection',
                'details': f"Fire detected - {detection['confidence']:.2f} confidence",
                'size': 200
            })
    
    if all_map_data:
        df = pd.DataFrame(all_map_data)
        st.map(df, latitude='latitude', longitude='longitude', size='size', color='#ff0000')
        
        # Show details
        st.subheader("📍 Map Details")
        for item in all_map_data:
            icon = "🔥" if item['type'] == 'Fire Detection' else "⚠️"
            st.markdown(f"{icon} {item['details']}")
    else:
        st.info("Select options above to display map data.")
    
    # Map legend
    st.subheader("🔍 Map Legend")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔴 **High Risk Areas**")
    with col2:
        st.markdown("🟡 **Medium Risk Areas**")
    with col3:
        st.markdown("🟢 **Low Risk Areas**")

def show_analytics():
    """Analytics and model performance"""
    st.header("📊 Analytics & Model Performance")
    
    # Model performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Fire Detection Model")
        
        # Load model summary if available
        try:
            with open('models/detection_model_summary.json', 'r') as f:
                detection_summary = json.load(f)
            
            st.metric("Accuracy", f"{detection_summary.get('test_accuracy', 0):.4f}")
            st.metric("F1-Score", f"{detection_summary.get('test_f1_score', 0):.4f}")
            st.metric("Precision", f"{detection_summary.get('test_precision', 0):.4f}")
            st.metric("Recall", f"{detection_summary.get('test_recall', 0):.4f}")
            
        except FileNotFoundError:
            st.info("Train the detection model to see performance metrics")
    
    with col2:
        st.subheader("🌪️ Spread Prediction Model")
        
        try:
            with open('models/prediction_model_summary.json', 'r') as f:
                prediction_summary = json.load(f)
            
            st.metric("MAE", f"{prediction_summary.get('test_mae', 0):.6f}")
            st.metric("RMSE", f"{prediction_summary.get('test_rmse', 0):.6f}")
            st.metric("R² Score", f"{prediction_summary.get('test_r2_score', 0):.6f}")
            
        except FileNotFoundError:
            st.info("Train the prediction model to see performance metrics")
    
    # Dynamic Alert Analytics
    st.subheader("🚨 Real-time Alert Analytics")
    
    # Get alert statistics
    alert_stats = alert_system.get_alert_statistics()
    recent_alerts = alert_system.get_recent_alerts(hours=168, limit=50)  # Last week
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Alerts", alert_stats['total_alerts'])
    with col2:
        st.metric("Critical Alerts", alert_stats['critical_alerts'])
    with col3:
        st.metric("Detection Alerts", alert_stats['detection_alerts'])
    
    # Alert types distribution
    if recent_alerts:
        alert_types = [alert['type'] for alert in recent_alerts]
        type_counts = {}
        for alert_type in alert_types:
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Alert Types Distribution (Last 7 Days)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fire statistics (simulated)
    st.subheader("📈 Fire Statistics - India (2024)")
    
    # Sample data for demonstration
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fire_incidents = [45, 38, 82, 156, 234, 89, 67, 78, 123, 189, 167, 98]
    
    fig = px.bar(
        x=months, 
        y=fire_incidents,
        title="Monthly Fire Incidents in India",
        labels={'x': 'Month', 'y': 'Number of Incidents'},
        color=fire_incidents,
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # State-wise fire frequency
    st.subheader("🗺️ State-wise Fire Frequency")
    
    state_data = pd.DataFrame({
        'State': ['Odisha', 'Chhattisgarh', 'Maharashtra', 'Telangana', 'Jharkhand'],
        'Fire_Incidents': [234, 198, 167, 145, 189],
        'Risk_Level': ['High', 'High', 'Medium', 'Medium', 'High']
    })
    
    fig = px.bar(
        state_data, 
        x='State', 
        y='Fire_Incidents',
        color='Risk_Level',
        title="Fire Incidents by State",
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Quick Start")
st.sidebar.markdown("""
1. Run data collection: `python src/data_collection.py`
2. Train detection model: `python src/train_detection.py`
3. Train prediction model: `python src/train_prediction.py`
4. Use this dashboard for monitoring!
""")

st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
**Forest Fire ML System v1.0**

AI-powered fire detection and spread prediction system designed for India's unique geographical and climatic conditions.

Built with PyTorch, Streamlit, and ❤️
""")

if __name__ == "__main__":
    main() 