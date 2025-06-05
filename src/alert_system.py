import json
import os
from datetime import datetime, timedelta
import random
import numpy as np
import requests
from typing import List, Dict, Optional

class FireAlertSystem:
    def __init__(self, alerts_file="data/alerts.json"):
        self.alerts_file = alerts_file
        self.ensure_alerts_file()
        
    def ensure_alerts_file(self):
        """Ensure alerts file exists"""
        os.makedirs(os.path.dirname(self.alerts_file), exist_ok=True)
        if not os.path.exists(self.alerts_file):
            self.save_alerts([])
    
    def load_alerts(self) -> List[Dict]:
        """Load alerts from file"""
        try:
            with open(self.alerts_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_alerts(self, alerts: List[Dict]):
        """Save alerts to file"""
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
    
    def add_alert(self, location: str, risk_level: str, alert_type: str, 
                  details: str = "", coordinates: Optional[tuple] = None):
        """Add a new alert"""
        alerts = self.load_alerts()
        
        new_alert = {
            "id": len(alerts) + 1,
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "risk_level": risk_level,
            "type": alert_type,
            "details": details,
            "coordinates": coordinates,
            "status": "active"
        }
        
        alerts.insert(0, new_alert)  # Add to beginning
        
        # Keep only last 50 alerts
        alerts = alerts[:50]
        
        self.save_alerts(alerts)
        return new_alert
    
    def get_recent_alerts(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Get recent alerts within specified hours"""
        alerts = self.load_alerts()
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in alerts:
            try:
                alert_time = datetime.fromisoformat(alert["timestamp"])
                if alert_time >= cutoff_time:
                    # Calculate relative time
                    time_diff = datetime.now() - alert_time
                    alert["relative_time"] = self.format_relative_time(time_diff)
                    recent_alerts.append(alert)
            except (ValueError, KeyError):
                continue
        
        return recent_alerts[:limit]
    
    def format_relative_time(self, time_diff: timedelta) -> str:
        """Format time difference into human readable format"""
        total_seconds = int(time_diff.total_seconds())
        
        if total_seconds < 60:
            return "Just now"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = total_seconds // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    def generate_weather_based_alerts(self):
        """Generate alerts based on current weather conditions"""
        # Simulate weather data (in real app, this would come from weather API)
        indian_states = [
            {"name": "Odisha", "coords": (20.9517, 85.0985)},
            {"name": "Chhattisgarh", "coords": (21.2787, 81.8661)},
            {"name": "Maharashtra", "coords": (19.7515, 75.7139)},
            {"name": "Karnataka", "coords": (15.3173, 75.7139)},
            {"name": "Telangana", "coords": (18.1124, 79.0193)},
            {"name": "Jharkhand", "coords": (23.6102, 85.2799)},
            {"name": "West Bengal", "coords": (22.9868, 87.8550)},
            {"name": "Madhya Pradesh", "coords": (22.9734, 78.6569)},
        ]
        
        for state in indian_states:
            # Simulate weather conditions
            temperature = random.uniform(25, 45)
            humidity = random.uniform(20, 80)
            wind_speed = random.uniform(5, 35)
            rainfall = random.uniform(0, 20)
            
            # Calculate fire risk based on weather
            fire_risk = self.calculate_weather_fire_risk(temperature, humidity, wind_speed, rainfall)
            
            # Generate alerts based on risk
            if fire_risk > 0.8:
                self.add_alert(
                    location=state["name"],
                    risk_level="Critical",
                    alert_type="Weather Warning",
                    details=f"Extreme fire weather: {temperature:.1f}°C, {humidity:.1f}% humidity, {wind_speed:.1f} km/h winds",
                    coordinates=state["coords"]
                )
            elif fire_risk > 0.6:
                self.add_alert(
                    location=state["name"],
                    risk_level="High",
                    alert_type="Weather Alert",
                    details=f"High fire risk weather: {temperature:.1f}°C, {humidity:.1f}% humidity",
                    coordinates=state["coords"]
                )
    
    def calculate_weather_fire_risk(self, temp: float, humidity: float, 
                                   wind_speed: float, rainfall: float) -> float:
        """Calculate fire risk based on weather parameters"""
        # Higher temperature increases risk
        temp_risk = min(temp / 50.0, 1.0)
        
        # Lower humidity increases risk
        humidity_risk = 1.0 - (humidity / 100.0)
        
        # Higher wind speed increases risk
        wind_risk = min(wind_speed / 40.0, 1.0)
        
        # Recent rainfall decreases risk
        rain_risk = max(0, 1.0 - (rainfall / 20.0))
        
        # Combined risk score
        fire_risk = (temp_risk * 0.3 + humidity_risk * 0.4 + 
                    wind_risk * 0.2 + rain_risk * 0.1)
        
        return min(fire_risk, 1.0)
    
    def simulate_detection_alerts(self):
        """Simulate fire detection alerts"""
        detection_locations = [
            {"name": "Simlipal National Park, Odisha", "coords": (21.6, 86.5)},
            {"name": "Bandhavgarh National Park, MP", "coords": (23.7, 81.0)},
            {"name": "Tadoba National Park, Maharashtra", "coords": (20.2, 79.3)},
            {"name": "Nagarhole National Park, Karnataka", "coords": (12.0, 76.1)},
        ]
        
        # Randomly generate detection alerts
        if random.random() < 0.3:  # 30% chance
            location = random.choice(detection_locations)
            confidence = random.uniform(0.75, 0.98)
            
            if confidence > 0.9:
                risk_level = "Critical"
                details = f"Fire detected with {confidence:.2f} confidence. Immediate response required."
            elif confidence > 0.8:
                risk_level = "High"
                details = f"Likely fire detected with {confidence:.2f} confidence."
            else:
                risk_level = "Medium"
                details = f"Possible fire detected with {confidence:.2f} confidence. Verification needed."
            
            self.add_alert(
                location=location["name"],
                risk_level=risk_level,
                alert_type="Fire Detection",
                details=details,
                coordinates=location["coords"]
            )
    
    def get_nasa_firms_data(self) -> List[Dict]:
        """Get real fire data from NASA FIRMS (if available)"""
        try:
            # NASA FIRMS API for India region (simplified)
            # In production, you'd need an API key
            url = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/YOUR_API_KEY/VIIRS_SNPP_NRT/IND/1"
            
            # For demo, return simulated data
            return []
            
        except Exception as e:
            print(f"Could not fetch NASA FIRMS data: {e}")
            return []
    
    def update_alerts(self):
        """Update alerts with new data"""
        # Clean old alerts (older than 7 days)
        alerts = self.load_alerts()
        cutoff_time = datetime.now() - timedelta(days=7)
        
        active_alerts = []
        for alert in alerts:
            try:
                alert_time = datetime.fromisoformat(alert["timestamp"])
                if alert_time >= cutoff_time:
                    active_alerts.append(alert)
            except (ValueError, KeyError):
                continue
        
        self.save_alerts(active_alerts)
        
        # Generate new alerts based on current conditions
        if random.random() < 0.6:  # 60% chance to generate weather alerts
            self.generate_weather_based_alerts()
        
        if random.random() < 0.4:  # 40% chance to generate detection alerts
            self.simulate_detection_alerts()
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        alerts = self.load_alerts()
        
        stats = {
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.get("risk_level") == "Critical"]),
            "high_alerts": len([a for a in alerts if a.get("risk_level") == "High"]),
            "detection_alerts": len([a for a in alerts if a.get("type") == "Fire Detection"]),
            "weather_alerts": len([a for a in alerts if "Weather" in a.get("type", "")]),
        }
        
        return stats

# Initialize global alert system
alert_system = FireAlertSystem() 