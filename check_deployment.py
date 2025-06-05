#!/usr/bin/env python3

import sys
import importlib

def check_imports():
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'pandas',
        'sklearn',
        'cv2',
        'PIL',
        'streamlit',
        'plotly',
        'folium',
        'streamlit_folium',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🚨 Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All packages available! Ready for deployment.")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1) 