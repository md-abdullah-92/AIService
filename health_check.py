#!/usr/bin/env python3
"""
Health check script for Render deployment
"""
import requests
import sys
import time
import os

def health_check():
    """Check if the service is responding on the expected port"""
    port = int(os.getenv('PORT', 8000))
    url = f"http://0.0.0.0:{port}/"
    
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Health check passed! Service responding on port {port}")
                return True
        except Exception as e:
            print(f"⏳ Attempt {i+1}/{max_retries}: {e}")
            time.sleep(2)
    
    print(f"❌ Health check failed after {max_retries} attempts")
    return False

if __name__ == "__main__":
    success = health_check()
    sys.exit(0 if success else 1)
