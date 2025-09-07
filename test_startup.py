#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI app can start properly
"""
import os
import sys
sys.path.insert(0, '.')

def test_app_startup():
    """Test that the FastAPI app can be imported and created without errors"""
    try:
        from main import app
        print("✅ FastAPI app imported successfully")
        
        # Test that the app has the expected routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/upload/", "/quiz/", "/generate-study-notes/", "/short-questions/"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"❌ Route {route} missing")
                return False
        
        print("✅ All expected routes found")
        return True
        
    except Exception as e:
        print(f"❌ Error importing app: {e}")
        return False

if __name__ == "__main__":
    success = test_app_startup()
    if success:
        print("🎉 App startup test passed!")
        sys.exit(0)
    else:
        print("💥 App startup test failed!")
        sys.exit(1)
