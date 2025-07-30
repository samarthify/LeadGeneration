#!/usr/bin/env python3
"""
Simple startup test for the Flask app.
This script tests if the app can initialize without errors.
"""

import os
import sys
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_startup():
    """Test if the app can start without errors."""
    print("Testing app startup...")
    
    try:
        # Import the app
        from app import app, logger
        
        print("✅ App imported successfully")
        
        # Test basic configuration
        print(f"✅ Generated directory: {app.config.get('GENERATED_DIR', 'Not set')}")
        
        # Test if we can create a test client
        with app.test_client() as client:
            print("✅ Test client created successfully")
            
            # Test root endpoint
            response = client.get('/')
            print(f"✅ Root endpoint: {response.status_code}")
            
            # Test health endpoint
            response = client.get('/health')
            print(f"✅ Health endpoint: {response.status_code}")
            
            # Test API status endpoint
            response = client.get('/api/status')
            print(f"✅ API status endpoint: {response.status_code}")
        
        print("✅ All startup tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_startup()
    sys.exit(0 if success else 1) 