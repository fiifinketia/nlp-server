#!/usr/bin/env python3
"""
Test script for the NLP Server
This script tests the basic functionality of the TTS server
"""

import requests
import json
import time
from pathlib import Path

# Server configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on localhost:8000")
        return False

def test_get_models():
    """Test getting available models"""
    print("\nTesting get models...")
    try:
        response = requests.get(f"{BASE_URL}/tts/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data['total_models']} total models, {data['loaded_models']} loaded")
            for model in data['models']:
                status = "✓" if model['is_loaded'] else "✗"
                print(f"  {status} {model['name']} ({model['type']}) - {model['language']}")
            return True
        else:
            print(f"✗ Get models failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Get models error: {e}")
        return False

def test_load_model(model_name):
    """Test loading a specific model"""
    print(f"\nTesting load model: {model_name}")
    try:
        response = requests.post(f"{BASE_URL}/tts/models/{model_name}/load")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ {data['message']}")
            return True
        else:
            print(f"✗ Load model failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Load model error: {e}")
        return False

def test_synthesize_speech(model_name, text="Hello, this is a test of the TTS system."):
    """Test speech synthesis"""
    print(f"\nTesting speech synthesis with model: {model_name}")
    try:
        data = {
            "text": text,
            "model_name": model_name,
            "length_scale": 1.0
        }
        
        # Add speaker for multi-speaker models
        if model_name == "ugtts_multi":
            data["speaker"] = "PT"
        
        response = requests.post(f"{BASE_URL}/tts/synthesize", json=data)
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print(f"✓ Speech synthesized successfully")
                print(f"  Audio path: {result['audio_path']}")
                if result['audio_info']:
                    print(f"  Duration: {result['audio_info']['duration']:.2f} seconds")
                    print(f"  Sample rate: {result['audio_info']['sample_rate']} Hz")
                return True
            else:
                print(f"✗ Synthesis failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Synthesis request failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Synthesis error: {e}")
        return False

def test_get_model_info(model_name):
    """Test getting model information"""
    print(f"\nTesting get model info: {model_name}")
    try:
        response = requests.get(f"{BASE_URL}/tts/models/{model_name}/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Model info retrieved:")
            print(f"  Name: {data['name']}")
            print(f"  Type: {data['type']}")
            print(f"  Language: {data['language']}")
            print(f"  Description: {data.get('description', 'N/A')}")
            if data.get('speakers'):
                print(f"  Speakers: {', '.join(data['speakers'])}")
            return True
        else:
            print(f"✗ Get model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Get model info error: {e}")
        return False

def main():
    """Main test function"""
    print("NLP Server Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("\nServer is not running or not accessible. Please start the server first:")
        print("python app.py")
        return
    
    # Test 2: Get models
    if not test_get_models():
        print("\nFailed to get models. Check server configuration.")
        return
    
    # Test 3: Load a model (try ugtts_multi first, then ugtts_single)
    model_to_test = None
    for model_name in ["ugtts_multi", "ugtts_single"]:
        if test_load_model(model_name):
            model_to_test = model_name
            break
    
    if not model_to_test:
        print("\nNo models could be loaded. Check model files and configuration.")
        return
    
    # Test 4: Get model info
    test_get_model_info(model_to_test)
    
    # Test 5: Synthesize speech
    test_synthesize_speech(model_to_test)
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo run the server:")
    print("python app.py")
    print("\nTo access the API documentation:")
    print("http://localhost:8000/docs")

if __name__ == "__main__":
    main() 