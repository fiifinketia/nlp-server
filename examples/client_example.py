#!/usr/bin/env python3
"""
Client example for the NLP Server
Demonstrates how to use the TTS API endpoints
"""

import requests
import json
import time
from pathlib import Path

class NLPServerClient:
    """Client for interacting with the NLP Server"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check server health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_models(self):
        """Get all available models"""
        try:
            response = self.session.get(f"{self.base_url}/tts/models")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get models: {e}")
            return None
    
    def load_model(self, model_name):
        """Load a specific model"""
        try:
            response = self.session.post(f"{self.base_url}/tts/models/{model_name}/load")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to load model {model_name}: {e}")
            return None
    
    def unload_model(self, model_name):
        """Unload a specific model"""
        try:
            response = self.session.delete(f"{self.base_url}/tts/models/{model_name}/unload")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to unload model {model_name}: {e}")
            return None
    
    def synthesize_speech(self, text, model_name, speaker=None, length_scale=1.0):
        """Synthesize speech from text"""
        try:
            data = {
                "text": text,
                "model_name": model_name,
                "length_scale": length_scale
            }
            if speaker:
                data["speaker"] = speaker
            
            response = self.session.post(f"{self.base_url}/tts/synthesize", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to synthesize speech: {e}")
            return None
    
    def get_audio_file(self, text, model_name, output_path, speaker=None, length_scale=1.0):
        """Get audio file directly"""
        try:
            data = {
                "text": text,
                "model_name": model_name,
                "length_scale": length_scale
            }
            if speaker:
                data["speaker"] = speaker
            
            response = self.session.post(f"{self.base_url}/tts/synthesize/file", data=data)
            response.raise_for_status()
            
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to get audio file: {e}")
            return None
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        try:
            response = self.session.get(f"{self.base_url}/tts/models/{model_name}/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model info: {e}")
            return None

def main():
    """Main example function"""
    print("NLP Server Client Example")
    print("=" * 50)
    
    # Initialize client
    client = NLPServerClient()
    
    # Check server health
    print("1. Checking server health...")
    health = client.health_check()
    if not health:
        print("Server is not running. Please start the server first:")
        print("python app.py")
        return
    
    print(f"✓ Server is healthy: {health}")
    
    # Get available models
    print("\n2. Getting available models...")
    models = client.get_models()
    if not models:
        return
    
    print(f"✓ Found {models['total_models']} models, {models['loaded_models']} loaded")
    
    # Find a model to use
    available_model = None
    for model in models['models']:
        if model['is_active']:
            available_model = model['name']
            break
    
    if not available_model:
        print("No active models found. Please check your model configuration.")
        return
    
    print(f"✓ Using model: {available_model}")
    
    # Load the model if not already loaded
    if not any(m['name'] == available_model and m['is_loaded'] for m in models['models']):
        print(f"\n3. Loading model: {available_model}")
        result = client.load_model(available_model)
        if not result:
            return
        print(f"✓ {result['message']}")
    
    # Get model information
    print(f"\n4. Getting model information...")
    model_info = client.get_model_info(available_model)
    if model_info:
        print(f"✓ Model: {model_info['name']}")
        print(f"  Type: {model_info['type']}")
        print(f"  Language: {model_info['language']}")
        print(f"  Description: {model_info.get('description', 'N/A')}")
        if model_info.get('speakers'):
            print(f"  Speakers: {', '.join(model_info['speakers'])}")
    
    # Synthesize speech
    print(f"\n5. Synthesizing speech...")
    test_texts = [
        "Hello, this is a test of the text-to-speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Welcome to the NLP server demonstration."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Test {i}: {text}")
        
        # Determine speaker for multi-speaker models
        speaker = None
        if model_info and model_info.get('speakers'):
            speaker = model_info['default_speaker']
        
        # Synthesize speech
        result = client.synthesize_speech(
            text=text,
            model_name=available_model,
            speaker=speaker,
            length_scale=1.0
        )
        
        if result and result.get('success'):
            print(f"   ✓ Success! Audio: {result['audio_path']}")
            if result.get('audio_info'):
                duration = result['audio_info']['duration']
                print(f"   ✓ Duration: {duration:.2f} seconds")
        else:
            print(f"   ✗ Failed: {result.get('error', 'Unknown error') if result else 'No response'}")
    
    # Get audio file directly
    print(f"\n6. Getting audio file directly...")
    output_path = "example_output.wav"
    result = client.get_audio_file(
        text="This audio was generated directly as a file.",
        model_name=available_model,
        output_path=output_path,
        speaker=speaker,
        length_scale=1.0
    )
    
    if result:
        print(f"✓ Audio file saved: {output_path}")
        file_size = Path(output_path).stat().st_size
        print(f"✓ File size: {file_size} bytes")
    else:
        print("✗ Failed to get audio file")
    
    print("\n" + "=" * 50)
    print("Client example completed!")
    print(f"\nGenerated files:")
    print(f"- {output_path}")

if __name__ == "__main__":
    main() 