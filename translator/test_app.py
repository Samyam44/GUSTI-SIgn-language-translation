"""
Test Script for Dual-Engine ASL Recognition Application
Tests all API endpoints and functionality.
"""

import requests
import json
import time
import sys


class ASLAppTester:
    """Test suite for ASL recognition application."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize tester.
        
        Args:
            base_url: Base URL of the Flask application
        """
        self.base_url = base_url
        self.test_results = []
    
    def log(self, message: str, success: bool = True):
        """Log test result."""
        status = "✅" if success else "❌"
        print(f"{status} {message}")
        self.test_results.append((message, success))
    
    def test_app_running(self):
        """Test if application is running."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                self.log("Application is running")
                return True
            else:
                self.log(f"Application returned status {response.status_code}", False)
                return False
        except requests.exceptions.RequestException as e:
            self.log(f"Cannot connect to application: {e}", False)
            return False
    
    def test_get_status(self):
        """Test status endpoint."""
        try:
            response = requests.get(f"{self.base_url}/get_status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log(f"Status endpoint working - Current engine: {data.get('current_engine')}")
                
                # Check all engines available
                if data.get('sentence_engine_available'):
                    self.log("  ↳ Sentence engine available")
                else:
                    self.log("  ↳ Sentence engine NOT available", False)
                
                if data.get('alphabet_engine_available'):
                    self.log("  ↳ Alphabet engine available")
                else:
                    self.log("  ↳ Alphabet engine NOT available", False)
                
                if data.get('webcam_available'):
                    self.log("  ↳ Webcam available")
                else:
                    self.log("  ↳ Webcam NOT available", False)
                
                return True
            else:
                self.log(f"Status endpoint failed with status {response.status_code}", False)
                return False
        except Exception as e:
            self.log(f"Status test error: {e}", False)
            return False
    
    def test_switch_to_sentence(self):
        """Test switching to sentence engine."""
        try:
            response = requests.post(
                f"{self.base_url}/set_engine",
                json={"engine": "sentence"},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('engine') == 'sentence':
                    self.log("Switched to sentence engine successfully")
                    return True
                else:
                    self.log("Switch to sentence engine failed", False)
                    return False
            else:
                self.log(f"Switch engine failed with status {response.status_code}", False)
                return False
        except Exception as e:
            self.log(f"Switch engine test error: {e}", False)
            return False
    
    def test_switch_to_alphabet(self):
        """Test switching to alphabet engine."""
        try:
            response = requests.post(
                f"{self.base_url}/set_engine",
                json={"engine": "alphabet"},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('engine') == 'alphabet':
                    self.log("Switched to alphabet engine successfully")
                    return True
                else:
                    self.log("Switch to alphabet engine failed", False)
                    return False
            else:
                self.log(f"Switch engine failed with status {response.status_code}", False)
                return False
        except Exception as e:
            self.log(f"Switch engine test error: {e}", False)
            return False
    
    def test_invalid_engine(self):
        """Test switching to invalid engine."""
        try:
            response = requests.post(
                f"{self.base_url}/set_engine",
                json={"engine": "invalid"},
                timeout=5
            )
            if response.status_code == 400:
                self.log("Invalid engine properly rejected (400)")
                return True
            else:
                self.log(f"Invalid engine not properly rejected (got {response.status_code})", False)
                return False
        except Exception as e:
            self.log(f"Invalid engine test error: {e}", False)
            return False
    
    def test_toggle_detection(self):
        """Test detection toggle."""
        try:
            # Toggle once
            response1 = requests.post(f"{self.base_url}/toggle_detection", timeout=5)
            if response1.status_code != 200:
                self.log("First detection toggle failed", False)
                return False
            
            state1 = response1.json().get('detection_enabled')
            
            # Toggle again
            response2 = requests.post(f"{self.base_url}/toggle_detection", timeout=5)
            if response2.status_code != 200:
                self.log("Second detection toggle failed", False)
                return False
            
            state2 = response2.json().get('detection_enabled')
            
            if state1 != state2:
                self.log(f"Detection toggle working (states: {state1} → {state2})")
                return True
            else:
                self.log("Detection toggle not changing state", False)
                return False
        except Exception as e:
            self.log(f"Toggle detection test error: {e}", False)
            return False
    
    def test_set_language(self):
        """Test language setting."""
        languages = ['es', 'fr', 'de', 'it']
        
        for lang in languages:
            try:
                response = requests.post(
                    f"{self.base_url}/set_language",
                    json={"language": lang},
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success') and data.get('language') == lang:
                        self.log(f"  ↳ Set language to {lang}")
                    else:
                        self.log(f"  ↳ Failed to set language to {lang}", False)
                        return False
                else:
                    self.log(f"Language setting failed for {lang}", False)
                    return False
            except Exception as e:
                self.log(f"Language test error for {lang}: {e}", False)
                return False
        
        self.log("Language setting working for all tested languages")
        return True
    
    def test_clear_sentence(self):
        """Test sentence clearing."""
        try:
            response = requests.post(f"{self.base_url}/clear_sentence", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.log("Sentence clearing successful")
                    return True
                else:
                    self.log("Sentence clearing failed", False)
                    return False
            else:
                self.log(f"Clear sentence failed with status {response.status_code}", False)
                return False
        except Exception as e:
            self.log(f"Clear sentence test error: {e}", False)
            return False
    
    def test_get_translation(self):
        """Test translation endpoint."""
        try:
            response = requests.get(f"{self.base_url}/get_translation", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log(f"Translation endpoint working - Engine: {data.get('engine')}")
                return True
            else:
                self.log(f"Translation endpoint failed with status {response.status_code}", False)
                return False
        except Exception as e:
            self.log(f"Translation test error: {e}", False)
            return False
    
    def test_engine_switching_sequence(self):
        """Test rapid engine switching."""
        try:
            engines = ['sentence', 'alphabet', 'sentence', 'alphabet']
            
            for engine in engines:
                response = requests.post(
                    f"{self.base_url}/set_engine",
                    json={"engine": engine},
                    timeout=5
                )
                if response.status_code != 200:
                    self.log(f"Rapid switch to {engine} failed", False)
                    return False
                time.sleep(0.1)  # Small delay
            
            self.log("Rapid engine switching sequence successful")
            return True
        except Exception as e:
            self.log(f"Engine switching sequence error: {e}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*60)
        print("ASL Recognition Application Test Suite")
        print("="*60 + "\n")
        
        # Check if app is running
        if not self.test_app_running():
            print("\n⚠️  Application is not running. Please start it first.")
            print("   Run: python app.py")
            return False
        
        print("\n" + "-"*60)
        print("Running Tests...")
        print("-"*60 + "\n")
        
        # Run tests
        self.test_get_status()
        print()
        
        self.test_switch_to_sentence()
        self.test_switch_to_alphabet()
        self.test_invalid_engine()
        print()
        
        self.test_toggle_detection()
        print()
        
        self.test_set_language()
        print()
        
        self.test_clear_sentence()
        self.test_get_translation()
        print()
        
        self.test_engine_switching_sequence()
        
        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60 + "\n")
        
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%\n")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Please check the output above.")
            return False


def main():
    """Main entry point."""
    tester = ASLAppTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
