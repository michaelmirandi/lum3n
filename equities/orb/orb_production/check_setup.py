#!/usr/bin/env python3
"""
Quick setup verification for Phase 1
Checks dependencies and basic imports
"""
import sys
from pathlib import Path

def check_imports():
    """Check all required imports"""
    print("🔍 Checking imports...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'ib_insync',
        'yaml',
        'asyncio'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_file_structure():
    """Check file structure is correct"""
    print("\n🔍 Checking file structure...")
    
    required_files = [
        'orb_live/__init__.py',
        'orb_live/data/__init__.py',
        'orb_live/data/stream.py',
        'orb_live/data/aggregator.py',
        'orb_live/data/cache.py',
        'orb_live/core/__init__.py',
        'orb_live/utils/__init__.py',
        'test_phase1.py',
        'requirements.txt'
    ]
    
    missing = []
    base_path = Path(__file__).parent
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️ Missing files: {missing}")
        return False
    
    return True

def check_module_imports():
    """Check our module imports work"""
    print("\n🔍 Checking module imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        from orb_live.data.aggregator import CandleAggregator
        print("  ✅ CandleAggregator")
        
        from orb_live.data.cache import DataCache
        print("  ✅ DataCache")
        
        from orb_live.data.stream import LiveDataStream
        print("  ✅ LiveDataStream")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Module import failed: {e}")
        return False

def main():
    """Run all setup checks"""
    print("="*50)
    print("ORB PRODUCTION - PHASE 1 SETUP CHECK")
    print("="*50)
    
    checks = [
        ("Package Dependencies", check_imports),
        ("File Structure", check_file_structure),
        ("Module Imports", check_module_imports)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        try:
            result = check_func()
            all_passed = all_passed and result
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("Ready to run: python test_phase1.py")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please fix the issues above before testing.")
    print("="*50)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)