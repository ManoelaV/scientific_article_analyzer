#!/usr/bin/env python3
"""
Master Test Runner
Executes all three test scenarios in sequence
"""

import asyncio
import sys
import subprocess
from pathlib import Path
import time

def run_test_script(script_name: str, test_description: str) -> bool:
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"🚀 Running {test_description}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / script_name
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n📊 Test Results for {test_description}:")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Exit code: {result.returncode}")
        
        if result.stdout:
            print(f"\n📝 Output:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print(f"\n❌ Errors:")
            print(result.stderr)
        
        success = result.returncode == 0
        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {test_description}")
        
        return success
        
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {str(e)}")
        return False

def main():
    """Run all test scenarios."""
    print("🧪 Scientific Article Analyzer - Complete Test Suite")
    print("=" * 60)
    
    # Test configuration
    tests = [
        ("test_pdf_processing.py", "Test 1: PDF Classification and Extraction"),
        ("test_url_processing.py", "Test 2: URL Article Processing"), 
        ("test_edge_case.py", "Test 3: Edge Case Handling")
    ]
    
    # Track results
    results = []
    start_time = time.time()
    
    # Run each test
    for script_name, description in tests:
        success = run_test_script(script_name, description)
        results.append((description, success))
        
        # Brief pause between tests
        if script_name != tests[-1][0]:  # Not the last test
            time.sleep(2)
    
    # Summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"📋 TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed_count = 0
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status}: {description}")
        if success:
            passed_count += 1
    
    print(f"\n📊 Results: {passed_count}/{len(tests)} tests passed")
    print(f"⏱️  Total duration: {total_duration:.1f} seconds")
    
    # Check output files
    print(f"\n📁 Generated Output Files:")
    output_dir = Path(__file__).parent.parent / "out"
    if output_dir.exists():
        output_files = list(output_dir.glob("*.json")) + list(output_dir.glob("*.md"))
        for file_path in sorted(output_files):
            size_kb = file_path.stat().st_size / 1024
            print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
    else:
        print("   ⚠️  Output directory not found")
    
    # Final status
    all_passed = passed_count == len(tests)
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! System is ready for demonstration.")
        print(f"   • PDF processing: ✅")
        print(f"   • URL processing: ✅") 
        print(f"   • Edge case handling: ✅")
    else:
        print(f"\n⚠️  {len(tests) - passed_count} test(s) failed. Review results above.")
    
    # Return appropriate exit code
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)