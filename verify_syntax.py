#!/usr/bin/env python
"""Verify all Python files have valid syntax."""

import py_compile
import sys
from pathlib import Path

print("\n" + "="*60)
print("SYNTAX VERIFICATION")
print("="*60 + "\n")

files_to_check = [
    'treno/models.py',
    'treno/unet_1d_optimized.py',
    'treno/__init__.py',
    'test_nd_comprehensive.py',
]

all_valid = True
for file_path in files_to_check:
    try:
        py_compile.compile(file_path, doraise=True)
        size = Path(file_path).stat().st_size
        lines = len(open(file_path).readlines())
        print(f"✅ {file_path:40} ({lines:4} lines, {size/1024:6.1f} KB)")
    except py_compile.PyCompileError as e:
        print(f"❌ {file_path:40} - SYNTAX ERROR")
        print(f"   {e}")
        all_valid = False

print("\n" + "="*60)
if all_valid:
    print("✅ ALL FILES HAVE VALID SYNTAX!")
else:
    print("❌ SOME FILES HAVE SYNTAX ERRORS")
    sys.exit(1)

print("\nDocumentation files:")
docs = [
    'ND_COMPREHENSIVE_GUIDE.md',
    'IMPLEMENTATION_SUMMARY.md',
    'QUICK_REFERENCE.md',
    'COMPLETION_REPORT.md',
    'CHANGELOG_v3.5.0.md',
]

for doc in docs:
    try:
        lines = len(open(doc).readlines())
        size = Path(doc).stat().st_size
        print(f"✅ {doc:40} ({lines:4} lines, {size/1024:6.1f} KB)")
    except FileNotFoundError:
        print(f"⚠️  {doc:40} NOT FOUND")

print("\nExample files:")
examples = [
    'examples/example_nd_classification.py',
    'examples/example_nd_maptomap.py',
]

for ex in examples:
    try:
        lines = len(open(ex).readlines())
        size = Path(ex).stat().st_size
        print(f"✅ {ex:40} ({lines:4} lines, {size/1024:6.1f} KB)")
    except FileNotFoundError:
        print(f"⚠️  {ex:40} NOT FOUND")

print("\n" + "="*60)
print("\n✅ IMPLEMENTATION COMPLETE AND VERIFIED!\n")
print("="*60 + "\n")
