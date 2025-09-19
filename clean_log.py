#!/usr/bin/env python3
import sys
import os

def clean_log_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return False
    
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        original_count = len(lines)
        cleaned_lines = [line for line in lines if '[Warning]' not in line]
        removed_count = original_count - len(cleaned_lines)
        
        with open(filepath, 'w') as file:
            file.writelines(cleaned_lines)
        
        print(f"Cleaned '{filepath}':")
        print(f"  Original lines: {original_count}")
        print(f"  Removed lines: {removed_count}")
        print(f"  Remaining lines: {len(cleaned_lines)}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_log.py <log_file_path>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    success = clean_log_file(log_file)
    sys.exit(0 if success else 1)