# dump_tsv_headers.py

import pandas as pd
from pathlib import Path

# --- Configuration ---
RAW_DATA_DIR = Path("ml/data/raw/mtg-jamendo")
METADATA_AND_FEATURES_FILE = RAW_DATA_DIR / "raw.meta.tsv" 

def dump_headers():
    """Reads the TSV and prints all column names."""
    
    print(f"--- Inspecting headers for: {METADATA_AND_FEATURES_FILE} ---")
    
    if not METADATA_AND_FEATURES_FILE.exists():
        print(f"ERROR: File not found at {METADATA_AND_FEATURES_FILE}. Cannot proceed.")
        return

    try:
        # Load only the header and first row to find the column names
        df = pd.read_csv(METADATA_AND_FEATURES_FILE, sep='\t', nrows=1)
        
        print("\n" + "="*80)
        print("RAW COLUMN HEADERS FOUND IN YOUR raw.meta.tsv:")
        print("="*80)
        # Print as a list for easy copying
        print(df.columns.tolist())
        print("="*80)
        
        print("\n[ACTION REQUIRED] Please provide the list above. We will use it to create the FINAL FEATURE_MAP.")

    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        print("Ensure the file exists and is tab-separated.")

if __name__ == "__main__":
    dump_headers()