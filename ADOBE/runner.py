import os
import json
from extractor import extract_outline

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

if __name__ == "__main__":
    # Process all PDFs in input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            print(f"Processing: {filename}")
            
            try:
                # Extract outline
                result = extract_outline(pdf_path)
                
                # Save results
                json_filename = f"{os.path.splitext(filename)[0]}.json"
                output_path = os.path.join(OUTPUT_DIR, json_filename)
                
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                    
                print(f"Created: {json_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")