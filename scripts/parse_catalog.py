import json
import csv
import os

def parse_catalog(json_path, csv_path):
    print(f"Reading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # The structure is a list of assemblies at the top level
    assemblies = data.get('assemblies', [])
    
    print(f"Processing {len(assemblies)} assemblies...")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['accession', 'file_path', 'file_type', 'size_bytes'])
        
        for assembly in assemblies:
            accession = assembly.get('accession')
            files = assembly.get('files', [])
            for file_info in files:
                writer.writerow([
                    accession,
                    file_info.get('filePath'),
                    file_info.get('fileType'),
                    file_info.get('uncompressedLengthBytes')
                ])

if __name__ == "__main__":
    input_file = "viral_complete_subset_data/ncbi_dataset/data/dataset_catalog.json"
    output_file = "dataset_catalog.csv"
    
    if os.path.exists(input_file):
        parse_catalog(input_file, output_file)
        print(f"Saved to {output_file}")
    else:
        print(f"Error: {input_file} not found.")
