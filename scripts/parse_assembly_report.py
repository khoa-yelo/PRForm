import json
import csv
import os

def parse_assembly_report(jsonl_path, csv_path):
    print(f"Reading {jsonl_path}...")
    count = 0
    with open(jsonl_path, 'r') as f_in, open(csv_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        # Header
        writer.writerow([
            'accession', 
            'assembly_level', 
            'assembly_name', 
            'submitter', 
            'total_sequence_length', 
            'gc_percent', 
            'taxid', 
            'organism_name', 
            'release_date'
        ])
        
        for line in f_in:
            data = json.loads(line)
            
            accession = data.get('accession', '')
            info = data.get('assemblyInfo', {})
            stats = data.get('assemblyStats', {})
            organism = data.get('organism', {})
            
            writer.writerow([
                accession,
                info.get('assemblyLevel', ''),
                info.get('assemblyName', ''),
                info.get('submitter', ''),
                stats.get('totalSequenceLength', ''),
                stats.get('gcPercent', ''),
                organism.get('taxId', ''),
                organism.get('organismName', ''),
                info.get('releaseDate', '')
            ])
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} lines...")
                
    print(f"Done! Processed {count} entries.")

if __name__ == "__main__":
    input_jsonl = 'viral_complete_subset_data/ncbi_dataset/data/assembly_data_report.jsonl'
    output_csv = 'assembly_data_report.csv'
    
    if os.path.exists(input_jsonl):
        parse_assembly_report(input_jsonl, output_csv)
    else:
        print(f"Error: {input_jsonl} not found.")
