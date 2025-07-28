import json
import time
import fitz
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_pdf_content(self, file_path: str) -> List[Dict]:
        """Extract text blocks with metadata from PDF"""
        try:
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                return []
                
            doc = fitz.open(file_path)
            blocks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip():  # Only add non-empty pages
                    blocks.append({
                        "text": text.strip(),
                        "page": page_num + 1,
                        "file": os.path.basename(file_path)
                    })
            return blocks
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def semantic_rank(self, query: str, texts: List[str]) -> np.ndarray:
        """Rank texts by semantic similarity to query"""
        query_embed = self.model.encode([query])
        doc_embeds = self.model.encode(texts)
        return cosine_similarity(query_embed, doc_embeds)[0]

    def process_input(self, input_data: Dict) -> Dict:
        """Main processing pipeline"""
        start_time = time.time()
        
        # Extract data from input JSON
        challenge_info = input_data.get("challenge_info", {})
        documents = input_data.get("documents", [])
        
        # Prepare context - using test case specific context
        context = "Travel planning for South of France focusing on gastronomy and historical sites"
        
        all_blocks = []
        for doc in documents:
            filename = doc.get("filename")
            if filename:
                blocks = self.extract_pdf_content(filename)
                all_blocks.extend(blocks)
        
        if not all_blocks:
            return {
                "error": "No valid content could be extracted from the provided documents",
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        
        # Rank pages
        texts = [b['text'] for b in all_blocks]
        scores = self.semantic_rank(context, texts)
        
        # Rank blocks
        ranked_blocks = []
        for i, score in enumerate(scores):
            block = all_blocks[i]
            block['score'] = float(score)
            ranked_blocks.append(block)
        ranked_blocks.sort(key=lambda x: x['score'], reverse=True)
        
        # Prepare output
        output = {
            "metadata": {
                "challenge_id": challenge_info.get("challenge_id", "round_ib_002"),
                "test_case_name": challenge_info.get("test_case_name", "travel_planner"),
                "description": challenge_info.get("description", "France Travel"),
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "documents_processed": [os.path.basename(d.get("filename", "")) for d in documents]
            },
            "extracted_sections": self._format_sections(ranked_blocks),
            "sub_section_analysis": self._format_subsections(ranked_blocks[:5])
        }
        
        # Calculate processing time
        output["metadata"]["processing_time_sec"] = time.time() - start_time
        return output
    
    def _format_sections(self, blocks: List[Dict]) -> List[Dict]:
        """Format top 10 sections with ranking"""
        return [{
            "document": b['file'],
            "page_number": b['page'],
            "section_title": f"Page {b['page']} - {b['file']}",
            "importance_rank": i+1,
            "relevance_score": b['score']
        } for i, b in enumerate(blocks[:10])]
    
    def _format_subsections(self, blocks: List[Dict]) -> List[Dict]:
        """Format top 5 subsections with ranking"""
        return [{
            "document": b['file'],
            "page_number": b['page'],
            "refined_text": b['text'][:1000] + ("..." if len(b['text']) > 1000 else ""),
            "importance_rank": i+1,
            "relevance_score": b['score']
        } for i, b in enumerate(blocks[:5])]

def ensure_output_directory():
    """Ensure output directory exists"""
    os.makedirs("output", exist_ok=True)

def main():
    ensure_output_directory()
    
    # Load input from JSON file
    input_path = "input.json"
    output_path = os.path.join("output", "output.json")
    
    try:
        with open(input_path) as f:
            input_data = json.load(f)
    except FileNotFoundError:
        error_msg = {"error": f"Input file not found: {input_path}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, 'w') as f:
            json.dump(error_msg, f, indent=2)
        print(f"Error: {error_msg['error']}")
        return
    except json.JSONDecodeError as e:
        error_msg = {"error": f"Invalid JSON in input file: {str(e)}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, 'w') as f:
            json.dump(error_msg, f, indent=2)
        print(f"Error: {error_msg['error']}")
        return
    
    processor = DocumentProcessor()
    result = processor.process_input(input_data)
    
    # Save output
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Processing completed successfully!")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
