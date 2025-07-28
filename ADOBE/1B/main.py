import json
import time
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class DocumentProcessor:
    def _init_(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_pdf_content(self, file_path: str) -> List[Dict]:
        """Extract text blocks with metadata from PDF"""
        try:
            doc = fitz.open(file_path)
            blocks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                blocks.append({
                    "text": text.strip(),
                    "page": page_num + 1,
                    "file": file_path
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
        
        # Extract challenge info and documents from the nested structure
        challenge_info = input_data.get("challenge_info", {})
        documents = input_data.get("documents", [])
        
        # Prepare context
        context = "Travel planning for South of France focusing on gastronomy and historical sites"
        
        all_blocks = []
        for doc in documents:
            filename = doc.get("filename")
            if filename:
                blocks = self.extract_pdf_content(filename)
                if blocks:  # Only add if we have blocks
                    all_blocks.extend(blocks)
        
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
        return {
            "metadata": {
                "challenge_id": challenge_info.get("challenge_id"),
                "test_case_name": challenge_info.get("test_case_name"),
                "description": challenge_info.get("description"),
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "extracted_sections": self._format_sections(ranked_blocks),
            "sub_section_analysis": self._format_subsections(ranked_blocks[:5])
        }
    
    def _format_sections(self, blocks: List[Dict]) -> List[Dict]:
        """Format top 10 sections with ranking"""
        return [{
            "document": b['file'],
            "page_number": b['page'],
            "section_title": f"Page {b['page']}",
            "importance_rank": i+1
        } for i, b in enumerate(blocks[:10])]
    
    def _format_subsections(self, blocks: List[Dict]) -> List[Dict]:
        """Format top 5 subsections with ranking"""
        return [{
            "document": b['file'],
            "page_number": b['page'],
            "refined_text": b['text'][:500] + ("..." if len(b['text']) > 500 else ""),
            "importance_rank": i+1
        } for i, b in enumerate(blocks[:5])]

def main():
    # Load input from JSON file
    try:
        with open('input.json') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print("Error: input.json file not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input.json")
        return
    
    processor = DocumentProcessor()
    result = processor.process_input(input_data)
    
    # Save output
    with open('output.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Processing completed successfully!")
    print(f"Output saved to output.json")

if _name_ == "_main_":
    main()
