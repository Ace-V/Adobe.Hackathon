import json
import time
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_pdf_content(self, file_path: str) -> List[Dict]:
    #Extract text blocks with metadata from PDF
        doc = fitz.open(file_path)
        blocks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text, block_no, block_type = block
                if block_type == 0:  # Text block
                    blocks.append({
                        "text": text.strip(),
                        "page": page_num + 1,
                        "coordinates": (x0, y0)
                    })
        return blocks

    def semantic_rank(self, query: str, texts: List[str]) -> np.ndarray:
        #Rank texts COS SIMILARITY
        query_embed = self.model.encode([query])
        doc_embeds = self.model.encode(texts)
        return cosine_similarity(query_embed, doc_embeds)[0]

    def process_input(self, input_data: Dict) -> Dict:
        #Main processing ppl
        start_time = time.time()
        
        # Prepare context
        context = f"{input_data['persona']} performing: {input_data['job']}"
        all_blocks = []
        for doc_path in input_data['documents']:
            blocks = self.extract_pdf_content(doc_path)
            for block in blocks:
                block['document'] = doc_path
            all_blocks.extend(blocks)
        
        # Flter empty blocks
        content_blocks = [b for b in all_blocks if b['text']]
        
        # Rank sections
        texts = [b['text'] for b in content_blocks]
        scores = self.semantic_rank(context, texts)
        
        # rank blocks
        ranked_blocks = []
        for i, score in enumerate(scores):
            block = content_blocks[i]
            block['score'] = float(score)
            ranked_blocks.append(block)
        ranked_blocks.sort(key=lambda x: x['score'], reverse=True)
        
        # repare output
        return {
            "metadata": {
                "input_documents": input_data['documents'],
                "persona": input_data['persona'],
                "job_to_be_done": input_data['job'],
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "extracted_sections": self._format_sections(ranked_blocks),
            "sub_section_analysis": self._format_subsections(ranked_blocks[:20]),
            "processing_time_sec": time.time() - start_time
        }
    
    def _format_sections(self, blocks: List[Dict]) -> List[Dict]:
        """Format top 10 sections with ranking"""
        return [{
            "document": b['document'],
            "page_number": b['page'],
            "section_title": b['text'][:50] + ("..." if len(b['text']) > 50 else ""),
            "importance_rank": i+1
        } for i, b in enumerate(blocks[:10])]
    
    def _format_subsections(self, blocks: List[Dict]) -> List[Dict]:
        """Format top 5 subsections with ranking"""
        return [{
            "document": b['document'],
            "page_number": b['page'],
            "refined_text": b['text'],
            "importance_rank": i+1
        } for i, b in enumerate(blocks[:5])]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.json", help="Input JSON file")
    parser.add_argument("--output", default="output.json", help="Output JSON file")
    args = parser.parse_args()
    
    with open(args.input) as f:
        input_data = json.load(f)
    
    processor = DocumentProcessor()
    result = processor.process_input(input_data)
    
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)