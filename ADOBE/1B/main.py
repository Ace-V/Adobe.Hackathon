import json
import time
import fitz  # PyMuPDF
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class DocumentProcessor:
    def _init_(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_pdf_content(self, file_path: str) -> list:
        """Extract text from PDF with page numbers"""
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return []
                
            doc = fitz.open(file_path)
            blocks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if text:  # Only add non-empty pages
                    blocks.append({
                        "text": text,
                        "page": page_num + 1,
                        "file": os.path.basename(file_path)
                    })
            return blocks
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []
            return blocks
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def semantic_rank(self, query: str, texts: list) -> np.ndarray:
        """Rank texts by similarity to query"""
        query_embed = self.model.encode([query])
        doc_embeds = self.model.encode(texts)
        return cosine_similarity(query_embed, doc_embeds)[0]

    def process_documents(self, input_data: dict) -> dict:
        """Main processing function"""
        start_time = time.time()
        
        # Get documents from input
        documents = input_data.get("documents", [])
        
        # Prepare search context
        context = "Travel planning for South of France focusing on gastronomy and historical sites"
        
        # Extract text from all PDFs
        all_blocks = []
        for doc in documents:
            filename = doc.get("filename")
            if filename:
                blocks = self.extract_pdf_content(filename)
                all_blocks.extend(blocks)
        for doc in documents:
            filename = doc.get("filename")
            if filename:
                blocks = self.extract_pdf_content(filename)
                all_blocks.extend(blocks)
        
        if not all_blocks:
            return {
                "error": "No content could be extracted from PDFs",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        
        # Rank the extracted content
        texts = [b["text"] for b in all_blocks]
        scores = self.semantic_rank(context, texts)
        
        # Add scores to blocks and sort
        for i, block in enumerate(all_blocks):
            block["score"] = float(scores[i])
        ranked_blocks = sorted(all_blocks, key=lambda x: x["score"], reverse=True)
        
        # Prepare the output structure
        output = {
            "metadata": {
                "challenge_info": input_data.get("challenge_info", {}),
                "processed_files": [os.path.basename(d.get("filename", "")) for d in documents],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "processing_time_sec": round(time.time() - start_time, 2)
            },
            "results": {
                "top_sections": self._format_sections(ranked_blocks[:10]),
                "key_passages": self._format_subsections(ranked_blocks[:5])
            }
        }
        
        return output
    
    def _format_sections(self, blocks: list) -> list:
        """Format the top sections"""
        return [{
            "document": b["file"],
            "page": b["page"],
            "title": f"{b['file']} - Page {b['page']}",
            "relevance_score": b["score"]
        } for b in blocks]
    
    def _format_subsections(self, blocks: list) -> list:
        """Format the key passages"""
        return [{
            "document": b["file"],
            "page": b["page"],
            "text": b["text"][:1500] + ("..." if len(b["text"]) > 1500 else ""),
            "relevance_score": b["score"]
        } for b in blocks]

def main():
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # File paths
    input_path = "input.json"
    output_path = os.path.join("output", "output.json")
    
    # Try to load input
    try:
        with open(input_path, "r") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        error_output = {"error": f"Input file not found: {input_path}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, "w") as f:
            json.dump(error_output, f, indent=2)
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        error_output = {"error": f"Invalid JSON: {str(e)}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(output_path, "w") as f:
            json.dump(error_output, f, indent=2)
        print(f"Error: Invalid JSON in input file")
        return
    
    # Process documents
    processor = DocumentProcessor()
    result = processor.process_documents(input_data)
    
    # Save output
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Success! Output saved to {output_path}")

if __name__ == "_main_":
    main()