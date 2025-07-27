import pdfplumber
import re
import numpy as np
from collections import defaultdict

def extract_outline(pdf_path):
    title = ""
    headings = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # Extract title from first page
        first_page = pdf.pages[0]
        title_candidates = []
        
        # Process words in reading order
        words = first_page.extract_words(
            extra_attrs=["size", "fontname"],
            keep_blank_chars=False
        )
        
        # Group words by line
        lines = defaultdict(list)
        for word in words:
            # Round to handle floating point precision issues
            top_key = round(word['top'])
            lines[top_key].append(word)
        
        # Find title candidate (top-most large text)
        for top in sorted(lines.keys()):
            line_words = lines[top]
            min_x0 = min(word['x0'] for word in line_words)
            max_x1 = max(word['x1'] for word in line_words)
            text = " ".join(word['text'] for word in line_words)
            avg_size = np.mean([word['size'] for word in line_words])
            
            # Consider as title candidate if large and centered
            if avg_size > 20 and (min_x0 < 100 and max_x1 > first_page.width - 100):
                title_candidates.append((avg_size, text))
        
        if title_candidates:
            title_candidates.sort(key=lambda x: x[0], reverse=True)
            title = title_candidates[0][1]
        
        # Extract headings from all pages
        heading_candidates = []
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(
                extra_attrs=["size", "fontname"],
                keep_blank_chars=False
            )
            
            # Group words into lines
            lines = defaultdict(list)
            for word in words:
                top_key = round(word['top'])
                lines[top_key].append(word)
            
            # Process each line
            for top in sorted(lines.keys()):
                line_words = lines[top]
                min_x0 = min(word['x0'] for word in line_words)
                max_x1 = max(word['x1'] for word in line_words)
                text = " ".join(word['text'] for word in line_words)
                max_size = max(word['size'] for word in line_words)
                fonts = [word['fontname'] for word in line_words]
                
                # Heading characteristics:
                # - Larger than body text
                # - Not too close to page edges
                # - Not too long
                if (max_size > 10 and 
                    min_x0 > 50 and 
                    max_x1 < page.width - 50 and
                    len(text) < 200 and 
                    len(line_words) < 10):
                    heading_candidates.append({
                        "text": text,
                        "size": max_size,
                        "page": page_number,
                        "y": top
                    })
        
        # Cluster heading levels
        if heading_candidates:
            sizes = [c['size'] for c in heading_candidates]
            distinct_sizes = sorted(set(sizes), reverse=True)[:3]
            
            # Assign heading levels
            level_map = {}
            for i, size in enumerate(distinct_sizes):
                level_map[size] = f"H{i+1}"
            
            # Create outline with hierarchical validation
            current_level = 0
            for candidate in sorted(heading_candidates, key=lambda x: (x['page'], x['y'])):
                level = level_map.get(candidate['size'], None)
                if level:
                    level_num = int(level[1:])
                    
                    # Maintain hierarchy (H1 > H2 > H3)
                    if level_num <= current_level + 1:
                        headings.append({
                            "level": level,
                            "text": candidate['text'],
                            "page": candidate['page']
                        })
                        current_level = level_num
    
    return {
        "title": title,
        "outline": headings
    }