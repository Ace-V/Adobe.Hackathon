import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document


load_dotenv()

# Configure embeddings
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit
st.set_page_config(
    page_title="PDF Document Outline Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Document Outline Extractor")
st.markdown("""
Upload a PDF document and automatically extract its structured outline including title and hierarchical headings.
The tool will identify H1, H2, and H3 level headings with their page numbers.
""")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Groq API key:", type="password", help="Get your API key from https://console.groq.com/")
    
    if api_key:
        st.success("✅ API key configured")
    else:
        st.warning("⚠️ Please enter your Groq API key to continue")

def extract_text_with_metadata(pdf_path: str) -> List[Dict]:
    """Extract text from PDF with page numbers and formatting hints"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_data = []
    for page in pages:
        # Split text into lines and analyze each line
        lines = page.page_content.split('\n')
        page_num = page.metadata.get('page', 1) + 1  # Convert to 1-based indexing
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                text_data.append({
                    'text': line,
                    'page': page_num,
                    'length': len(line),
                    'is_upper': line.isupper(),
                    'word_count': len(line.split()),
                    'has_numbers': bool(re.search(r'\d+\.', line))  # Check for numbering
                })
    
    return text_data

def create_enhanced_extraction_prompt() -> str:
    """Create an enhanced prompt for better document structure extraction"""
    prompt = """You are an expert document structure analyzer. Your task is to extract a clean, hierarchical outline from a PDF document.

ANALYSIS INSTRUCTIONS:
1. Identify the main document title (usually the largest text at the beginning)
2. Find hierarchical headings by analyzing:
   - Text formatting patterns (longer lines often indicate headings)
   - Numbering schemes (1., 1.1, 1.1.1, I., A., etc.)
   - Context and logical flow
   - Position in document structure

HEADING CLASSIFICATION RULES:
- H1: Major sections, chapters, or main topics (often numbered 1., 2., 3., etc.)
- H2: Subsections under H1 (often numbered 1.1, 1.2, 2.1, etc.)
- H3: Sub-subsections under H2 (often numbered 1.1.1, 1.1.2, etc.)

EXTRACTION GUIDELINES:
- Focus on structural headings, not body text
- Ignore headers, footers, page numbers, table of contents entries
- Look for consistent formatting patterns
- Consider section depth and logical hierarchy
- Extract ALL relevant headings (H1, H2, H3) from the entire document
- Provide accurate page numbers for each heading

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure:
{{
  "title": "Main Document Title",
  "outline": [
    {{
      "level": "H1",
      "text": "Section Title",
      "page": 1
    }},
    {{
      "level": "H2", 
      "text": "Subsection Title",
      "page": 2
    }},
    {{
      "level": "H3",
      "text": "Sub-subsection Title",
      "page": 3
    }}
  ]
}}

DOCUMENT CONTENT TO ANALYZE:
{context}

Remember: Return ONLY the JSON object, no additional text or explanations."""
    return prompt

def clean_and_validate_json(response_text: str) -> Optional[Dict]:
    """Clean and validate JSON response from LLM"""
    try:
        # First try direct parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        json_patterns = [
            r'```json\s*({.*?})\s*```',  # JSON in code blocks
            r'```\s*({.*?})\s*```',       # Generic code blocks
            r'({\s*"title".*?})',          # JSON starting with title
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None

def format_outline_display(json_data: Dict) -> None:
    """Display the extracted outline in a readable format"""
    if not json_data:
        return
    
    st.subheader("📖 Extracted Document Structure")
    
    # Display title
    if 'title' in json_data and json_data['title']:
        st.markdown(f"**📋 Document Title:** {json_data['title']}")
        st.markdown("---")
    
    # Display outline
    if 'outline' in json_data and json_data['outline']:
        st.markdown("**🗂️ Document Outline:**")
        
        # Count headings by level
        h1_count = sum(1 for item in json_data['outline'] if item.get('level') == 'H1')
        h2_count = sum(1 for item in json_data['outline'] if item.get('level') == 'H2')
        h3_count = sum(1 for item in json_data['outline'] if item.get('level') == 'H3')
        
        st.info(f"📊 Found: {h1_count} H1 headings, {h2_count} H2 headings, {h3_count} H3 headings")
        
        for item in json_data['outline']:
            level = item.get('level', 'H1')
            text = item.get('text', 'Untitled')
            page = item.get('page', 'N/A')
            
            # Create indentation based on heading level
            if level == 'H1':
                st.markdown(f"**1️⃣ {text}** *(Page {page})*")
            elif level == 'H2':
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**2️⃣ {text}** *(Page {page})*")
            elif level == 'H3':
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**3️⃣ {text}** *(Page {page})*")
    else:
        st.warning("No outline structure was extracted from the document.")

def main():
    """Main application logic"""
    if not api_key:
        st.info("👈 Please enter your Groq API key in the sidebar to get started.")
        return
    
    # Initialize LLM with better model for complex document analysis
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-3b-It", temperature=0)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document (up to 50 pages recommended)"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"📊 File size: {file_size:.2f} MB")
        
        # Auto-extract outline when file is uploaded
        with st.spinner("🔍 Analyzing document structure..."):
            try:
                # Extract text with metadata
                text_data = extract_text_with_metadata(temp_path)
                
                if not text_data:
                    st.error("❌ Could not extract text from the PDF. Please check if the file is readable.")
                    return
                
                # Prepare context for LLM (use more content for better analysis)
                context_lines = []
                for item in text_data:
                    context_lines.append(f"Page {item['page']}: {item['text']}")
                
                # Use more context but still within token limits
                context = "\n".join(context_lines[:1000])  # Increased from 500 to 1000 lines
                
                # Create extraction prompt
                prompt_template = ChatPromptTemplate.from_template(create_enhanced_extraction_prompt())
                
                # Create document chain
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                
                # Process with LLM
                
                doc = Document(page_content=context)
                response = document_chain.invoke({"context": [doc]})
                
                # Clean and validate response
                json_data = clean_and_validate_json(response)
                
                if json_data:
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        format_outline_display(json_data)
                    
                    with col2:
                        st.subheader("📋 Raw JSON Output")
                        st.json(json_data)
                        
                        # Download button
                        json_string = json.dumps(json_data, indent=2)
                        title = json_data.get('title', 'document').replace(' ', '_')
                        st.download_button(
                            label="💾 Download JSON",
                            data=json_string,
                            file_name=f"{title}_outline.json",
                            mime="application/json"
                        )
                
                else:
                    st.error("❌ Failed to extract a valid document structure. The document might not have clear headings.")
                    with st.expander("🔍 View raw response for debugging"):
                        st.text(response)
                        
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()