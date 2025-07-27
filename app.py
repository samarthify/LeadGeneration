import os
import io
import csv
import uuid
import re
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template
from dotenv import load_dotenv
from tavily import TavilyClient
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Initialize clients
openai.api_key = OPENAI_API_KEY
tavily = TavilyClient(TAVILY_API_KEY)

app = Flask(__name__)

# For Vercel deployment, use temporary storage
if os.environ.get('VERCEL'):
    # In Vercel environment, use /tmp for temporary files
    GENERATED_DIR = '/tmp'
else:
    # Local development
    GENERATED_DIR = 'generated'
    os.makedirs(GENERATED_DIR, exist_ok=True)

# Strengthened prompt
STRICT_CSV_PROMPT = (
    "Find and list employees for the company '{company}'. "
    "Focus on executives, VPs, directors, and regional heads. "
    "Search official company websites, press releases, reputable news, and professional networking sites. "
    "List as many relevant employees as you can find, including those from international offices or subsidiaries. "
    "For each employee, extract department and precise job title if available. "
    "Provide names in both native script (if found) and Roman alphabet. "
    "Deduplicate entries based on Romanized names (lowercase). "
    "Determine the most likely primary email domain for employees. "
    "Format the deduplicated list as a strict CSV table with these columns: "
    "Company name, Department, Job title, Last name, First name, Last name (lowercase Roman letters), First name (lowercase Roman letters), Domain likely to be used in email addresses. "
    "IMPORTANT: "
    "1. No commas within any data fields. If a field contains a comma, replace it with a semicolon or space. "
    "2. If a field is missing, leave the cell empty. "
    "3. Ensure each row has exactly 8 columns. "
    "4. Do not include duplicate entries. "
    "5. Do not truncate or cut off any entries. "
    "6. Ensure the CSV is properly formatted and complete. "
    "**Output ONLY the strict CSV data, including the header row, inside a single markdown code block (```csv ... ```), with NO extra text, explanation, or commentary.** "
    "If no data is found, output only the header row in the code block."
)

def extract_code_blocks(text):
    """Extracts content within triple backticks (code blocks)."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)

def parse_csv_from_ai_output(raw_ai_output):
    """Robust parsing that always returns a DataFrame, even with malformed data."""
    blocks = extract_code_blocks(raw_ai_output)
    if not blocks:
        return pd.DataFrame()
    
    csv_text = blocks[-1].strip()
    print(f"Original CSV text length: {len(csv_text)}")
    
    # Define expected columns
    expected_columns = [
        'Company name', 'Department', 'Job title', 'Last name', 'First name',
        'Last name (lowercase Roman letters)', 'First name (lowercase Roman letters)',
        'Domain likely to be used in email addresses'
    ]
    
    # Split into lines and process each line
    lines = csv_text.split('\n')
    processed_rows = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Skip header if it's malformed
        if i == 0 and 'Company name' not in line:
            continue
            
        # Split by comma and clean up
        parts = line.split(',')
        
        # Pad or truncate to exactly 8 columns
        while len(parts) < 8:
            parts.append('')
        parts = parts[:8]
        
        # Clean up each field
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            # Remove any @ symbols from domain field (last column)
            if len(cleaned_parts) == 7 and part.startswith('@'):
                part = part[1:]
            cleaned_parts.append(part)
        
        # Skip completely empty rows
        if all(not part for part in cleaned_parts):
            continue
            
        # Skip rows where both first and last names are empty
        if not cleaned_parts[3].strip() and not cleaned_parts[4].strip():
            continue
            
        processed_rows.append(cleaned_parts)
    
    if not processed_rows:
        print("No valid rows found, creating empty DataFrame")
        return pd.DataFrame(columns=expected_columns)
    
    # Create DataFrame
    df = pd.DataFrame(processed_rows, columns=expected_columns)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Clean up whitespace in all text columns safely
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
    
    # Remove rows where both first and last names are empty after cleaning (safely)
    try:
        # Create boolean masks safely
        first_name_empty = df['First name'].astype(str).str.strip() == ''
        last_name_empty = df['Last name'].astype(str).str.strip() == ''
        both_empty = first_name_empty & last_name_empty
        
        # Filter out rows where both names are empty
        df = df[~both_empty]
    except Exception as e:
        print(f"Warning: Could not filter empty names: {e}")
        # Continue without filtering if there's an error
    
    print(f"Successfully parsed {len(df)} rows")
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'], strict_slashes=False)
def generate():
    data = request.get_json()
    company_name = data.get('company_name', '').strip()
    if not company_name:
        return jsonify({'error': 'Company name is required.'}), 400
    try:
        # Tavily search
        query = f"{company_name} executive team site:{company_name}.co.jp"
        tavily_res = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5
        )
        urls = [item['url'] for item in tavily_res.get('results', [])][:5]
        if not urls:
            return jsonify({'error': 'No relevant pages found.'}), 404
        # OpenAI prompt
        prompt = STRICT_CSV_PROMPT.format(company=company_name) + f"\nSources: {urls}"
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200
        )
        csv_content = response.choices[0].message.content.strip()
        print("RAW AI OUTPUT:", csv_content)  # Debug logging
        df = parse_csv_from_ai_output(csv_content)
        print(f"Parsed DataFrame shape: {df.shape}")  # Debug logging
        print(f"DataFrame columns: {list(df.columns)}")  # Debug logging
        
        # Return AI response and parsing results
        result = {
            'ai_response': csv_content,
            'parsed_successfully': not df.empty,
            'rows_found': len(df) if not df.empty else 0
        }
        
        # Only create CSV file if parsing was successful
        if not df.empty:
            filename = f"{company_name}_{uuid.uuid4().hex[:8]}.csv"
            filepath = os.path.join(GENERATED_DIR, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            result['download_url'] = f'/download/{filename}'
        else:
            # Add warning message when parsing fails but AI response is available
            result['warning'] = 'Parsing failed but AI response is available below for manual extraction.'
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'ai_response': 'Error occurred before AI processing could complete.'
        }), 500



@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)

# For Vercel deployment
app.debug = False

if __name__ == '__main__':
    app.run(debug=True) 