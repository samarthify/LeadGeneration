import os
import io
import csv
import uuid
import re
import pandas as pd
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template, Response
from dotenv import load_dotenv
from tavily import TavilyClient
import openai
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Setup logging
def setup_logging(log_filename: str) -> logging.Logger:
    """Configure logging to both file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to prevent duplicate logs
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Console handler (always available)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    
    # File handler (only if not in Vercel environment)
    if not os.environ.get('VERCEL'):
        try:
            file_handler = logging.FileHandler(log_filename, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, just log to console
            logger.warning(f"Failed to setup file logging: {e}")
    
    return logger

# Configuration management
@dataclass
class Config:
    """Centralized configuration for the lead generation application."""
    # OpenAI settings
    model: str = "gpt-4o"
    max_tokens: Optional[int] = 2000
    temperature: float = 0.2
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # API settings
    max_retries: int = 3
    request_delay: float = 0.5
    
    # Tavily settings
    tavily_max_results: int = 5
    tavily_search_depth: str = "advanced"
    
    # Processing settings
    max_urls_per_company: int = 8
    max_search_results: int = 10
    
    # File settings
    generated_dir: str = 'generated'
    log_filename: str = 'lead_generation.log'
    
    # CSV settings
    expected_columns: list = None
    
    def __post_init__(self):
        if self.expected_columns is None:
            self.expected_columns = [
                'Company name', 'Department', 'Job title', 'Last name', 'First name',
                'Last name (lowercase Roman letters)', 'First name (lowercase Roman letters)',
                'Domain likely to be used in email addresses'
            ]
            # Also support Japanese column names
            self.expected_columns_japanese = [
                '会社名', '部署', '役職', '姓', '名',
                '姓（小文字ローマ字）', '名（小文字ローマ字）',
                'メールアドレスに使用される可能性が高いドメイン'
            ]
        
        # Adjust file paths for Vercel environment
        if os.environ.get('VERCEL'):
            self.generated_dir = '/tmp'
            self.log_filename = '/tmp/lead_generation.log'

# Initialize configuration
config = Config()

# Initialize logging
try:
    logger = setup_logging(config.log_filename)
except Exception as e:
    # Fallback to basic console logging if setup fails
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.warning(f"Failed to setup logging with file handler: {e}")

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
    GENERATED_DIR = config.generated_dir
    os.makedirs(GENERATED_DIR, exist_ok=True)

# Define tools for OpenAI function calling
def define_tools():
    """Define the tools available to the AI."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for real-time information about company employees, executives, and organizational structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string to find employee information.",
                        },
                        "site": {
                            "type": "string",
                            "description": "Optional domain to restrict search to (e.g., 'company.com').",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

def make_api_request_with_tools(messages, max_retries=None, retry_count=0):
    """Call OpenAI chat completion with retries and tool handling."""
    if max_retries is None:
        max_retries = config.max_retries
    
    logger.info(f"Making API request (attempt {retry_count + 1}/{max_retries + 1})")
    
    try:
        resp = openai.chat.completions.create(
            model=config.model,
            messages=messages,
            tools=define_tools(),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
        )

        # Handle tool calls
        if resp.choices[0].message.tool_calls:
            tool_calls = resp.choices[0].message.tool_calls
            tool_outputs = []
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name == "web_search":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        query = args.get("query")
                        site = args.get("site", "")
                        
                        if query:
                            logger.info(f"AI requested search: {query}")
                            if site:
                                query = f"{query} site:{site}"
                            
                            # Execute Tavily search
                            search_results = tavily.search(
                                query=query,
                                search_depth=config.tavily_search_depth,
                                max_results=config.tavily_max_results
                            ).get('results', [])
                            
                            # Format results for the model
                            formatted_results = "\n".join([
                                f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content'][:1000]}..."
                                for res in search_results
                            ])
                            
                            if not formatted_results:
                                formatted_results = "No search results found."
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": formatted_results,
                            })
                        else:
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": "Error: Missing 'query' argument for web_search.",
                            })
                    except json.JSONDecodeError:
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": "Error: Invalid JSON arguments for web_search.",
                        })
                    except Exception as e:
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": f"Error during web_search execution: {e}",
                        })
                else:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": f"Error: Unknown tool function: {function_name}",
                    })

            # Submit tool outputs and get the final response
            messages.append(resp.choices[0].message)
            messages.extend([
                {"role": "tool", "tool_call_id": to["tool_call_id"], "content": to["output"]} 
                for to in tool_outputs
            ])

            # Make another API call with tool outputs
            messages_with_tool_outputs = [
                {"role": "system", "content": "Process the search results provided by the tool to extract the requested employee information and format it as a strict CSV within a markdown code block, including the header row, as requested in the user prompt."},
                *messages
            ]
            
            resp_with_outputs = openai.chat.completions.create(
                model=config.model,
                messages=messages_with_tool_outputs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty
            )
            return resp_with_outputs.choices[0].message.content.strip()

        # Handle regular text response
        return resp.choices[0].message.content.strip()

    except Exception as e:
        if retry_count < max_retries:
            logger.warning(f"API request failed (attempt {retry_count + 1}): {e}")
            time.sleep(config.request_delay * (2 ** retry_count))  # Exponential back-off with configurable delay
            return make_api_request_with_tools(messages, max_retries, retry_count + 1)
        logger.error(f"API request failed permanently: {e}")
        return None

def fetch_url_content(url, max_length=5000):
    """Fetch content from a URL and extract relevant text."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # soup = BeautifulSoup(response.content, 'html.parser')
        
        # # Remove script and style elements
        # for script in soup(["script", "style"]):
        #     script.decompose()
        
        # # Extract text content
        # text = soup.get_text()
        
        # # Clean up whitespace
        # lines = (line.strip() for line in text.splitlines())
        # chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # text = ' '.join(chunk for chunk in chunks if chunk)
        
        # # Limit length to avoid token limits
        # if len(text) > max_length:
        #     text = text[:max_length] + "..."
        
        return response.text # Return raw text content
    except Exception as e:
        print(f"Failed to fetch content from {url}: {e}")
        return ""

def extract_employee_data_from_urls(urls, company_name):
    """Extract employee-related content from URLs."""
    employee_content = []
    
    for url in urls[:5]:  # Limit to first 5 URLs to avoid token limits
        content = fetch_url_content(url)
        if content:
            # Look for employee-related keywords in the content
            employee_keywords = ['executive', 'director', 'manager', 'president', 'ceo', 'cfo', 'cto', 'vice president', 'employee', 'staff', 'team']
            if any(keyword in content.lower() for keyword in employee_keywords):
                employee_content.append(f"Content from {url}:\n{content}\n")
    
    return "\n".join(employee_content)

# Comprehensive prompt for employee data extraction
STRICT_CSV_PROMPT = (
    "Find and list employees for the company '{company}'. "
    "YOUR PRIMARY GOAL: Extract AS MANY EMPLOYEES AS POSSIBLE from this company. "
    "IMPORTANT: You must perform MULTIPLE comprehensive web searches to find different types of employees. "
    "Search for ALL types of employees including: "
    "- Executives (CEO, CFO, CTO, Directors, Auditors, Executive Officers) "
    "- Senior Management (VPs, Senior Directors, Regional Heads) "
    "- Middle Management (Managers, Team Leads, Department Heads) "
    "- General Employees (Engineers, Analysts, Specialists, Coordinators, etc.) "
    "- Support Staff (HR, IT, Marketing, Sales, Operations, etc.) "
    "- Technical Staff (Developers, Designers, Researchers, etc.) "
    "- Administrative Staff (Assistants, Coordinators, etc.) "
    "- Contractors and Consultants "
    "- Board Members and Advisors "
    "- Recent Hires and New Employees "
    "SEARCH STRATEGY - Perform these searches in order: "
    "1. Search for executives and leadership team "
    "2. Search for middle management and department heads "
    "3. Search for technical staff and engineers "
    "4. Search for support staff and administrative roles "
    "5. Search for all employees and staff directory "
    "6. Search for recent hires and new employees "
    "7. Search for company team pages and about us sections "
    "8. Search for LinkedIn profiles and professional networks "
    "Search multiple reliable sources such as official company websites (e.g., 'About Us', 'Leadership', 'Team', 'Our People', 'Careers', 'Meet the Team' pages), "
    "press releases, reputable business news articles, and professional networking sites (if accessible via search). "
    "List as many relevant employees as you can find, including those from international offices or subsidiary companies part of the group. "
    "AIM TO FIND 50+ EMPLOYEES if available. "
    "For each employee, extract their department and precise job title if available. "
    "Provide names in both native script (if found) and Roman alphabet. "
    "After collecting comprehensive data from multiple searches, carefully deduplicate entries based on their Romanized names (lowercase). "
    "Determine the most likely primary email domain for employees of this company. If regional domains exist (e.g., @us.company.com, @uk.company.com), prioritize the global domain if unsure, or list regional domains if clearly associated with a specific employee location found in the data. "
    "Finally, format the complete, deduplicated list as a **strict CSV** table with the following Japanese columns, ensuring data accuracy and consistency:\n"
    "会社名, 部署, 役職, 姓, 名, 姓（小文字ローマ字）, 名（小文字ローマ字）, メールアドレスに使用される可能性が高いドメイン\n"
    "**IMPORTANT:** Ensure that no commas are included *within* any of the data fields (e.g., Department or Job title). If a department or job title contains a comma in the source, remove or replace the comma (e.g., with a semicolon or space) in the output CSV field.\n"
    "If a field is missing, leave the corresponding cell empty. "
    "If a job title is not clearly available from the source, use the department name or the best possible approximation. Ensure all Romanized names are in lowercase. "
    "**CRITICAL:** Output ONLY the CSV data within a markdown code block (```csv ... ```), with the header row first, then data rows. No extra text outside the code block."
)

def extract_code_blocks(text):
    """Extracts content within triple backticks (code blocks)."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)

def parse_csv_from_ai_output(raw_ai_output):
    """Robust parsing that always returns a DataFrame, even with malformed data."""
    logger.info("Attempting to parse AI output as CSV...")
    
    if not raw_ai_output:
        logger.warning("No raw output provided for CSV parsing.")
        return pd.DataFrame()
    
    blocks = extract_code_blocks(raw_ai_output)
    if not blocks:
        logger.warning("No code blocks found in AI response.")
        return pd.DataFrame()
    
    # Assume the last code block contains the final CSV
    csv_text = blocks[-1].strip()
    logger.info(f"Code block found. Length: {len(csv_text)}")
    
    # Use expected columns from config - support both English and Japanese
    expected_columns = config.expected_columns
    expected_columns_japanese = config.expected_columns_japanese
    
    # Use robust CSV parsing with error handling
    data = []
    try:
        # Clean up the CSV text - remove any extra text before/after the code block
        csv_text = csv_text.strip()
        
        # Use io.StringIO to read the string content
        csvfile = io.StringIO(csv_text)
        
        # Try to sniff the delimiter
        try:
            sample_size = min(len(csv_text), 4096)
            dialect = csv.Sniffer().sniff(csv_text[:sample_size])
            reader = csv.reader(csvfile, dialect)
            logger.info(f"Detected CSV dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
        except Exception as sniff_e:
            logger.warning("CSV Sniffer failed, falling back to comma delimiter.")
            csvfile.seek(0)  # Reset file pointer
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        
        # Read the header row
        try:
            header = next(reader)
            data.append(header)
            logger.info(f"Found header: {header}")
        except StopIteration:
            logger.warning("CSV text is empty or only contains header.")
            return pd.DataFrame()
        
        # Read the data rows
        row_count = 0
        for row in reader:
            data.append(row)
            row_count += 1
            # Log warning if column count is unexpected
            if len(row) != len(header):
                logger.warning(f"Unexpected column count in row {row_count}. Expected {len(header)}, found {len(row)}")
        
        logger.info(f"Read {row_count} data rows")
        
        # Convert to DataFrame
        if len(data) > 1:
            df = pd.DataFrame(data[1:], columns=data[0])
            logger.info(f"Successfully parsed CSV data. Found {len(df)} employee records.")
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            
            # Check if expected columns are present (support both English and Japanese)
            found_english_columns = all(col.strip() in df.columns for col in expected_columns)
            found_japanese_columns = all(col.strip() in df.columns for col in expected_columns_japanese)
            
            if not found_english_columns and not found_japanese_columns:
                logger.warning("Parsed CSV is missing some expected columns.")
                logger.warning(f"Expected English: {expected_columns}")
                logger.warning(f"Expected Japanese: {expected_columns_japanese}")
                logger.warning(f"Found: {list(df.columns)}")
            
            # Clean up domain column - remove @ symbols (support both languages)
            domain_columns = ['Domain likely to be used in email addresses', 'メールアドレスに使用される可能性が高いドメイン']
            for domain_col in domain_columns:
                if domain_col in df.columns:
                    df[domain_col] = df[domain_col].str.replace('@', '')
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Skip rows where both first and last names are empty (support both languages)
            name_columns = [
                ('First name', 'Last name'),
                ('名', '姓')
            ]
            for first_name_col, last_name_col in name_columns:
                if first_name_col in df.columns and last_name_col in df.columns:
                    df = df.dropna(subset=[first_name_col, last_name_col], how='all')
                    break
            
            return df
        else:
            logger.warning("Parsed CSV data is empty or only contains header.")
            return pd.DataFrame(columns=header if header else expected_columns)
    
    except Exception as e:
        logger.error(f"Failed to parse CSV data: {e}")
        logger.error(f"Raw CSV content for inspection: {csv_text}")
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'], strict_slashes=False)
def generate():
    data = request.get_json()
    company_name = data.get('company_name', '').strip()
    company_website = data.get('company_website', '').strip()
    if not company_name:
        return jsonify({'error': 'Company name is required.'}), 400
    try:
        logger.info(f"Starting employee research for company: {company_name}")
        
        # Extract domain from website if provided
        domain = None
        if company_website:
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(company_website)
                domain = parsed_url.netloc
                if domain.startswith('www.'):
                    domain = domain[4:]
                logger.info(f"Extracted domain: {domain}")
            except Exception as e:
                logger.error(f"Failed to parse website URL: {e}")
        
        # Tavily search - Enhanced to find both executives and general employees
        queries = []
        
        if domain:
            # When website is provided, prioritize domain-specific searches
            queries = [
                f"executive team leadership site:{domain}",
                f"employees staff directory site:{domain}",
                f"team members about us site:{domain}",
                f"management team site:{domain}",
                f"company directory employees site:{domain}",
                f"about us team site:{domain}",
                f"leadership team site:{domain}",
                f"management site:{domain}",
                f"our people site:{domain}",
                f"team site:{domain}",
                f"leadership site:{domain}",
                f"executives site:{domain}",
                f"board of directors site:{domain}",
                f"organization chart site:{domain}",
                # Add company name only for domain-specific searches
                f"{company_name} site:{domain}",
                f"{company_name} executive site:{domain}",
                f"{company_name} team site:{domain}",
                f"{company_name} management site:{domain}"
            ]
            logger.info(f"Using domain-specific searches for {domain}")
        else:
            # Fallback to company name searches when no website provided
            queries = [
                # English queries - comprehensive
                f"{company_name} executive team leadership",
                f"{company_name} employees staff directory",
                f"{company_name} team members organization",
                f"{company_name} management positions",
                f"{company_name} company directory personnel",
                f"{company_name} board of directors",
                f"{company_name} employee list",
                f"{company_name} organization chart",
                # Fallback to common domains
                f"{company_name} executive team leadership site:{company_name}.com",
                f"{company_name} employees staff directory site:{company_name}.com",
                f"{company_name} team members about us site:{company_name}.com",
                f"{company_name} management team site:{company_name}.com",
                f"{company_name} company directory employees site:{company_name}.com",
                f"{company_name} executive team leadership site:{company_name}.co.jp",
                f"{company_name} employees staff directory site:{company_name}.co.jp",
                f"{company_name} team members about us site:{company_name}.co.jp",
                f"{company_name} management team site:{company_name}.co.jp",
                f"{company_name} company directory employees site:{company_name}.co.jp"
            ]
            logger.info(f"Using general company name searches for {company_name}")
        
        # Search for URLs
        all_urls = []
        all_results = []
        search_progress = []
        
        logger.info(f"Starting search with {len(queries)} queries")
        for i, query in enumerate(queries):
            try:
                logger.info(f"Searching query {i+1}/{len(queries)}: {query}")
                tavily_res = tavily.search(
                    query=query,
                    search_depth=config.tavily_search_depth,
                    max_results=config.tavily_max_results
                )
                urls = [item['url'] for item in tavily_res.get('results', [])]
                all_urls.extend(urls)
                
                # Log search progress
                search_progress.append({
                    'query': query,
                    'urls_found': len(urls),
                    'urls': urls[:3]  # Show first 3 URLs
                })
                
                # Also collect the search result content for AI analysis
                for result in tavily_res.get('results', []):
                    all_results.append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': result.get('url', '')
                    })
                logger.info(f"Found {len(urls)} URLs for query: {query}")
            except Exception as e:
                logger.error(f"Search query failed for '{query}': {e}")
                continue
        
        # Remove duplicates and limit to top results
        urls = list(dict.fromkeys(all_urls))[:config.max_urls_per_company]
        logger.info(f"Found {len(urls)} unique URLs for {company_name}: {urls}")
        
        if not urls:
            logger.warning("No relevant pages found.")
            return jsonify({'error': 'No relevant pages found.'}), 404
        
        # Prepare search results content for AI
        search_content = ""
        for i, result in enumerate(all_results[:config.max_search_results]):
            search_content += f"\nResult {i+1}:\nTitle: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:500]}...\n"
        
        # Add website information to prompt if available
        website_info = ""
        if company_website:
            website_info = f"\nCompany Website: {company_website}\nDomain: {domain if domain else 'Not specified'}"
        
        # OpenAI prompt with function calling approach
        system_message = (
            "You are a highly skilled research assistant AI specializing in extracting employee data from web search results. "
            "IMPORTANT: Focus on finding a DIVERSE mix of employees, not just executives. "
            "Use the web_search tool to find comprehensive information about ALL types of employees including: "
            "- Executives and Senior Management "
            "- Middle Management and Team Leads "
            "- General Employees and Technical Staff "
            "- Support Staff and Administrative Staff "
            "Search for multiple sources including company websites, press releases, and professional networking sites. "
            "After gathering data from multiple searches, format the complete, deduplicated list as a **strict CSV** table "
            "with the specified columns within a markdown code block (```csv ... ```), including the header row, with no extra text."
        )
        
        user_message = STRICT_CSV_PROMPT.format(company=company_name)
        if company_website:
            user_message += f"\n\nCompany Website: {company_website}"
            if domain:
                user_message += f"\nDomain: {domain}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        logger.info("Sending request to OpenAI with function calling")
        csv_content = make_api_request_with_tools(messages)
        if not csv_content:
            logger.error("Failed to get response from AI after retries.")
            return jsonify({'error': 'Failed to get response from AI after retries.'}), 500
        
        logger.info("Received response from OpenAI, parsing CSV")
        print("RAW AI OUTPUT:", csv_content)  # Debug logging
        df = parse_csv_from_ai_output(csv_content)
        print(f"Parsed DataFrame shape: {df.shape}")  # Debug logging
        print(f"DataFrame columns: {list(df.columns)}")  # Debug logging
        
        # Save the raw AI response as a .csv file for download
        raw_filename = f"{company_name}_raw_{uuid.uuid4().hex[:8]}.csv"
        raw_filepath = os.path.join(GENERATED_DIR, raw_filename)
        try:
            with open(raw_filepath, 'w', encoding='utf-8-sig') as f:
                f.write(csv_content)
        except Exception as e:
            logger.error(f"Failed to save raw file {raw_filename}: {e}")
            raw_filename = None
        
        # Return AI response and parsing results
        result = {
            'ai_response': csv_content,
            'parsed_successfully': not df.empty,
            'rows_found': len(df) if not df.empty else 0,
            'search_progress': search_progress,
            'total_urls_found': len(urls),
            'unique_urls': urls,
        }
        
        # Add download URLs only if files were successfully saved
        if raw_filename:
            result['raw_download_url'] = f'/download/{raw_filename}'
        
        if not df.empty:
            # Save parsed CSV
            filename = f"{company_name}_{uuid.uuid4().hex[:8]}.csv"
            filepath = os.path.join(GENERATED_DIR, filename)
            try:
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                result['download_url'] = f'/download/{filename}'
                logger.info(f"Successfully processed {company_name}: {len(df)} employees found")
            except Exception as e:
                logger.error(f"Failed to save parsed CSV {filename}: {e}")
        else:
            logger.warning(f"No employee data parsed for {company_name}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing {company_name}: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/upload', methods=['POST'], strict_slashes=False)
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    try:
        # Read CSV file into DataFrame
        stream = io.StringIO(file.stream.read().decode('utf-8'))
        reader = csv.reader(stream)
        companies_data = []
        for row in reader:
            if row and row[0].strip():
                company_name = row[0].strip()
                company_website = row[1].strip() if len(row) > 1 else ""
                companies_data.append({
                    'name': company_name,
                    'website': company_website
                })
        
        if not companies_data:
            return jsonify({'error': 'No company names found in the CSV.'}), 400
        if len(companies_data) > 100:
            return jsonify({'error': 'CSV contains more than 100 companies. Please split into smaller batches.'}), 400
        
        # Store companies in session or temporary storage
        session_id = str(uuid.uuid4())
        # For simplicity, we'll process immediately and stream results
        # In a production app, you'd store this in Redis or similar
        
        def generate():
            # Send initial heartbeat
            yield f"data: {json.dumps({'status': 'started', 'total_companies': len(companies_data)})}\n\n"
            
            for i, company_data in enumerate(companies_data):
                company_name = company_data['name']
                company_website = company_data['website']
                
                # Send progress update
                yield f"data: {json.dumps({'status': 'processing', 'company': company_name, 'progress': f'{i+1}/{len(companies_data)}'})}\n\n"
                
                try:
                    logger.info(f"Processing company {i+1}/{len(companies_data)}: {company_name}")
                    
                    # Extract domain from website if provided
                    domain = None
                    if company_website:
                        try:
                            from urllib.parse import urlparse
                            parsed_url = urlparse(company_website)
                            domain = parsed_url.netloc
                            if domain.startswith('www.'):
                                domain = domain[4:]
                        except Exception as e:
                            logger.error(f"Failed to parse website URL: {e}")
                    
                    # Comprehensive search queries - full set for maximum results
                    queries = []
                    
                    if domain:
                        # When website is provided, prioritize domain-specific searches
                        queries = [
                            f"executive team leadership site:{domain}",
                            f"employees staff directory site:{domain}",
                            f"team members about us site:{domain}",
                            f"management team site:{domain}",
                            f"company directory employees site:{domain}",
                            f"about us team site:{domain}",
                            f"leadership team site:{domain}",
                            f"management site:{domain}",
                            f"our people site:{domain}",
                            f"team site:{domain}",
                            f"leadership site:{domain}",
                            f"executives site:{domain}",
                            f"board of directors site:{domain}",
                            f"organization chart site:{domain}",
                            # Add company name only for domain-specific searches
                            f"{company_name} site:{domain}",
                            f"{company_name} executive site:{domain}",
                            f"{company_name} team site:{domain}",
                            f"{company_name} management site:{domain}"
                        ]
                    else:
                        # Fallback to company name searches when no website provided
                        queries = [
                            # English queries - comprehensive
                            f"{company_name} executive team leadership",
                            f"{company_name} employees staff directory",
                            f"{company_name} team members organization",
                            f"{company_name} management positions",
                            f"{company_name} company directory personnel",
                            f"{company_name} board of directors",
                            f"{company_name} employee list",
                            f"{company_name} organization chart",
                            # Fallback to common domains
                            f"{company_name} executive team leadership site:{company_name}.com",
                            f"{company_name} employees staff directory site:{company_name}.com",
                            f"{company_name} team members about us site:{company_name}.com",
                            f"{company_name} management team site:{company_name}.com",
                            f"{company_name} company directory employees site:{company_name}.com",
                            # Additional company domains
                            f"{company_name} executive team leadership site:{company_name}.co.jp",
                            f"{company_name} employees staff directory site:{company_name}.co.jp",
                            f"{company_name} team members about us site:{company_name}.co.jp",
                            f"{company_name} management team site:{company_name}.co.jp",
                            f"{company_name} company directory employees site:{company_name}.co.jp"
                        ]
                    
                    all_urls = []
                    all_results = []
                    search_progress = []
                    
                    # Process search queries with timeout and rate limiting
                    for query_idx, query in enumerate(queries):
                        try:
                            logger.info(f"Searching query {query_idx+1}/{len(queries)} for {company_name}: {query}")
                            
                            tavily_res = tavily.search(
                                query=query,
                                search_depth=config.tavily_search_depth,
                                max_results=10  # Increased from 5 to 10 results per query
                            )
                            urls = [item['url'] for item in tavily_res.get('results', [])]
                            all_urls.extend(urls)
                            
                            # Log search progress
                            search_progress.append({
                                'query': query,
                                'urls_found': len(urls),
                                'urls': urls[:3]  # Show first 3 URLs
                            })
                            
                            # Also collect the search result content for AI analysis
                            for result in tavily_res.get('results', []):
                                all_results.append({
                                    'title': result.get('title', ''),
                                    'content': result.get('content', ''),
                                    'url': result.get('url', '')
                                })
                            
                            # Add small delay between searches to avoid rate limiting
                            time.sleep(0.5)
                            
                        except Exception as e:
                            logger.error(f"Search query failed for '{query}': {e}")
                            continue
                    
                    urls = list(dict.fromkeys(all_urls))[:20]  # Increased from 8 to 20 URLs
                    logger.info(f"Found {len(urls)} unique URLs for {company_name}: {urls[:5]}...")  # Log first 5 URLs
                    if not urls:
                        result = {
                            'company': company_name, 
                            'error': 'No relevant pages found.',
                            'search_progress': search_progress
                        }
                        yield f"data: {json.dumps(result)}\n\n"
                        continue
                    
                    # Prepare search results content for AI
                    search_content = ""
                    for i, result in enumerate(all_results[:50]):  # Increased to 50 results
                        search_content += f"\nResult {i+1}:\nTitle: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:1000]}...\n"
                    
                    # Add website information to prompt if available
                    website_info = ""
                    if company_website:
                        website_info = f"\nCompany Website: {company_website}\nDomain: {domain if domain else 'Not specified'}"
                    
                    # OpenAI prompt with function calling approach
                    system_message = (
                        "You are a highly skilled research assistant AI specializing in extracting MAXIMUM employee data from web search results. "
                        "YOUR PRIMARY GOAL: Find and extract AS MANY EMPLOYEES AS POSSIBLE from the company. "
                        "IMPORTANT: You must perform MULTIPLE comprehensive web searches to find different types of employees. "
                        "Use the web_search tool to find comprehensive information about ALL types of employees including: "
                        "- Executives and Senior Management (CEO, CFO, CTO, Directors, VPs, etc.) "
                        "- Middle Management and Team Leads (Managers, Supervisors, etc.) "
                        "- General Employees and Technical Staff (Engineers, Developers, Analysts, etc.) "
                        "- Support Staff and Administrative Staff (HR, IT, Marketing, Sales, Operations, etc.) "
                        "- Entry-level and Junior Staff "
                        "- Contractors and Consultants "
                        "- Board Members and Advisors "
                        "SEARCH STRATEGY: "
                        "1. Search for executives and leadership team "
                        "2. Search for middle management and department heads "
                        "3. Search for technical staff and engineers "
                        "4. Search for support staff and administrative roles "
                        "5. Search for all employees and staff directory "
                        "6. Search for recent hires and new employees "
                        "7. Search for company team pages and about us sections "
                        "8. Search for LinkedIn profiles and professional networks "
                        "Be EXTREMELY thorough and extract as many employees as possible from each source. "
                        "Aim to find 50+ employees per company if available. "
                        "MINIMUM TARGET: Find at least 20-30 employees per company. "
                        "If you find fewer than 15 employees, continue searching for more sources. "
                        "After gathering data from multiple searches, format the complete, deduplicated list as a **strict CSV** table "
                        "with the specified columns within a markdown code block (```csv ... ```), including the header row, with no extra text."
                    )
                    
                    user_message = STRICT_CSV_PROMPT.format(company=company_name)
                    if company_website:
                        user_message += f"\n\nCompany Website: {company_website}"
                        if domain:
                            user_message += f"\nDomain: {domain}"
                    
                    # Add the search results to the user message so AI can analyze them
                    user_message += f"\n\nSEARCH RESULTS FOUND:\n{search_content}\n\n"
                    user_message += f"URLS FOUND: {urls}\n\n"
                    user_message += "IMPORTANT: Analyze the search results above and extract employee information from them. "
                    user_message += "Then perform additional web searches to find more employees. "
                    user_message += "Your goal is to find at least 20-30 employees from this company. "
                    user_message += "CRITICAL: Look at the URLs found and extract employee names, titles, and departments from the search result content. "
                    user_message += "If the search results contain employee information, extract it immediately. "
                    user_message += "Then use web_search to find additional employee data from those URLs and other sources. "
                    user_message += "SPECIFIC INSTRUCTIONS: "
                    user_message += "1. First, extract any employee names from the search results above "
                    user_message += "2. Then use web_search to search each URL found for employee information "
                    user_message += "3. Search for 'employees', 'team', 'staff', 'leadership' on each URL "
                    user_message += "4. Continue searching until you find at least 20-30 employees "
                    user_message += "5. If you find fewer than 15 employees, search more sources"
                    
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                    
                    logger.info(f"Sending request to OpenAI with function calling for {company_name}")
                    
                    # Simplified approach with 5-minute timeout
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    def api_call():
                        try:
                            result = make_api_request_with_tools(messages)
                            result_queue.put(('success', result))
                        except Exception as e:
                            result_queue.put(('error', str(e)))
                    
                    # Start API call in separate thread
                    api_thread = threading.Thread(target=api_call)
                    api_thread.daemon = True
                    api_thread.start()
                    
                    # Wait for result with timeout (5 minutes = 300 seconds)
                    try:
                        result_type, result_data = result_queue.get(timeout=300)
                        if result_type == 'success':
                            csv_content = result_data
                        else:
                            raise Exception(result_data)
                    except queue.Empty:
                        logger.error(f"OpenAI API call timed out for {company_name} after 5 minutes")
                        result = {'company': company_name, 'error': 'OpenAI API call timed out after 5 minutes.'}
                        yield f"data: {json.dumps(result)}\n\n"
                        continue
                    except Exception as e:
                        logger.error(f"Failed to get response from AI for {company_name}: {e}")
                        result = {'company': company_name, 'error': f'Failed to get response from AI: {str(e)}'}
                        yield f"data: {json.dumps(result)}\n\n"
                        continue
                    
                    if not csv_content:
                        logger.error(f"Failed to get response from AI after retries for {company_name}")
                        result = {'company': company_name, 'error': 'Failed to get response from AI after retries.'}
                        yield f"data: {json.dumps(result)}\n\n"
                        continue
                    
                    logger.info(f"Received response from OpenAI, parsing CSV for {company_name}")
                    print(f"RAW AI OUTPUT for {company_name}:", csv_content)  # Debug logging
                    df = parse_csv_from_ai_output(csv_content)
                    print(f"Parsed DataFrame shape for {company_name}: {df.shape}")  # Debug logging
                    print(f"DataFrame columns for {company_name}: {list(df.columns)}")  # Debug logging
                    
                    result = {
                        'company': company_name,
                        'ai_response': csv_content,
                        'parsed_successfully': not df.empty,
                        'rows_found': len(df) if not df.empty else 0,
                        'search_progress': search_progress,
                        'total_urls_found': len(urls),
                        'unique_urls': urls
                    }
                    
                    # Debug: Check if result is properly formatted
                    print(f"RESULT STRUCTURE for {company_name}:")
                    print(f"  - Company: {result['company']}")
                    print(f"  - Parsed successfully: {result['parsed_successfully']}")
                    print(f"  - Rows found: {result['rows_found']}")
                    print(f"  - Search progress length: {len(result['search_progress'])}")
                    print(f"  - Total URLs found: {result['total_urls_found']}")
                    
                    if not df.empty:
                        # Save parsed CSV
                        filename = f"{company_name}_{uuid.uuid4().hex[:8]}.csv"
                        filepath = os.path.join(GENERATED_DIR, filename)
                        try:
                            df.to_csv(filepath, index=False, encoding='utf-8-sig')
                            result['download_url'] = f'/download/{filename}'
                            logger.info(f"Successfully processed {company_name}: {len(df)} employees found")
                            
                        except Exception as e:
                            logger.error(f"Failed to save parsed CSV {filename}: {e}")
                    else:
                        logger.warning(f"No employee data parsed for {company_name}")
                        # Save the raw AI response as a .csv file for download
                        raw_filename = f"{company_name}_raw_{uuid.uuid4().hex[:8]}.csv"
                        raw_filepath = os.path.join(GENERATED_DIR, raw_filename)
                        try:
                            with open(raw_filepath, 'w', encoding='utf-8-sig') as f:
                                f.write(csv_content)
                            result['raw_download_url'] = f'/download/{raw_filename}'
                        except Exception as e:
                            logger.error(f"Failed to save raw file {raw_filename}: {e}")
                    
                    yield f"data: {json.dumps(result)}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error processing {company_name}: {e}")
                    result = {
                        'company': company_name,
                        'error': str(e),
                        'ai_response': 'Error occurred before AI processing could complete.'
                    }
                    logger.info(f"Sending result for {company_name}: {len(result)} fields")
                    print(f"SENDING RESULT: {json.dumps(result, ensure_ascii=False)}")
                    yield f"data: {json.dumps(result)}\n\n"
            
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control',
            'X-Accel-Buffering': 'no',
            'Transfer-Encoding': 'chunked'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


# For Vercel deployment
app.debug = False

if __name__ == '__main__':
    app.run(debug=True) 