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
    tavily_max_results: int = 10
    tavily_search_depth: str = "advanced"
    
    # Processing settings
    max_urls_per_company: int = 20
    max_search_results: int = 50
    
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
        
        # Adjust file paths for Vercel and Railway environment
        # Railway doesn't always set RAILWAY env var, so check for PORT and other indicators
        is_railway = bool(os.environ.get('RAILWAY')) or bool(os.environ.get('PORT')) or 'railway' in os.environ.get('HOSTNAME', '').lower()
        if os.environ.get('VERCEL') or is_railway:
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

# Initialize clients with error handling
try:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")

# Initialize Tavily client with error handling
tavily = None

# Check what version of tavily is installed
try:
    import tavily
    logger.info(f"Tavily package version: {getattr(tavily, '__version__', 'unknown')}")
    logger.info(f"Tavily package path: {tavily.__file__}")
except ImportError:
    logger.error("Tavily package not found")
except Exception as e:
    logger.error(f"Error checking tavily package: {e}")

try:
    logger.info(f"Attempting to initialize TavilyClient with API key: {TAVILY_API_KEY[:10]}...")
    tavily = TavilyClient(TAVILY_API_KEY)
    logger.info("Tavily client initialized successfully with TavilyClient")
except Exception as e:
    logger.error(f"Failed to initialize Tavily client with TavilyClient: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error details: {str(e)}")
    
    # Try alternative initialization method
    try:
        logger.info("Attempting to import Client from tavily")
        from tavily import Client
        logger.info("Successfully imported Client from tavily")
        tavily = Client(api_key=TAVILY_API_KEY)
        logger.info("Tavily client initialized successfully with Client")
    except Exception as e2:
        logger.error(f"Failed to initialize Tavily client with Client: {e2}")
        logger.error(f"Error type: {type(e2).__name__}")
        logger.error(f"Error details: {str(e2)}")
        
        # Try with minimal parameters
        try:
            logger.info("Attempting TavilyClient with minimal parameters")
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            logger.info("Tavily client initialized with minimal parameters")
        except Exception as e3:
            logger.error(f"Failed to initialize Tavily client with minimal parameters: {e3}")
            logger.error(f"Error type: {type(e3).__name__}")
            logger.error(f"Error details: {str(e3)}")
            
            # Create a dummy client that returns empty results
            class DummyTavilyClient:
                def search(self, **kwargs):
                    logger.warning("Using dummy Tavily client - no search results will be returned")
                    return {'results': []}
            tavily = DummyTavilyClient()
            logger.warning("Using dummy Tavily client due to initialization failure")

app = Flask(__name__)

# For Railway deployment, use temporary storage
# Railway doesn't always set RAILWAY env var, so check for PORT and other indicators
is_railway = bool(os.environ.get('RAILWAY')) or bool(os.environ.get('PORT')) or 'railway' in os.environ.get('HOSTNAME', '').lower()
if os.environ.get('VERCEL') or is_railway:
    # In Vercel/Railway environment, use /tmp for temporary files
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
                            
                            # Execute Tavily search with error handling
                            try:
                                search_results = tavily.search(
                                    query=query,
                                    search_depth=config.tavily_search_depth,
                                    max_results=config.tavily_max_results
                                ).get('results', [])
                            except Exception as search_error:
                                logger.error(f"Tavily search failed: {search_error}")
                                search_results = []
                            
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



# Step 1: Data gathering prompt
GATHER_DATA_PROMPT = (
    "Analyze the provided search results and gather comprehensive employee data for Japanese company '{company}' (website: {company_website}). "
    "STRATEGY: "
    "1. First, extract all employee information from the provided search results and URLs. "
    "2. Identify what types of employees are missing (executives, managers, technical staff, etc.). "
    "3. Use web_search to find missing employee types, focusing on gaps identified. "
    "4. Prioritize searches for: CEOs, Presidents, CTOs, CFOs, Directors, Managers, Engineers, Sales staff. "
    "5. Target sources: LinkedIn profiles, company websites, press releases, executive directories. "
    "OUTPUT: List all found employees with names, titles, departments. Format as simple list."
)

# Step 2: Data processing prompt
PROCESS_DATA_PROMPT = (
    "Process the gathered employee data into structured CSV format. "
    "GOAL: Create clean, deduplicated CSV with 50+ employees. "
    "QUALITY: Complete names preferred, no duplicates, include all employee types. "
    "OUTPUT FORMAT - CRITICAL: "
    "Output ONLY a CSV table with these exact columns: "
    "会社名, 部署, 役職, 姓, 名, 姓（小文字ローマ字）, 名（小文字ローマ字）, メールアドレスに使用される可能性が高いドメイン "
    "Format: ```csv [header row] [data rows] ``` - NO other text outside code block. "
    "IMPORTANT: Use double quotes around all fields to handle commas in job titles. "
    "CRITICAL: Replace commas with underscores (_) in job titles and other fields. "
    "Example: \"Company Name\",\"Department\",\"Job Title_with_commas\",\"Last Name\",\"First Name\",\"lastname\",\"firstname\",\"domain.com\" "
    "Extract: Company name, Department, Job title, Surname (kanji/kana), First name (kanji/kana), "
    "Romanized English surname (lowercase), Romanized English first name (lowercase), Email domain."
)

def extract_code_blocks(text):
    """Extracts content within triple backticks (code blocks)."""
    pattern = r"```(?:\w+)?\n(.*?)```"
    return re.findall(pattern, text, re.DOTALL)

def generate_search_queries(company_name: str, company_website: str = "", domain: str = None) -> list:
    """Generate 10 comprehensive search queries for employee discovery."""
    
    # Basic executive and management searches (most important)
    queries = [
        # 1. Basic Executive Search - CEO, President, CTO, etc.
        f'"{company_name}" (CEO OR President OR CTO OR CFO OR COO OR "Chief Executive" OR "Chief Technology" OR "Chief Financial")',
        
        # 2. Japanese Executive Search
        f'"{company_name}" (社長 OR 代表取締役 OR 取締役 OR 副社長 OR 専務 OR 常務)',
        
        # 3. Management Team Search
        f'"{company_name}" (Director OR Manager OR VP OR "Vice President" OR 部長 OR 課長 OR マネージャー)',
        
        # 4. LinkedIn Employee Search
        f'site:linkedin.com/in "{company_name}" "Japan"',
        
        # 5. Company Website Employee Search
        f'"{company_name}" (employee OR staff OR team OR 社員 OR 従業員)',
        
        # 6. Executive Team and Leadership
        f'"{company_name}" (executive OR leadership OR management OR 役員 OR 幹部)',
        
        # 7. Board of Directors
        f'"{company_name}" (board OR director OR 取締役会 OR 監査役)',
        
        # 8. Technical Staff
        f'"{company_name}" (engineer OR developer OR エンジニア OR 技術者)',
        
        # 9. Sales and Business Staff
        f'"{company_name}" (sales OR business OR 営業 OR ビジネス)',
        
        # 10. Company About/Team Pages
        f'"{company_name}" (about OR team OR company OR 会社 OR 企業)'
    ]
    
    # Add domain-specific searches if available
    if domain:
        domain_queries = [
            # Domain-specific executive searches
            f'"{company_name}" (CEO OR President OR 社長 OR 代表取締役) site:{domain}',
            f'"{company_name}" (executive OR management OR 役員) site:{domain}',
            f'"{company_name}" (team OR staff OR 社員) site:{domain}',
            f'"{company_name}" (about OR company OR 会社) site:{domain}',
            f'"{company_name}" (leadership OR 幹部) site:{domain}'
        ]
        # Replace last 5 queries with domain-specific ones
        queries = queries[:5] + domain_queries
    
    logger.info(f"Generated {len(queries)} comprehensive search queries for {company_name}")
    return queries

def extract_domain_from_website(company_website: str) -> str:
    """Extract domain from company website URL."""
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
    return domain

def create_system_message_step1() -> str:
    """Create system message for Step 1: Data gathering."""
    return (
        "You are an analytical research assistant gathering employee data. "
        "First analyze the provided search results to extract employee information. "
        "Then identify gaps in employee coverage (missing executives, managers, technical staff, etc.). "
        "Use web_search tool strategically to fill these gaps. "
        "Focus on finding CEOs, Presidents, CTOs, CFOs, Directors, Managers, and Engineers. "
        "Output comprehensive employee data in simple list format."
    )

def create_system_message_step2() -> str:
    """Create system message for Step 2: Data processing."""
    return (
        "Process employee data into structured CSV format. "
        "Clean, deduplicate, and format the gathered employee information. "
        "Output ONLY CSV table in markdown code block (```csv ... ```). "
        "No extra text outside code block."
    )

def save_csv_file(df: pd.DataFrame, company_name: str, suffix: str = "") -> str:
    """Save DataFrame as CSV file and return download URL."""
    filename = f"{company_name}_{uuid.uuid4().hex[:8]}{suffix}.csv"
    filepath = os.path.join(GENERATED_DIR, filename)
    try:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        download_url = f'/download/{filename}'
        logger.info(f"Successfully saved CSV file: {filename}")
        return download_url
    except Exception as e:
        logger.error(f"Failed to save CSV file {filename}: {e}")
        return None

def save_raw_file(content: str, company_name: str, suffix: str = "_raw") -> str:
    """Save raw content as file and return download URL."""
    filename = f"{company_name}_{uuid.uuid4().hex[:8]}{suffix}.csv"
    filepath = os.path.join(GENERATED_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            f.write(content)
        download_url = f'/download/{filename}'
        logger.info(f"Successfully saved raw file: {filename}")
        return download_url
    except Exception as e:
        logger.error(f"Failed to save raw file {filename}: {e}")
        return None

def prepare_search_content(all_results: list, max_results: int = None, content_length: int = 500) -> str:
    """Prepare search results content for AI analysis."""
    if max_results is None:
        max_results = config.max_search_results
    
    search_content = ""
    for i, result in enumerate(all_results[:max_results]):
        search_content += f"\nResult {i+1}:\nTitle: {result['title']}\nURL: {result['url']}\nContent: {result['content'][:content_length]}...\n"
    
    return search_content

def prepare_website_info(company_website: str, domain: str) -> str:
    """Prepare website information for AI prompt."""
    website_info = ""
    if company_website:
        website_info = f"\nCompany Website: {company_website}\nDomain: {domain if domain else 'Not specified'}"
    return website_info

def log_processing_debug(company_name: str, csv_content: str, df: pd.DataFrame, is_batch: bool = False):
    """Log debug information for processing results."""
    if is_batch:
        print(f"RAW AI OUTPUT for {company_name}:", csv_content)  # Debug logging
        print(f"Parsed DataFrame shape for {company_name}: {df.shape}")  # Debug logging
        print(f"DataFrame columns for {company_name}: {list(df.columns)}")  # Debug logging
    else:
        print("RAW AI OUTPUT:", csv_content)  # Debug logging
        print(f"Parsed DataFrame shape: {df.shape}")  # Debug logging
        print(f"DataFrame columns: {list(df.columns)}")  # Debug logging

def log_processing_start(company_name: str, is_batch: bool = False, batch_index: int = None, total_companies: int = None):
    """Log the start of processing for a company."""
    if is_batch and batch_index is not None and total_companies is not None:
        logger.info(f"Processing company {batch_index+1}/{total_companies}: {company_name}")
    else:
        logger.info(f"Starting employee research for company: {company_name}")

def log_processing_error(company_name: str, error: Exception, is_batch: bool = False):
    """Log processing errors consistently."""
    logger.error(f"Error processing {company_name}: {error}")

def log_urls_found(company_name: str, urls: list, is_batch: bool = False):
    """Log the number of URLs found for a company."""
    if is_batch:
        logger.info(f"Found {len(urls)} unique URLs for {company_name}: {urls[:5]}...")  # Log first 5 URLs
    else:
        logger.info(f"Found {len(urls)} unique URLs for {company_name}: {urls}")

def log_openai_request(company_name: str, is_batch: bool = False):
    """Log OpenAI API request initiation."""
    if is_batch:
        logger.info(f"Sending request to OpenAI with function calling for {company_name}")
    else:
        logger.info("Sending request to OpenAI with function calling")

def log_openai_response(company_name: str, is_batch: bool = False):
    """Log OpenAI API response received."""
    if is_batch:
        logger.info(f"Received response from OpenAI, parsing CSV for {company_name}")
    else:
        logger.info("Received response from OpenAI, parsing CSV")

def handle_no_urls_found(company_name: str, search_progress: list, is_batch: bool = False):
    """Handle the case when no URLs are found for a company."""
    logger.warning("No relevant pages found.")
    if is_batch:
        return {
            'company': company_name, 
            'error': 'No relevant pages found.',
            'search_progress': search_progress
        }
    else:
        return jsonify({'error': 'No relevant pages found.'}), 404

def make_api_request_without_tools(messages, max_retries=None, retry_count=0):
    """Make OpenAI API request without function calling (for Step 2 processing)."""
    if max_retries is None:
        max_retries = config.max_retries
    
    try:
        logger.info(f"Making API request without tools (attempt {retry_count + 1}/{max_retries + 1})")
        
        response = openai.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        if retry_count < max_retries:
            logger.warning(f"API request failed (attempt {retry_count + 1}): {e}")
            time.sleep(config.request_delay * (2 ** retry_count))  # Exponential backoff
            return make_api_request_without_tools(messages, max_retries, retry_count + 1)
        else:
            logger.error(f"API request failed after {max_retries + 1} attempts: {e}")
            return None

def execute_two_step_process(company_name: str, company_website: str, domain: str, 
                           search_content: str, urls: list) -> str:
    """Execute two-step process: gather data, then process into CSV."""
    
    # Step 1: Gather comprehensive employee data (WITH tools)
    logger.info(f"Step 1: Gathering employee data for {company_name}")
    
    system_message_step1 = create_system_message_step1()
    user_message_step1 = GATHER_DATA_PROMPT.format(company=company_name, company_website=company_website)
    user_message_step1 += f"\n\nSEARCH RESULTS:\n{search_content}\n\n"
    user_message_step1 += f"URLS: {urls}\n\n"
    user_message_step1 += "Use web_search to find additional employee information. "
    user_message_step1 += "Collect names, titles, departments from all sources. "
    user_message_step1 += "Output as simple list of employees found."
    
    messages_step1 = [
        {"role": "system", "content": system_message_step1},
        {"role": "user", "content": user_message_step1}
    ]
    
    gathered_data = make_api_request_with_tools(messages_step1)
    if not gathered_data:
        logger.error("Step 1 failed: No data gathered")
        return None
    
    logger.info(f"Step 1 completed: Gathered employee data for {company_name}")
    
    # Step 2: Process gathered data into CSV format (WITHOUT tools)
    logger.info(f"Step 2: Processing data into CSV for {company_name}")
    
    system_message_step2 = create_system_message_step2()
    user_message_step2 = PROCESS_DATA_PROMPT
    user_message_step2 += f"\n\nGATHERED EMPLOYEE DATA:\n{gathered_data}\n\n"
    user_message_step2 += f"COMPANY: {company_name}\n"
    user_message_step2 += f"WEBSITE: {company_website}\n"
    user_message_step2 += "Process this data into structured CSV format with Japanese column headers."
    
    messages_step2 = [
        {"role": "system", "content": system_message_step2},
        {"role": "user", "content": user_message_step2}
    ]
    
    csv_content = make_api_request_without_tools(messages_step2)
    if not csv_content:
        logger.error("Step 2 failed: No CSV generated")
        return None
    
    logger.info(f"Step 2 completed: Generated CSV for {company_name}")
    return csv_content

def create_result_dict(company_name: str, csv_content: str, df: pd.DataFrame, 
                      search_progress: list, urls: list, is_batch: bool = False) -> dict:
    """Create standardized result dictionary for both single and batch processing."""
    result = {
        'ai_response': csv_content,
        'parsed_successfully': not df.empty,
        'rows_found': len(df) if not df.empty else 0,
        'search_progress': search_progress,
        'total_urls_found': len(urls),
        'unique_urls': urls,
    }
    
    # Add company name for batch processing
    if is_batch:
        result['company'] = company_name
    
    # Add download URLs if files were successfully saved
    if not df.empty:
        download_url = save_csv_file(df, company_name)
        if download_url:
            result['download_url'] = download_url
    else:
        # Save raw content if parsing failed
        raw_download_url = save_raw_file(csv_content, company_name)
        if raw_download_url:
            result['raw_download_url'] = raw_download_url
    
    return result

def execute_search_queries(queries: list, company_name: str, max_results: int = None, delay: float = 0.0) -> tuple:
    """Execute search queries and return URLs, results, and progress."""
    if max_results is None:
        max_results = 8  # Reduced from config.tavily_max_results (10) to 8 for token limit
    
    all_urls = []
    all_results = []
    search_progress = []
    
    logger.info(f"Starting search with {len(queries)} queries for {company_name}")
    for i, query in enumerate(queries):
        try:
            logger.info(f"Searching query {i+1}/{len(queries)}: {query}")
            try:
                tavily_res = tavily.search(
                    query=query,
                    search_depth=config.tavily_search_depth,
                    max_results=max_results
                )
                urls = [item['url'] for item in tavily_res.get('results', [])]
                all_urls.extend(urls)
            except Exception as search_error:
                logger.error(f"Tavily search failed for query '{query}': {search_error}")
                urls = []
                tavily_res = {'results': []}
            
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
            
            # Add delay between searches if specified
            if delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Search query failed for '{query}': {e}")
            continue
    
    return all_urls, all_results, search_progress

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
        
        # Try to sniff the delimiter with better error handling
        try:
            sample_size = min(len(csv_text), 4096)
            dialect = csv.Sniffer().sniff(csv_text[:sample_size])
            reader = csv.reader(csvfile, dialect)
            logger.info(f"Detected CSV dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
        except Exception as sniff_e:
            logger.warning("CSV Sniffer failed, falling back to comma delimiter with proper quoting.")
            csvfile.seek(0)  # Reset file pointer
            reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        
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
            
            # Additional data quality filtering
            if not df.empty:
                # Remove rows with generic titles without specific names
                generic_titles = ['administrative assistant', 'intern', 'coordinator', 'assistant']
                if 'Job title' in df.columns:
                    df = df[~df['Job title'].str.lower().isin(generic_titles)]
                if '役職' in df.columns:
                    df = df[~df['役職'].str.lower().isin(generic_titles)]
                
                # Remove duplicate entries based on romanized names
                if 'Last name (lowercase Roman letters)' in df.columns and 'First name (lowercase Roman letters)' in df.columns:
                    df = df.drop_duplicates(subset=['Last name (lowercase Roman letters)', 'First name (lowercase Roman letters)'])
                if '姓（小文字ローマ字）' in df.columns and '名（小文字ローマ字）' in df.columns:
                    df = df.drop_duplicates(subset=['姓（小文字ローマ字）', '名（小文字ローマ字）'])
                
                # Limit to maximum 40 entries
                if len(df) > 40:
                    df = df.head(40)
            
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

@app.route('/api/status')
def api_status():
    """Simple status endpoint for health checks."""
    return jsonify({'status': 'ok', 'message': 'Lead Generation API is running'})

@app.route('/api/packages')
def check_packages():
    """Check installed package versions."""
    import pkg_resources
    import sys
    
    packages = {}
    packages_to_check = ['tavily-python', 'openai', 'flask', 'requests', 'python-dotenv', 'pandas']
    
    for package in packages_to_check:
        try:
            version = pkg_resources.get_distribution(package).version
            packages[package] = version
        except Exception as e:
            packages[package] = f"ERROR: {e}"
    
    return jsonify({
        'python_version': sys.version,
        'packages': packages,
        'environment': {
            'railway': bool(os.environ.get('RAILWAY')),
            'vercel': bool(os.environ.get('VERCEL')),
            'port': os.environ.get('PORT', '5000')
        }
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway."""
    try:
        # Basic health checks
        checks = {
            'app_running': True,
            'openai_key_set': bool(OPENAI_API_KEY),
            'tavily_key_set': bool(TAVILY_API_KEY),
            'environment': {
                'railway': bool(os.environ.get('RAILWAY')),
                'vercel': bool(os.environ.get('VERCEL')),
                'port': os.environ.get('PORT', '5000'),
                'generated_dir': GENERATED_DIR
            }
        }
        
        return jsonify({
            'status': 'healthy', 
            'timestamp': time.time(),
            'checks': checks
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/test-sse')
def test_sse():
    """Test SSE streaming endpoint for debugging."""
    def generate():
        yield f"data: {json.dumps({'message': 'SSE test started'})}\n\n"
        time.sleep(1)
        yield f"data: {json.dumps({'message': 'SSE test message 1'})}\n\n"
        time.sleep(1)
        yield f"data: {json.dumps({'message': 'SSE test message 2'})}\n\n"
        time.sleep(1)
        yield f"data: {json.dumps({'message': 'SSE test completed'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control, Content-Type',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'X-Accel-Buffering': 'no',
        'Transfer-Encoding': 'chunked',
        'Content-Type': 'text/event-stream; charset=utf-8'
    })

@app.route('/generate', methods=['POST'], strict_slashes=False)
def generate():
    data = request.get_json()
    company_name = data.get('company_name', '').strip()
    company_website = data.get('company_website', '').strip()
    if not company_name:
        return jsonify({'error': 'Company name is required.'}), 400
    try:
        log_processing_start(company_name)
        
        # Extract domain from website if provided
        domain = extract_domain_from_website(company_website)
        
        # Generate and execute search queries using unified functions
        queries = generate_search_queries(company_name, company_website, domain)
        all_urls, all_results, search_progress = execute_search_queries(queries, company_name)
        
        # Remove duplicates and limit to top results
        urls = list(dict.fromkeys(all_urls))[:config.max_urls_per_company]
        log_urls_found(company_name, urls)
        
        if not urls:
            return handle_no_urls_found(company_name, search_progress)
        
        # Prepare search results content for AI
        search_content = prepare_search_content(all_results)
        
        # Add website information to prompt if available
        website_info = prepare_website_info(company_website, domain)
        
        # Execute two-step process: gather data, then process into CSV
        log_openai_request(company_name)
        csv_content = execute_two_step_process(company_name, company_website, domain, search_content, urls)
        if not csv_content:
            logger.error("Failed to complete two-step process.")
            return jsonify({'error': 'Failed to complete two-step process.'}), 500
        
        log_openai_response(company_name)
        df = parse_csv_from_ai_output(csv_content)
        log_processing_debug(company_name, csv_content, df)
        
        # Create standardized result dictionary
        result = create_result_dict(company_name, csv_content, df, search_progress, urls)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing {company_name}: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/upload', methods=['POST', 'OPTIONS'], strict_slashes=False)
def upload_csv():
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        return Response('', status=200, headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control, Content-Type',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Max-Age': '86400'
        })
    
    # Handle actual POST request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    
    # Get batch parameters
    batch_size = int(request.form.get('batch_size', 4))  # Default to 4 companies per batch
    start_index = int(request.form.get('start_index', 0))  # Start from beginning by default
    
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
        
        total_companies = len(companies_data)
        
        # Since we're receiving only the current batch, process all companies in the file
        current_batch = companies_data  # Process all companies in the received file
        
        logger.info(f"Processing batch with {len(current_batch)} companies (batch size: {batch_size})")
        
        # Process companies in current batch
        results = []
        
        for i, company_data in enumerate(current_batch):
            company_name = company_data['name']
            company_website = company_data['website']
            
            log_processing_start(company_name, is_batch=True, batch_index=start_index+i, total_companies=total_companies)
            
            try:
                # Extract domain from website if provided
                domain = extract_domain_from_website(company_website)
                
                # Generate and execute search queries using unified functions
                queries = generate_search_queries(company_name, company_website, domain)
                all_urls, all_results, search_progress = execute_search_queries(
                    queries, company_name, delay=0.3
                )
                
                urls = list(dict.fromkeys(all_urls))[:config.max_urls_per_company]
                log_urls_found(company_name, urls, is_batch=True)
                if not urls:
                    result = handle_no_urls_found(company_name, search_progress, is_batch=True)
                    results.append(result)
                    continue
                
                # Prepare search results content for AI
                search_content = prepare_search_content(all_results)
                
                # Add website information to prompt if available
                website_info = prepare_website_info(company_website, domain)
                
                # Execute two-step process: gather data, then process into CSV
                log_openai_request(company_name, is_batch=True)
                
                # Simplified approach with 3-minute timeout for Vercel
                import threading
                import queue
                
                result_queue = queue.Queue()
                def api_call():
                    try:
                        result = execute_two_step_process(company_name, company_website, domain, search_content, urls)
                        result_queue.put(('success', result))
                    except Exception as e:
                        result_queue.put(('error', str(e)))
                
                # Start API call in separate thread
                api_thread = threading.Thread(target=api_call)
                api_thread.daemon = True
                api_thread.start()
                
                # Wait for result with timeout (3 minutes = 180 seconds for Vercel)
                try:
                    result_type, result_data = result_queue.get(timeout=180)
                    if result_type == 'success':
                        csv_content = result_data
                    else:
                        raise Exception(result_data)
                except queue.Empty:
                    logger.error(f"Two-step process timed out for {company_name} after 3 minutes")
                    result = {'company': company_name, 'error': 'Two-step process timed out after 3 minutes.'}
                    results.append(result)
                    continue
                except Exception as e:
                    logger.error(f"Failed to complete two-step process for {company_name}: {e}")
                    result = {'company': company_name, 'error': f'Failed to complete two-step process: {str(e)}'}
                    results.append(result)
                    continue
                
                if not csv_content:
                    logger.error(f"Failed to complete two-step process for {company_name}")
                    result = {'company': company_name, 'error': 'Failed to complete two-step process.'}
                    results.append(result)
                    continue
                
                log_openai_response(company_name, is_batch=True)
                df = parse_csv_from_ai_output(csv_content)
                log_processing_debug(company_name, csv_content, df, is_batch=True)
                
                # Create standardized result dictionary for batch processing
                result = create_result_dict(company_name, csv_content, df, search_progress, urls, is_batch=True)
                
                results.append(result)
                
            except Exception as e:
                log_processing_error(company_name, e, is_batch=True)
                result = {
                    'company': company_name,
                    'error': str(e),
                    'ai_response': 'Error occurred before AI processing could complete.'
                }
                logger.info(f"Sending result for {company_name}: {len(result)} fields")
                print(f"SENDING RESULT: {json.dumps(result, ensure_ascii=False)}")
                results.append(result)
        
        # Return batch results
        response_data = {
            'status': 'complete',
            'total_companies': total_companies,
            'processed_companies': len(results),
            'batch_size': len(current_batch),
            'results': results
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500





@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)

@app.route('/logs')
def show_logs():
    """Show recent application logs for deployment monitoring."""
    try:
        # Get log file path based on environment
        if os.environ.get('VERCEL') or os.environ.get('RAILWAY'):
            log_file = '/tmp/lead_generation.log'
        else:
            log_file = config.log_filename
        
        logs = []
        if os.path.exists(log_file):
            try:
                # Read last 100 lines of the log file
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Get last 100 lines, or all if less than 100
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    logs = [line.strip() for line in recent_lines if line.strip()]
            except Exception as e:
                logs = [f"Error reading log file: {str(e)}"]
        else:
            logs = ["No log file found"]
        
        # Get system information
        import psutil
        import platform
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            'memory_percent': f"{psutil.virtual_memory().percent:.1f}%",
            'disk_usage': f"{psutil.disk_usage('/').percent:.1f}%"
        }
        
        # Get environment information
        env_info = {
            'vercel': bool(os.environ.get('VERCEL')),
            'railway': bool(os.environ.get('RAILWAY')),
            'port': os.environ.get('PORT', '5000'),
            'generated_dir': GENERATED_DIR,
            'openai_key_set': bool(OPENAI_API_KEY),
            'tavily_key_set': bool(TAVILY_API_KEY),
            'tavily_client_initialized': tavily is not None
        }
        
        # Get recent API usage stats (if available)
        api_stats = {
            'last_request_time': None,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        # Try to get some basic stats from logs
        for log in logs:
            if 'Starting employee research for company:' in log:
                api_stats['total_requests'] += 1
            if 'Successfully processed' in log:
                api_stats['successful_requests'] += 1
            if 'Error processing' in log or 'Failed to get response' in log:
                api_stats['failed_requests'] += 1
        
        return jsonify({
            'status': 'success',
            'timestamp': time.time(),
            'log_file': log_file,
            'log_entries': len(logs),
            'recent_logs': logs,
            'system_info': system_info,
            'environment_info': env_info,
            'api_stats': api_stats
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/logs/raw')
def show_raw_logs():
    """Show raw log content for debugging."""
    try:
        # Get log file path based on environment
        if os.environ.get('VERCEL') or os.environ.get('RAILWAY'):
            log_file = '/tmp/lead_generation.log'
        else:
            log_file = config.log_filename
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return Response(content, mimetype='text/plain')
        else:
            return Response("No log file found", mimetype='text/plain')
            
    except Exception as e:
        return Response(f"Error reading logs: {str(e)}", mimetype='text/plain')

@app.route('/logs/clear')
def clear_logs():
    """Clear log file (for debugging purposes)."""
    try:
        # Get log file path based on environment
        if os.environ.get('VERCEL') or os.environ.get('RAILWAY'):
            log_file = '/tmp/lead_generation.log'
        else:
            log_file = config.log_filename
        
        if os.path.exists(log_file):
            # Clear the log file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')
            
            return jsonify({
                'status': 'success',
                'message': 'Log file cleared successfully',
                'timestamp': time.time()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No log file found to clear',
                'timestamp': time.time()
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500


# For Railway deployment
app.debug = False

if __name__ == '__main__':
    # Get port from Railway environment variable, default to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    
    # Log startup information
    is_railway = bool(os.environ.get('RAILWAY')) or bool(os.environ.get('PORT')) or 'railway' in os.environ.get('HOSTNAME', '').lower()
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Environment: Railway={is_railway}, Vercel={bool(os.environ.get('VERCEL'))}")
    logger.info(f"Generated directory: {GENERATED_DIR}")
    logger.info(f"OpenAI API key set: {bool(OPENAI_API_KEY)}")
    logger.info(f"Tavily API key set: {bool(TAVILY_API_KEY)}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        raise 