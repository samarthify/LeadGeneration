# ---------- Sales Research AI – Google Colab (one-cell run) ----------
import os
import io
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import csv # Import csv module for robust parsing
import re # Keep import for extracting code blocks

import pandas as pd # Import pandas for DataFrame creation and handling
from openai import OpenAI
from tqdm import tqdm
# Removed import of google.colab.files as it's causing NameErrors
from google.colab import userdata # Import userdata to access secrets
# Removed ipywidgets and related display imports for command-line input
from IPython.display import display, HTML # Re-add display and HTML for final output table

from tavily import TavilyClient # Import TavilyClient

# -------------------- Configuration --------------------
@dataclass
class Config:
    """Class for managing all configuration values."""
    # OpenAI
    model: str = "gpt-4-turbo" # Changed model to gpt-4-turbo
    max_tokens: Optional[int] = None
    temperature: float = 0.7

    # Processing
    max_workers: int = 3          # Max parallel threads (respect API rate-limit)
    request_delay: float = 0.5    # Delay between requests (seconds)
    max_retries: int = 3          # Retry attempts per request
    tavily_max_results: int = 10  # Maximum number of results to request from Tavily

    # File handling
    company_column: str = "company"
    output_filename: str = "employee_research_formatted.csv" # Standardized output filename (for info purposes)
    log_filename: str = "research_log.txt"

    # Prompt handling
    enable_conversation_memory: bool = True  # Keep memory for the detailed prompt interaction
    max_conversation_history: int = 10       # Max messages to keep

    # Strict CSV Prompt
    strict_csv_prompt_template: str = (
        "Find and list employees for the company '{company}'. "
        "Focus specifically on executives, VPs, directors, and regional heads. "
        "Search multiple reliable sources such as official company websites (e.g., 'About Us', 'Leadership', 'Investor Relations' pages), "
        "press releases, reputable business news articles, and professional networking sites (if accessible via search). "
        "List as many relevant employees as you can find, including those from international offices or subsidiary companies part of the group (e.g., PFU, Fujitsu Uvance for Fujitsu). "
        "For each employee, extract their department and precise job title if available. "
        "Provide names in both native script (if found) and Roman alphabet. "
        "After collecting comprehensive data from multiple searches, carefully deduplicate entries based on their Romanized names (lowercase). "
        "Determine the most likely primary email domain for employees of this company. If regional domains exist (e.g., @us.company.com, @uk.company.com), prioritize the global domain if unsure, or list regional domains if clearly associated with a specific employee location found in the data. "
        "Finally, format the complete, deduplicated list as a **strict CSV** table with the following Japanese columns, ensuring data accuracy and consistency:\n"
        "会社名, 部署, 役職, 姓, 名, 姓（小文字ローマ字）, 名（小文字ローマ字）, メールアドレスに使用される可能性が高いドメイン\n" # Explicitly ask for strict CSV columns WITHOUT brackets
        "**IMPORTANT:** Ensure that no commas are included *within* any of the data fields (e.g., Department or Job title). If a department or job title contains a comma in the source, remove or replace the comma (e.g., with a semicolon or space) in the output CSV field.\n" # Explicitly add instruction to avoid commas in fields
        "If a field is missing, leave the corresponding cell empty. "
        "If a job title is not clearly available from the source, use the department name or the best possible approximation. Ensure all Romanized names are in lowercase. "
        "Output the final **strict CSV data**, including the header row, within a single markdown code block, with no extra text or formatting outside the code block." # Emphasize strict CSV output and no extra text
    )


# -------------------- Logging --------------------
def setup_logging(log_filename: str) -> logging.Logger:
    """Configure logging to both file and console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to prevent duplicate logs
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# -------------------- Main Class --------------------
class SalesResearchAI:
    """Core class for the Sales Research AI."""

    DEFAULT_PROMPTS = [
        "What are the current US tariffs related to the industry of '{company}'?",
        "Find recent news or announcements about changes to these tariffs.",
        "Summarize the potential impact of these tariffs on '{company}'.",
        "Are there any proposed or upcoming changes to these tariffs?",
    ]


    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_filename)
        self.client = self._initialize_openai_client()
        self.tavily_client = self._initialize_tavily_client() # Initialize Tavily client
        self.prompts = self.DEFAULT_PROMPTS.copy() # Keep default prompts but primarily use the single company one
        self.tools = self._define_tools() # Define tools here

    # ---------- API Initialization ----------
    def _initialize_openai_client(self) -> OpenAI:
        """Initialize OpenAI client."""
        api_key = userdata.get("OPENAI_API_KEY") # Use userdata.get for secrets
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found.\n"
                "Go to the Colab sidebar ➜ ⚙️ Secrets ➜ Add OPENAI_API_KEY."
            )
        return OpenAI(api_key=api_key)

    def _initialize_tavily_client(self) -> TavilyClient:
        """Initialize Tavily client."""
        api_key = userdata.get("TAVILY_API_KEY") # Use userdata.get for secrets
        if not api_key:
             raise RuntimeError(
                "TAVILY_API_KEY not found.\n"
                "Go to the Colab sidebar ➜ ⚙️ Secrets ➜ Add TAVILY_API_KEY."
            )
        return TavilyClient(api_key=api_key)


    def _define_tools(self) -> List[Dict]:
        """Define the tools available to the AI."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string.",
                            }
                            # Add more parameters if needed for specific search types (e.g., "site:company.com")
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    # ---------- CSV Utilities ----------
    def load_companies_from_csv(self, csv_data: bytes) -> List[str]:
        """Read company names from the uploaded CSV."""
        try:
            df = pd.read_csv(io.BytesIO(csv_data))

            print("📋 CSV structure:")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Rows    : {len(df)}")

            if self.config.company_column not in df.columns:
                raise ValueError(
                    f"Column '{self.config.company_column}' not found.\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Update config.company_column if needed."
                )

            companies = (
                df[self.config.company_column]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            companies = [c for c in companies if c and c.lower() != "nan"]

            self.logger.info("%d companies loaded", len(companies))
            print(f"✅ Loaded {len(companies)} companies")
            if companies:
                print(f"📝 Example companies: {companies[:3]}")
            return companies

        except Exception as e:
            self.logger.error("CSV load error: %s", e)
            raise

    def upload_csv_colab(self) -> List[str]:
        """Upload a CSV file in Colab and return company list."""
        # Note: This method uses google.colab.files.upload which seems to work differently
        # than google.colab.files.download in the current context. Keeping it for now.
        print(f"📂 Upload the company list CSV (column name: {self.config.company_column})")
        # Re-add files import here just in case upload needs it and the top-level one is flaky
        import google.colab.files
        uploaded = google.colab.files.upload()

        if not uploaded:
            raise ValueError("No file was uploaded.")

        csv_name = next(iter(uploaded))
        self.logger.info("Uploaded file: %s", csv_name)
        print(f"📁 Uploaded file: {csv_name}")
        return self.load_companies_from_csv(csv_data=uploaded[csv_name])

    # ---------- Custom Prompt Setter ----------
    def set_custom_prompts(self, prompts: List[str]):
        self.prompts = prompts
        self.logger.info("Custom prompts set: %d", len(prompts))
        print(f"🔧 Custom prompts set: {len(prompts)}")

    # ---------- API Request ----------
    def _make_api_request(self, messages: List[Dict], retry_count: int = 0) -> Optional[str]:
        """Call OpenAI chat completion with retries and tool handling."""
        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                tools=self.tools, # Pass defined tools
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
                            if query:
                                self.logger.info("Calling web_search with query: %s", query)
                                # --- Implement actual web search using Tavily ---
                                # Using search_depth="advanced" for potentially better results
                                search_results = self.tavily_client.search(query=query, search_depth="advanced", max_results=self.config.tavily_max_results).get('results', []) # Added max_results
                                # Format results for the model (you might need to adjust this based on what the model expects)
                                # Providing more context from results
                                formatted_results = "\n".join([
                                    f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content'][:2000]}..." # Increase content length limit
                                    for res in search_results
                                ])
                                if not formatted_results:
                                    formatted_results = "No search results found."
                                # -------------------------------------------------------------
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
                messages.append(resp.choices[0].message) # Append the assistant's message requesting tool calls
                messages.extend([{"role": "tool", "tool_call_id": to["tool_call_id"], "content": to["output"]} for to in tool_outputs])

                # Make another API call with tool outputs
                # Add a system message here potentially guiding the model on how to use tool outputs
                messages_with_tool_outputs = [
                    {"role": "system", "content": "Process the search results provided by the tool to extract the requested employee information and format it as a strict CSV within a markdown code block, including the header row, as requested in the user prompt."}, # System message guiding towards final strict CSV
                    *messages
                ]
                resp_with_outputs = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages_with_tool_outputs,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                return resp_with_outputs.choices[0].message.content.strip()

            # Handle regular text response
            return resp.choices[0].message.content.strip()

        except Exception as e:
            if retry_count < self.config.max_retries:
                self.logger.warning("API request failed (attempt %d): %s", retry_count + 1, e)
                time.sleep(2 ** retry_count)  # Exponential back-off
                return self._make_api_request(messages, retry_count + 1)
            self.logger.error("API request failed permanently: %s", e)
            return None

    # --- Utility function to extract code blocks ---
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extracts content within triple backticks (code blocks)."""
        # re is available due to import at the top
        pattern = r"```(?:\w+)?\n(.*?)```"
        return re.findall(pattern, text, re.DOTALL)

    # ---------- Detailed Employee Research (Single AI Call - Strict CSV Attempt) ----------
    def perform_employee_research(self, company: str) -> str: # Returns raw AI response string
        """Performs detailed employee research for a single company and returns raw AI output."""
        print(f"🕵️ Starting detailed employee research for '{company}' (Single AI Call - Strict CSV Attempt)...")
        self.logger.info(f"Starting detailed employee research for: {company} (Single AI Call - Strict CSV Attempt)")

        # Construct the specific prompt asking for strict CSV output
        research_prompt = self.config.strict_csv_prompt_template.format(company=company)

        # Initial messages for the AI call
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly skilled research assistant AI specializing in extracting employee data from web search results. "
                    "Follow the user's detailed instructions precisely, performing web searches using the 'web_search' tool, "
                    "and extracting structured employee data (names, departments, job titles). "
                    "Format the complete, deduplicated list as a **strict CSV** table with the specified columns within a markdown code block, including the header row, with no extra text." # Reiterate instruction for strict CSV
                ),
            },
            {"role": "user", "content": research_prompt} # Start with the main research prompt
        ]

        print("🗣️ Sending research prompt to the AI model (requesting strict CSV)...")
        self.logger.info("Sending research prompt to OpenAI (requesting strict CSV).")

        # This single API call is expected to orchestrate the steps using tool calls and output strict CSV
        raw_ai_response = self._make_api_request(messages)

        if not raw_ai_response:
            print("❌ No response from AI after research. Check logs for errors.")
            self.logger.error("AI did not return a response for research.")
            return "No response received from AI." # Return a message indicating no response

        print("✅ AI research process completed. Returning raw response string.")
        self.logger.info("Received raw response string from AI.")

        return raw_ai_response # Return the raw response string


    def parse_csv_from_ai_output(self, raw_ai_output: str) -> pd.DataFrame:
        """Extracts code block from raw AI output and parses its content as CSV into a DataFrame."""
        print("🔄 Attempting to parse AI output as CSV...")
        self.logger.info("Attempting to parse AI output as CSV.")

        if not raw_ai_output or raw_ai_output == "No response received from AI.":
            print("❌ No raw output provided for CSV parsing.")
            self.logger.warning("No raw output provided for CSV parsing.")
            return pd.DataFrame() # Return empty DataFrame if no raw output

        # --- Parsing the raw response for the strict CSV code block ---
        blocks = self._extract_code_blocks(raw_ai_output) # Use the class method

        if not blocks:
            self.logger.warning("No code blocks found in raw AI response for CSV data.")
            print("⚠️ Unable to find formatted data blocks (```code block```) in AI response. Outputting raw response below for inspection.\n")
            print(raw_ai_output)
            # Return empty DataFrame as we expect the final output in a code block
            return pd.DataFrame()

        # Assume the last code block contains the final CSV
        csv_text = blocks[-1].strip()
        print("🟢 Code block found. Attempting to parse as CSV using the 'csv' module for robustness.")
        self.logger.info(f"Extracted code block (first 200 chars): {csv_text[:200]}...")

        # --- Robust CSV Parsing using the 'csv' module ---
        data = []
        try:
            # Use io.StringIO to read the string content
            csvfile = io.StringIO(csv_text)
            # Use csv.reader to handle parsing, specifying the delimiter and quoting
            # Attempt to auto-detect delimiter or stick to comma
            try:
                # Try to sniff the delimiter
                # Read a larger sample for sniffing
                sample_size = min(len(csv_text), 4096) # sniff up to 4KB or less if text is shorter
                dialect = csv.Sniffer().sniff(csv_text[:sample_size])
                reader = csv.reader(csvfile, dialect)
                self.logger.info(f"CSV Sniffer detected dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
                print(f"🔬 Detected CSV dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
            except Exception as sniff_e:
                 self.logger.warning(f"CSV Sniffer failed, falling back to comma delimiter: {sniff_e}")
                 print("⚠️ CSV Sniffer failed, falling back to comma delimiter.")
                 csvfile.seek(0) # Reset file pointer
                 reader = csv.reader(csvfile, delimiter=',', quotechar='"')


            # Read the header row
            try:
                header = next(reader)
                data.append(header) # Keep header for DataFrame
            except StopIteration:
                self.logger.warning("CSV text is empty or only contains header.")
                print("⚠️ Warning: CSV text is empty or only contains header.")
                return pd.DataFrame() # Return empty DF if no header or data


            # Read the data rows - Include all rows
            for row in reader:
                data.append(row) # Append all rows regardless of column count
                # Log a warning if column count is unexpected, but don't skip the row
                if len(row) != len(header):
                     self.logger.warning(f"Unexpected column count in row during CSV parse: Expected {len(header)}, found {len(row)}. Row: {row}")
                     print(f"⚠️ Warning: Unexpected column count in a row during CSV parse. See logs for details.")


            # Convert the parsed data (list of lists) to a pandas DataFrame
            # Use the first row as headers
            if len(data) > 1: # Check if there's a header and at least one data row
                df = pd.DataFrame(data[1:], columns=data[0])
                print(f"✅ Successfully parsed CSV data using 'csv' module. Found {len(df)} employee records.")
                self.logger.info(f"Successfully parsed CSV output using 'csv' module: {len(df)} rows, columns: {list(df.columns)}")

                # Updated expected_cols to match the format requested in the prompt (without brackets)
                expected_cols = [
                    "Company name", "Department", "Job title", "Last name",
                    "First name", "Last name (lowercase Roman letters)",
                    "First name (lowercase Roman letters)", "Domain likely to be used in email addresses"
                ]
                df.columns = df.columns.str.strip() # Clean column names for comparison
                # Check if all expected columns are present after stripping
                if not all(col.strip() in df.columns for col in expected_cols):
                     self.logger.warning("Parsed CSV is missing expected columns or has unexpected names.")
                     print("⚠️ Warning: Parsed CSV is missing some expected columns or has unexpected names.")
                     print(f"Expected: {expected_cols}")
                     print(f"Found   : {list(df.columns)}")
                     # Continue with the available data but warn the user

                return df
            else:
                 self.logger.warning("Parsed CSV data is empty or only contains header after reading with 'csv' module.")
                 print("⚠️ Warning: Parsed CSV data is empty or only contains the header.")
                 # Return empty DF with expected columns if header is missing or doesn't match
                 return pd.DataFrame(columns=header if header else expected_cols) # Return empty DF with expected or found headers

        except Exception as e:
            self.logger.error(f"Failed to parse CSV data from the extracted block using 'csv' module: {e}", exc_info=True)
            print("❌ Failed to parse the extracted code block as CSV using the 'csv' module. Outputting the block content for inspection:\n")
            print(csv_text)
            # Return empty DataFrame if parsing fails
            return pd.DataFrame()


    # ---------- CSV Utilities ----------
    # (Keep load_companies_from_csv and upload_csv_colab if needed for other tasks)
    def load_companies_from_csv(self, csv_data: bytes) -> List[str]:
        """Read company names from the uploaded CSV."""
        try:
            df = pd.read_csv(io.BytesIO(csv_data))

            print("📋 CSV structure:")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Rows    : {len(df)}")

            if self.config.company_column not in df.columns:
                raise ValueError(
                    f"Column '{self.config.company_column}' not found.\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Update config.company_column if needed."
                )

            companies = (
                df[self.config.company_column]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            companies = [c for c in companies if c and c.lower() != "nan"]

            self.logger.info("%d companies loaded", len(companies))
            print(f"✅ Loaded {len(companies)} companies")
            if companies:
                print(f"📝 Example companies: {companies[:3]}")
            return companies

        except Exception as e:
            self.logger.error("CSV load error: %s", e)
            raise

    def upload_csv_colab(self) -> List[str]:
        """Upload a CSV file in Colab and return company list."""
        # Note: This method uses google.colab.files.upload which seems to work differently
        # than google.colab.files.download in the current context. Keeping it for now.
        print(f"📂 Upload the company list CSV (column name: {self.config.company_column})")
        # Re-add files import here just in case upload needs it and the top-level one is flaky
        import google.colab.files
        uploaded = google.colab.files.upload()

        if not uploaded:
            raise ValueError("No file was uploaded.")

        csv_name = next(iter(uploaded))
        self.logger.info("Uploaded file: %s", csv_name)
        print(f"📁 Uploaded file: {csv_name}")
        return self.load_companies_from_csv(csv_data=uploaded[csv_name])


    # ---------- All Companies (Original - Not used in single mode) ----------
    # This method is not used when processing a single company via the frontend input.
    # Keeping it here for now but the execution flow bypasses it.
    def process_all_companies(self, companies: List[str], use_parallel: bool = True) -> pd.DataFrame:
        """Run all prompts for every company."""
        print("Note: process_all_companies is not used in single company command-line mode.")
        self.logger.info("process_all_companies was called but is not used in single company mode.")
        return pd.DataFrame() # Return empty DataFrame if called in this mode


    # Removed save_raw_output method


    def generate_summary_report_dataframe(self, df: pd.DataFrame, company_name: str) -> str:
        """Prints a summary of the execution for single company employee research from DataFrame."""
        total_employees_found = len(df) if df is not None else 0

        report = f"""
📊 Sales Research AI – Single Company Employee Report (Formatted CSV)
================================
Run time: {datetime.now():%Y-%m-%d %H:%M:%S}

📈 Stats
- Company processed …… {company_name if company_name else 'N/A'}
- Employees found ……… {total_employees_found}
- Output format ………… Parsed CSV from AI (Printed Below)

⚙️ Settings
- Model ………………… {self.config.model}
- Chat history ……… {'enabled' if self.config.enable_conversation_memory else 'disabled'}
"""
        print(report)
        return report


# -------------------- Execution --------------------
try:
    print("🚀 Starting Sales Research AI …")
    print("=" * 50)

    cfg = Config(
        model="gpt-4-turbo", # Changed model to gpt-4-turbo
        # max_workers might not be directly used in the perform_employee_research method
        max_workers=3,
        request_delay=0.5,
        tavily_max_results=10, # Added tavily_max_results to config
        enable_conversation_memory=True, # Keep memory for the detailed prompt interaction
        # company_column is not used in this single company flow
        # output_filename is not directly used for the main output now
        log_filename="employee_research_log.txt" # More specific log file
    )

    # Configure logging early
    logger = setup_logging(cfg.log_filename)

    print("🔧 Initializing system …")
    # Pass the configured logger to the class
    research_ai = SalesResearchAI(cfg)
    print("✅ Initialization complete")

    # --- Get Company Name via simple command-line input ---
    print("\n" + "=" * 50)
    company_name = input("Please enter the company name and press Enter: ").strip()
    print("=" * 50)


    if not company_name:
        print("❌ No company name provided. Aborting.")
        logger.error("No company name provided via input. Aborting.")
    else:
        print(f"✅ Received company name: {company_name}")
        print("\n" + "=" * 50)
        print(f"🚀 Executing detailed employee research for {company_name} (Attempting to generate Structured CSV) …")
        logger.info(f"Starting detailed employee research workflow for {company_name} (Attempting to generate Structured CSV)")


        # --- Execute the single AI call research and get raw output ---
        # The AI is prompted to return strict CSV in a code block
        raw_ai_output = research_ai.perform_employee_research(company=company_name)


        print("\n" + "=" * 50)
        print("--- Raw AI Output ---")
        print(raw_ai_output)
        print("--- End of Raw AI Output ---")
        print("=" * 50)

        if raw_ai_output and raw_ai_output != "No response received from AI.":
             print(f"\n--- Attempting to Parse AI Output as CSV for {company_name} ---")
             # --- Attempt to parse the raw output as a DataFrame ---
             results_df = research_ai.parse_csv_from_ai_output(raw_ai_output)

             print("\n" + "=" * 50)

             if results_df is not None and not results_df.empty:
                  # --- Generate Report ---
                  research_ai.generate_summary_report_dataframe(results_df, company_name)


                  # --- Add download prompt for the DataFrame ---
                  print("\n⬇️ Downloading the generated CSV file...")
                  # Re-add files import here
                  import google.colab.files
                  # Generate filename based on company and timestamp
                  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                  csv_filename = f"{company_name.lower().replace(' ', '_')}_employees_{timestamp}.csv"

                  # Convert DataFrame to CSV string and save to a file
                  csv_string = results_df.to_csv(index=False, encoding="utf-8-sig") # Use utf-8-sig for BOM
                  with open(csv_filename, "w", encoding="utf-8-sig") as f:
                      f.write(csv_string)

                  # Trigger download
                  google.colab.files.download(csv_filename)
                  print(f"✅ CSV file '{csv_filename}' downloaded.")
                  logger.info(f"Generated and prompted download for '{csv_filename}'.")

                  # Optionally still print the CSV for quick viewing in notebook
                  print("\n📊 Formatted CSV Output Printed Below (for quick viewing):")
                  print("```csv") # Start markdown CSV code block
                  print(csv_string)
                  print("```") # End markdown CSV code block


                  print("\n💡 Tips:")
                  print("- The CSV file should have automatically downloaded to your computer.")
                  print("- If the automatic download did not start, look for a download prompt in your browser.")
                  print("- For deeper analysis, work with the downloaded file.")
                  print("- If data is missing or inaccurate, consider refining the research prompt.")
                  print("- If parsing errors occurred, check the raw AI response printed above and the logs (`employee_research_log.txt`).")


             else:
                 print("⚠️ No employee data was successfully parsed into a DataFrame.")
                 print(f"Please review the logs (`{cfg.log_filename}`), the raw AI response printed above, and the CSV parsing logic in `parse_csv_from_ai_output`.")
                 logger.warning(f"No employee data parsed into DataFrame for company {company_name}. Review logs and AI response.")

        else:
             print("⚠️ No raw AI output received from the research step. Cannot proceed with parsing.")
             logger.warning("No raw AI output received from research step.")


except KeyboardInterrupt:
    print("\n⚠️ Execution interrupted by user.")
    if 'logger' in locals():
         logger.info("Execution interrupted by user.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
    if 'logger' in locals():
         logger.error(f"An unexpected error occurred: {e}", exc_info=True) # Log exception details
    print("\n🔍 Troubleshooting:")
    print("1. Verify OPENAI_API_KEY and TAVILY_API_KEY are set correctly in Colab Secrets (Left sidebar ➜ 🔑).")
    print("2. Ensure your internet connection is stable.")
    print(f"3. Check the '{cfg.log_filename if 'cfg' in locals() else 'research_log.txt'}' file for detailed error messages.") # Updated log filename hint
    import traceback
    traceback.print_exc() # Print full traceback for debugging