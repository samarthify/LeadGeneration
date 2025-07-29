# Lead Generation Tool

A powerful AI-powered lead generation tool that automatically finds and extracts contact information for potential leads from company websites. This tool uses advanced web scraping, AI analysis, and data processing to generate comprehensive lead lists.

## 🚀 Features

- **AI-Powered Lead Discovery**: Uses OpenAI GPT-4 and Tavily search to find potential leads
- **Batch Processing**: Upload CSV files with company information for bulk processing
- **Real-time Progress Tracking**: Monitor processing progress with live updates
- **Export Results**: Download generated lead lists in CSV format
- **Modern UI**: Beautiful, responsive web interface with real-time feedback
- **Error Handling**: Robust error handling and retry mechanisms
- **Logging**: Comprehensive logging for debugging and monitoring

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/samarthify/LeadGeneration.git
cd LeadGeneration
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔑 API Keys Setup

This application requires two API keys to function properly:

### 1. OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to "API Keys" in the sidebar
4. Click "Create new secret key"
5. Copy the generated key

### 2. Tavily API Key

1. Go to [Tavily AI](https://tavily.com/)
2. Sign up for a free account
3. Navigate to your dashboard
4. Copy your API key

### 3. Environment Configuration

Create a `.env` file in the root directory of the project:

```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

## 🚀 Running the Application

### Development Mode

```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

### Production Mode

For production deployment, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 Usage Guide

### 1. Web Interface

1. Open your browser and navigate to `http://127.0.0.1:5000`
2. You'll see the Lead Generation Tool interface

### 2. Upload CSV File

The application expects a CSV file with the following columns. You can use either English or Japanese column headers:

**English Format:**
| Column Name | Description | Required |
|-------------|-------------|----------|
| Company name | The company name to search for | Yes |
| Department | Target department (optional) | No |
| Job title | Target job title (optional) | No |
| Last name | Contact's last name in kanji (optional) | No |
| First name | Contact's first name in kanji (optional) | No |
| Last name (lowercase Roman letters) | Romanized last name (optional) | No |
| First name (lowercase Roman letters) | Romanized first name (optional) | No |
| Domain likely to be used in email addresses | Company domain | Yes |

**Japanese Format (日本語形式):**
| Column Name (Japanese) | Description | Required |
|----------------------|-------------|----------|
| 会社名 | The company name to search for | Yes |
| 部署 | Target department (optional) | No |
| 役職 | Target job title (optional) | No |
| 姓 | Contact's last name in kanji (optional) | No |
| 名 | Contact's first name in kanji (optional) | No |
| 姓（小文字ローマ字） | Romanized last name (optional) | No |
| 名（小文字ローマ字） | Romanized first name (optional) | No |
| メールアドレスに使用される可能性が高いドメイン | Company domain | Yes |

**Important Notes:**
- The application supports both English and Japanese column headers
- Japanese names (姓/名) should be in kanji format
- Romanized names (姓（小文字ローマ字）/名（小文字ローマ字）) should be in lowercase Roman letters
- The domain column is required for email generation

### 3. Processing

1. Click "Choose File" and select your CSV file
2. Click "Upload and Process" to start the lead generation
3. Monitor the progress in real-time
4. Download the results when processing is complete

### 4. Output

The application generates a CSV file with the following Japanese column headers and information for each lead:

**Japanese Output Format:**
- 会社名 (Company name)
- 部署 (Department)
- 役職 (Job title)
- 姓 (Last name in kanji)
- 名 (First name in kanji)
- 姓（小文字ローマ字） (Romanized last name)
- 名（小文字ローマ字） (Romanized first name)
- メールアドレスに使用される可能性が高いドメイン (Domain for email generation)

**Additional Information Generated:**
- Email addresses (generated based on company domain)
- LinkedIn profiles (if found)
- Phone numbers (if found)
- Additional notes and context

## 📁 Project Structure

```
LeadGeneration/
├── app.py                 # Main Flask application
├── collab.py             # Collaboration utilities
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel deployment configuration
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore file
├── templates/
│   └── index.html       # Web interface template
└── generated/           # Output directory for generated files
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `TAVILY_API_KEY` | Your Tavily API key | Yes |

### Application Settings

The application uses a `Config` class in `app.py` for centralized configuration:

- **OpenAI Settings**: Model, tokens, temperature, etc.
- **API Settings**: Retry limits, request delays
- **Tavily Settings**: Search depth, result limits
- **Processing Settings**: URL limits, search result limits

## 🚀 Deployment

### Vercel Deployment

This application is configured for Vercel deployment:

1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Deploy to Vercel:
   ```bash
   vercel --prod
   ```

3. Set environment variables in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`

### Other Platforms

The application can be deployed to any platform that supports Python Flask applications:

- **Heroku**: Use the provided `requirements.txt`
- **Railway**: Direct deployment from GitHub
- **DigitalOcean App Platform**: Container deployment
- **AWS Elastic Beanstalk**: Platform as a service

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your API keys are correctly set in the `.env` file
   - Verify your API keys have sufficient credits/quota

2. **Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

3. **Port Already in Use**
   - Change the port in `app.py` or kill the process using the port
   - Use: `lsof -ti:5000 | xargs kill -9` (Linux/Mac)

4. **File Upload Issues**
   - Ensure the `generated/` directory exists
   - Check file permissions

### Logs

The application logs to both console and file:
- Console: Real-time output
- File: `lead_generation.log` (in development)

## 📈 Performance Tips

1. **API Rate Limits**: The application includes built-in rate limiting and retry mechanisms
2. **Batch Processing**: Process multiple companies in a single upload for efficiency
3. **Caching**: Results are cached to avoid duplicate API calls
4. **Parallel Processing**: The application processes multiple search queries concurrently

## 🔒 Security Considerations

1. **API Keys**: Never expose API keys in client-side code
2. **File Uploads**: Validate uploaded files for security
3. **Environment Variables**: Use `.env` files for sensitive configuration
4. **HTTPS**: Use HTTPS in production for secure data transmission

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the application logs
3. Create an issue in the repository
4. Contact the development team

## 🎯 Roadmap

- [ ] Enhanced email validation
- [ ] Social media profile detection
- [ ] Advanced filtering options
- [ ] Integration with CRM systems
- [ ] Mobile app version
- [ ] Multi-language support

---

**Happy Lead Generation! 🎉** 
