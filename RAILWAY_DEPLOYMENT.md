# Railway Deployment Guide

This guide will help you deploy your Lead Generation Flask application to Railway.

## Prerequisites

1. Create a Railway account at [railway.app](https://railway.app)
2. Install Railway CLI (optional but recommended):
   ```bash
   npm install -g @railway/cli
   ```

## Deployment Steps

### Method 1: Using Railway Dashboard (Recommended)

1. **Connect your GitHub repository:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

2. **Configure environment variables:**
   - In your Railway project dashboard, go to "Variables"
   - Add the following environment variables:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     TAVILY_API_KEY=your_tavily_api_key_here
     ```

3. **Deploy:**
   - Railway will automatically detect the Python project
   - It will use the `Procfile` to start the application
   - The app will be available at the provided Railway URL

### Method 2: Using Railway CLI

1. **Login to Railway:**
   ```bash
   railway login
   ```

2. **Initialize Railway project:**
   ```bash
   railway init
   ```

3. **Set environment variables:**
   ```bash
   railway variables set OPENAI_API_KEY=your_openai_api_key_here
   railway variables set TAVILY_API_KEY=your_tavily_api_key_here
   ```

4. **Deploy:**
   ```bash
   railway up
   ```

## Configuration Files

The following files have been created/updated for Railway deployment:

- **`Procfile`**: Tells Railway how to run the application
- **`runtime.txt`**: Specifies Python version
- **`railway.json`**: Railway-specific configuration
- **`.railwayignore`**: Excludes unnecessary files from deployment
- **`requirements.txt`**: Updated with specific versions for stability

## Environment Variables

Make sure to set these environment variables in your Railway project:

- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key
- `PORT`: Automatically set by Railway (don't change this)

## Troubleshooting

### Common Issues:

1. **Build fails:**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version in `runtime.txt` is supported

2. **App crashes on startup:**
   - Check environment variables are set correctly
   - Review logs in Railway dashboard

3. **File storage issues:**
   - The app uses `/tmp` directory for file storage on Railway
   - Files are temporary and will be lost on restart

### Checking Logs:

- Use Railway dashboard to view real-time logs
- Or use CLI: `railway logs`

## Local Testing

To test the Railway configuration locally:

```bash
# Set environment variables
export PORT=5000
export OPENAI_API_KEY=your_key_here
export TAVILY_API_KEY=your_key_here

# Run the app
python app.py
```

The app should start on `http://localhost:5000`

## Notes

- Railway provides a `PORT` environment variable automatically
- The app is configured to use `/tmp` for file storage (temporary)
- All generated CSV files will be available for download but are temporary
- The app automatically handles Railway's environment variables 