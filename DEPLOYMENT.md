# Deploying to Vercel

## Prerequisites
1. Install Vercel CLI: `npm i -g vercel`
2. Have a Vercel account (sign up at vercel.com)

## Environment Variables
You need to set these environment variables in your Vercel project:

1. Go to your Vercel dashboard
2. Select your project
3. Go to Settings > Environment Variables
4. Add the following variables:
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `TAVILY_API_KEY` - Your Tavily API key

## Deployment Steps

### Option 1: Using Vercel CLI
1. Login to Vercel: `vercel login`
2. Deploy: `vercel`
3. Follow the prompts to link to your project

### Option 2: Using GitHub Integration
1. Push your code to GitHub
2. Connect your GitHub repository to Vercel
3. Vercel will automatically deploy on every push

### Option 3: Using Vercel Dashboard
1. Go to vercel.com
2. Click "New Project"
3. Import your GitHub repository
4. Configure environment variables
5. Deploy

## Important Notes

- The app uses `/tmp` directory for file storage in Vercel (serverless environment)
- Generated CSV files are temporary and will be deleted after the function execution
- Make sure all environment variables are properly set in Vercel dashboard
- The app is configured for serverless deployment with `app.debug = False`

## Troubleshooting

If you encounter issues:
1. Check Vercel function logs in the dashboard
2. Ensure all environment variables are set correctly
3. Verify that all dependencies are in `requirements.txt`
4. Check that the `vercel.json` configuration is correct 