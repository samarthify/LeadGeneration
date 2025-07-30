# Railway Deployment Troubleshooting Guide

## Common Issues and Solutions

### 1. "Failed to process CSV batch" Error

**Symptoms:**
- Frontend shows "Failed to process CSV batch" error
- Backend logs show processing has started
- No streaming data reaches the frontend

**Causes:**
- Railway's proxy/load balancer buffering SSE responses
- CORS issues with streaming responses
- Network timeouts

**Solutions:**

#### A. Test SSE Streaming
Visit your Railway URL + `/test-sse` to test if SSE streaming works:
```
https://your-app.railway.app/test-sse
```

#### B. Check Health Endpoint
Visit your Railway URL + `/health` to verify the app is running:
```
https://your-app.railway.app/health
```

#### C. Environment Variables
Ensure these are set in Railway dashboard:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 2. Connection Timeout Issues

**Symptoms:**
- Request times out after 30 seconds
- Large batches fail to process

**Solutions:**
- Try with smaller batches (5-10 companies)
- Check Railway logs for memory/CPU limits
- Consider upgrading Railway plan if needed

### 3. File Storage Issues

**Symptoms:**
- CSV files not downloadable
- "File not found" errors

**Solutions:**
- Files are stored in `/tmp` on Railway (temporary)
- Files are lost on app restart
- Consider using external storage (S3, etc.) for production

### 4. Memory/CPU Issues

**Symptoms:**
- App crashes during processing
- Slow response times

**Solutions:**
- Reduce batch size
- Check Railway resource usage
- Consider upgrading plan

## Debugging Steps

### Step 1: Check Railway Logs
1. Go to Railway dashboard
2. Click on your project
3. Go to "Deployments" tab
4. Click on latest deployment
5. Check "Logs" tab

### Step 2: Test Basic Functionality
1. Test health endpoint: `GET /health`
2. Test SSE streaming: `GET /test-sse`
3. Test single company: Use the single company form

### Step 3: Check Environment
Look for these in Railway logs:
```
Environment: {'railway': True, 'vercel': False, 'port': '3000', 'generated_dir': '/tmp'}
```

### Step 4: Monitor Resource Usage
- Check CPU usage in Railway dashboard
- Check memory usage
- Look for "out of memory" errors

## Railway-Specific Configuration

### Environment Variables
Railway automatically sets:
- `PORT`: The port your app should listen on
- `RAILWAY`: Set to `true` when deployed on Railway

### File Storage
- Use `/tmp` directory for temporary files
- Files are lost on restart
- Consider external storage for production

### Network Configuration
- Railway uses HTTPS by default
- CORS headers are properly configured
- SSE streaming should work with current setup

## Performance Optimization

### For Large Batches:
1. Process companies one by one
2. Add delays between API calls
3. Use smaller search result limits
4. Implement proper error handling

### Memory Management:
1. Clear variables after use
2. Use generators for large data
3. Limit concurrent operations

## Common Error Messages

### "Failed to fetch"
- Network connectivity issue
- Check Railway app status
- Verify URL is correct

### "Connection timeout"
- Request taking too long
- Reduce batch size
- Check API rate limits

### "Internal Server Error"
- Check Railway logs
- Verify environment variables
- Check API key validity

### "Client.__init__() got an unexpected keyword argument 'proxies'"
- This is a Tavily client version compatibility issue
- Fixed by downgrading to tavily-python==0.2.8
- Check if TAVILY_API_KEY is set correctly

## Getting Help

1. **Check Railway Logs**: Always start here
2. **Test Endpoints**: Use `/health` and `/test-sse`
3. **Reduce Complexity**: Try with single company first
4. **Check Environment**: Verify all variables are set
5. **Monitor Resources**: Check CPU/memory usage

## Production Considerations

For production use, consider:
- External file storage (S3, etc.)
- Database for storing results
- Queue system for large batches
- Monitoring and alerting
- Rate limiting and API quotas 