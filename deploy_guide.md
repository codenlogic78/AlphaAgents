# AlphaAgents Deployment Guide

## 🚀 Quick Deploy to Streamlit Cloud

### Prerequisites:
- GitHub account
- AlphaAgents code in GitHub repo
- OpenAI API key
- NewsAPI key

### Steps:

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "AlphaAgents system ready for deployment"
   git remote add origin https://github.com/yourusername/alphaagents.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repo
   - Set main file: `streamlit_simple.py`
   - Add secrets in "Advanced settings":
     ```
     OPENAI_API_KEY = "your-openai-key"
     NEWS_API_KEY = "your-news-key"
     ```

3. **Deploy!**
   - Click "Deploy"
   - Get public URL for sharing

### Environment Variables Needed:
- `OPENAI_API_KEY`
- `NEWS_API_KEY`
- `API_DELAY=2`
- `MAX_RETRIES=1`
- `OPENAI_MAX_CALLS_PER_ANALYSIS=3`
- `OPENAI_CALL_DELAY=2`

## 🐳 Docker Deployment

### Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_simple.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Deploy Commands:
```bash
docker build -t alphaagents .
docker run -p 8501:8501 --env-file .env alphaagents
```

## 🌐 Alternative Platforms

### Heroku:
- Create `Procfile`: `web: streamlit run streamlit_simple.py --server.port=$PORT --server.address=0.0.0.0`
- Add buildpack: `heroku/python`

### Railway:
- Connect GitHub repo
- Auto-detects Python app
- Add environment variables

### Render:
- Connect GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run streamlit_simple.py --server.port=$PORT --server.address=0.0.0.0`
