# Impact Analysis Agent

Intelligent GitHub repository analyzer that generates comprehensive tech stack recommendations using Groq AI.

## Render Deployment

### Quick Deploy to Render

1. **Fork/Clone this repository**
2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repository
   - Use these settings:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `python impact_agent.py`
     - **Environment:** Python 3

3. **Set Environment Variables in Render Dashboard:**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   GROQ_TEMPERATURE=0.3
   ```

4. **Deploy!** - Render will automatically deploy your app

### Get Groq API Key
1. Visit [console.groq.com](https://console.groq.com/)
2. Sign up/Login
3. Create API key
4. Add to Render environment variables

## Features

- 🚀 GitHub repository analysis
- 🤖 AI-powered tech stack recommendations
- 📂 Document processing (PDF, DOCX, TXT)
- 🔄 Alternative technology suggestions
- 📝 Implementation guides
- 🌐 Web interface

## Local Development

```bash
pip install -r requirements.txt
python impact_agent.py
```

Open: http://localhost:10000