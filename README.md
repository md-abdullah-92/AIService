# AI Service - PDF Processing API

A FastAPI-based service for processing PDF files and generating educational content including MCQs, short answer questions, and study notes using Google's Gemini AI.

## Features

- PDF Upload and Page Analysis
- MCQ Generation
- Short Answer Question Generation
- Study Notes Generation
- Page Range Selection

## Deployment on Render (Free)

### Prerequisites
- GitHub account
- Google AI API Key (Gemini)

### Steps

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Sign up/login with GitHub
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Choose this repository
   - Fill in the settings:
     - **Name**: aiservice (or your preferred name)
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free

3. **Set Environment Variables**:
   - In Render dashboard → Environment
   - Add: `GEMINI_API_KEY` = your_google_ai_api_key

4. **Deploy**: Click "Create Web Service"

### API Endpoints

- `GET /` - Health check
- `POST /upload/` - Upload PDF and get page count
- `POST /quiz/` - Generate quiz questions
- `POST /generate-study-notes/` - Generate study notes
- `POST /short-questions/` - Generate all short questions

### Environment Variables Required

- `GEMINI_API_KEY`: Your Google AI API key

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit: http://localhost:8000
