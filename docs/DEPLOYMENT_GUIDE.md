# Deployment Guide - Hugging Face Spaces

## 🚀 Deploy Backend to Hugging Face Spaces

### Prerequisites
- Hugging Face account
- Git installed (optional)
- Environment variables ready

---

## 📦 Method 1: Web Interface Upload

### Step 1: Create New Space

1. Go to: https://huggingface.co/new-space
2. Fill in details:
   - **Name:** `eastern-services-chatbot`
   - **License:** MIT
   - **SDK:** Docker
   - **Hardware:** CPU Basic (free)
3. Click **"Create Space"**

### Step 2: Upload Files

Upload all files from this repository (project root):

**Core Files:**
1. `app.py`
2. `chatbot_core.py`
3. `yolov8_detector.py`
4. `mock_rag_local.py`
5. `requirements.txt`
6. `Dockerfile`

**Model:**
7. `models/best.pt` (create `models/` folder first)

**How to Upload:**
- Click **"Files"** tab
- Click **"Add file"** → **"Upload files"**
- Select files and commit

### Step 3: Configure Environment Variables

1. Go to **"Settings"** tab
2. Click **"Variables and secrets"**
3. Add these secrets:

```
GROQ_API_KEY=your_groq_key
SUPABASE_URI=postgresql://...
REDIS_URI=rediss://...
COHERE_API_KEY=your_cohere_key
```

4. Click **"Save"**

### Step 4: Wait for Build

1. Go to **"Logs"** tab
2. Wait 10-15 minutes for build
3. Look for:
   ```
   [OK] YOLOv8 model loaded from models/best.pt
   [OK] Chatbot initialized successfully!
   Running on http://0.0.0.0:7860
   ```

### Step 5: Test API

```bash
curl https://your-space-name.hf.space/health
```

Expected:
```json
{
  "status": "healthy",
  "initialized": true
}
```

---

## 📦 Method 2: Git Push

### Step 1: Clone Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/eastern-services-api
cd eastern-services-chatbot
```

### Step 2: Copy Files

```bash
# Copy all backend files from your local project root into the cloned Space
cp -r /path/to/eastern-services-chatbot/* .
```

### Step 3: Push to HF

```bash
git add .
git commit -m "Initial backend deployment"
git push
```

### Step 4: Configure Secrets

Same as Method 1, Step 3

---

## 🔧 Dockerfile Configuration

The `Dockerfile` is already configured for HF Spaces:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY yolov8_detector.py .
COPY chatbot_core.py .
COPY mock_rag_local.py .

# Copy model
RUN mkdir -p models
COPY models/best.pt models/best.pt

# Expose port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 🌐 Custom Domain (Optional)

To use your own domain (e.g., `api.easternservices.pk`):

1. Go to Space Settings
2. Click **"Domains"**
3. Add custom domain
4. Update DNS records:
   ```
   CNAME api.easternservices.pk -> your-space.hf.space
   ```

---

## 📊 Monitoring

### Health Check

Set up monitoring with UptimeRobot:

1. Go to: https://uptimerobot.com
2. Add new monitor:
   - **Type:** HTTP(s)
   - **URL:** `https://your-space.hf.space/health`
   - **Interval:** 5 minutes
3. This keeps your API warm (no cold starts)

### Logs

View real-time logs:
```bash
# In HF Space, go to "Logs" tab
# Or use HF CLI:
huggingface-cli space logs YOUR_USERNAME/eastern-services-chatbot
```

---

## 🔐 Security

### CORS Configuration

Update `app.py` to allow your domain:

```python
origins = [
    "https://easternservices.pk",
    "https://www.easternservices.pk",
    "http://localhost:3000"  # for testing
]
```

### API Keys

- Never commit `.env` file
- Use HF Secrets for production
- Rotate keys regularly

---

## ⚡ Performance

### Expected Performance:
- **Cold Start:** 30-60 seconds (first request)
- **Warm Response:** 1-3 seconds
- **Image Analysis:** 3-5 seconds

### Optimization Tips:
1. Use UptimeRobot to keep warm
2. Implement caching for common queries
3. Consider upgrading to GPU Space for faster inference

---

## 🐛 Troubleshooting

### Build Fails

**Check:**
- All files uploaded correctly
- `requirements.txt` has correct versions
- Dockerfile syntax is correct

**View logs:**
- Go to "Logs" tab in HF Space
- Look for error messages

### API Not Responding

**Check:**
- Space is running (green status)
- Environment variables are set
- Health endpoint returns 200

**Test:**
```bash
curl https://your-space.hf.space/health
```

### CORS Errors

**Solution:**
Add your domain to CORS origins in `app.py`

---

## 📞 Support

For deployment support:
- HF Docs: https://huggingface.co/docs/hub/spaces
- Email: abdullahnaseem4802@gmail.com

---

## ✅ Deployment Checklist

- [ ] HF Space created
- [ ] All files uploaded
- [ ] Environment variables configured
- [ ] Build completed successfully
- [ ] Health check passes
- [ ] Chat endpoint tested
- [ ] Image analysis tested
- [ ] UptimeRobot configured
- [ ] Custom domain configured (optional)
- [ ] Frontend integrated

**Your backend is ready for production!** 🎉
