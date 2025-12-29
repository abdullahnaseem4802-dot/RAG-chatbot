# API Documentation - Eastern Services Pest Control Backend

## Base URL
```
Production: https://huggingface.co/spaces/raja3134802/eastern-services-api
Local: http://localhost:7860
```

---

## 🔍 Endpoints

### 1. Health Check
**GET** `/health`

Check if API is running.

**Response:**
```json
{
  "status": "healthy",
  "initialized": true,
  "error": null,
  "components": {
    "redis": true,
    "database": true,
    "cohere": true,
    "llm": true,
    "yolov8": true
  }
}
```

---

### 2. Welcome Message
**GET** `/welcome`

Get chatbot welcome message.

**Response:**
```json
{
  "message": "السلام وعلیکم! 👋\n\nWelcome to Eastern Services Pest Control AI Assistant..."
}
```

---

### 3. Chat (RAG)
**POST** `/chat`

Send text message to chatbot.

**Request Body:**
```json
{
  "question": "termite treatment price",
  "session_id": "user123",  // optional
  "language": "english"      // optional: english, urdu, roman_urdu
}
```

**Response:**
```json
{
  "response": "Termite treatment costs Rs. 15,000 - 25,000...",
  "session_id": "user123",
  "language": "english"
}
```

**Example:**
```javascript
const response = await fetch('https://your-api.com/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    question: "How much for termite treatment?",
    session_id: "abc123"
  })
});
const data = await response.json();
console.log(data.response);
```

---

### 4. Image Analysis (YOLOv8)
**POST** `/api/analyze-image`

Detect pests in uploaded image.

**Request Body:**
```json
{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "language": "english",
  "session_id": "user123"
}
```

**Response:**
```json
{
  "analysis": "🔎 Initial Assessment: The visual analysis strongly indicates the presence of termite.\n\nBased on this preliminary identification, we recommend the following professional service plan:\n\nTermite Treatment - Eastern Services\nTreatment: Complete termite proofing with chemical barrier\nPrice: Rs. 15,000 - 25,000 (depends on area)\nCoverage: Full property treatment\nGuarantee: 5-year warranty\nFollow-up: Annual inspections included\n\nContact: +92 336 1101234",
  "pest_identified": "termite",
  "treatment_recommendation": "...",
  "session_id": "user123",
  "language": "english"
}
```

**Example:**
```javascript
// Convert image to base64
const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });
};

// Upload image
const file = document.getElementById('imageInput').files[0];
const base64 = await fileToBase64(file);

const response = await fetch('https://your-api.com/api/analyze-image', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    image_data: base64,
    language: 'english',
    session_id: 'user123'
  })
});
const data = await response.json();
console.log(data.analysis);
```

---

### 5. Text-to-Speech
**POST** `/api/tts`

Convert text to speech audio. Returns an MP3 stream.

**Request Body:**
```json
{
  "text": "Hello, how can I help you?",
  "language": "en"  // en or ur
}
```

**Response:**
- Content-Type: `audio/mpeg`
- Body: MP3 stream

**Example (browser):**
```javascript
const res = await fetch('https://your-api.com/api/tts', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Welcome to Eastern Services', language: 'en' })
});
const blob = await res.blob();
const url = URL.createObjectURL(blob);
const audio = new Audio(url);
audio.play();
```

---

## 🌐 Language Support

| Code | Language | Example |
|------|----------|---------|
| `english` | English | "How much for termite treatment?" |
| `urdu` | Urdu | "دیمک کا علاج کتنے کا ہے؟" |
| `roman_urdu` | Roman Urdu | "Deemak ka ilaaj kitne ka hai?" |

---

## 🔐 CORS Configuration

By default, the API allows all origins (`*`) to simplify integration. For production, you can restrict origins in `app.py`.
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://easternservices.pk", "https://www.easternservices.pk"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ⚡ Rate Limiting

Not enforced in the current backend. Recommended gateway limits (if needed):
- Chat: 60 requests/minute per session
- Image Analysis: 10 requests/minute per session
- TTS: 30 requests/minute per session

---

## 🐛 Error Handling

All endpoints return standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 503 | Service Unavailable (chatbot not initialized) |
| 500 | Internal Server Error |

**Error Response:**
```json
{
  "detail": "Error message here"
}
```

---

## 📊 Supported Pests (YOLOv8)

| Pest | Accuracy | Detection Speed |
|------|----------|-----------------|
| Mosquito | 93% | ~500ms |
| Termite | 89% | ~500ms |
| Cockroach | 85% | ~500ms |
| Rodent | 87% | ~500ms |

---

## 🔧 Testing

### Using cURL:

```bash
# Health check
curl https://your-api.com/health

# Chat
curl -X POST https://your-api.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"termite price","session_id":"test123"}'

# Image analysis (with base64 image)
curl -X POST https://your-api.com/api/analyze-image \
  -H "Content-Type: application/json" \
  -d '{"image_data":"data:image/jpeg;base64,...","language":"english"}'
```

### Using Postman:

1. Import collection from `postman_collection.json`
2. Set base URL variable
3. Test all endpoints

---

## 📞 Support

For API integration support:
- Email: abdullahnaseem4802@gmail.com
- Documentation: See `INTEGRATION_GUIDE.md`
