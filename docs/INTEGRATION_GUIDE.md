# Integration Guide - Eastern Services Website Chatbot

## 🎯 Overview

This guide shows how to integrate the Eastern Services Pest Control AI backend with your existing website chatbot widget.

---

## 📋 Prerequisites

- Your existing chatbot widget (visible on easternservices.pk)
- Backend API deployed on Hugging Face Spaces
- Basic JavaScript knowledge

---

## 🔗 Integration Steps

### Step 1: Update API Base URL

In your chatbot widget JavaScript, set the API URL:

```javascript
const API_BASE_URL = 'https://huggingface.co/spaces/raja3134802/eastern-services-api'; // replace with your deployed Space URL
```

---

### Step 2: Chat Integration

Replace your existing chat function with API calls:

```javascript
async function sendMessage(userMessage) {
    // Show user message in chat
    displayMessage(userMessage, 'user');
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Call backend API
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: userMessage,
                session_id: getSessionId(), // Your session management
                language: getCurrentLanguage() // 'english', 'urdu', or 'roman_urdu'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Display AI response
        displayMessage(data.response, 'bot');
        
    } catch (error) {
        console.error('Chat error:', error);
        hideTypingIndicator();
        displayMessage('Sorry, there was an error. Please try again.', 'bot');
    }
}
```

---

### Step 3: Image Upload Integration

Add image upload functionality to your chatbot:

```javascript
async function handleImageUpload(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
        alert('Image too large. Please select an image smaller than 10MB');
        return;
    }
    
    // Show processing message
    displayMessage(`🖼️ Analyzing image: ${file.name}...`, 'user');
    showTypingIndicator();
    
    try {
        // Convert image to base64
        const base64Image = await fileToBase64(file);
        
        // Call backend API
        const response = await fetch(`${API_BASE_URL}/api/analyze-image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_data: base64Image,
                language: getCurrentLanguage(),
                session_id: getSessionId()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Display analysis result
        displayMessage(data.analysis, 'bot');
        
    } catch (error) {
        console.error('Image upload error:', error);
        hideTypingIndicator();
        displayMessage('Sorry, there was an error analyzing the image.', 'bot');
    }
}

// Helper function to convert file to base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}
```

---

### Step 4: Text-to-Speech Integration (Optional)

Add voice playback for bot responses:

```javascript
async function playResponse(text) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/tts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                language: getCurrentLanguage() === 'urdu' ? 'ur' : 'en'
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // The endpoint returns an MP3 stream
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play();

    } catch (error) {
        console.error('TTS error:', error);
    }
}

// Add play button to bot messages
function displayMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = message;
    
    if (sender === 'bot') {
        // Add play button
        const playButton = document.createElement('button');
        playButton.innerHTML = '🔊 Play';
        playButton.onclick = () => playResponse(message);
        messageDiv.appendChild(playButton);
    }
    
    chatContainer.appendChild(messageDiv);
}
```

---

### Step 5: Session Management

Implement session tracking:

```javascript
function getSessionId() {
    // Check if session exists in localStorage
    let sessionId = localStorage.getItem('eastern_chatbot_session');
    
    if (!sessionId) {
        // Generate new session ID
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('eastern_chatbot_session', sessionId);
    }
    
    return sessionId;
}
```

---

### Step 6: Language Support

Add language switcher:

```javascript
function getCurrentLanguage() {
    const languageSelector = document.getElementById('languageSelector');
    const value = languageSelector ? languageSelector.value : 'en-US';
    
    // Map to backend language codes
    const languageMap = {
        'ur-PK': 'urdu',
        'en-US': 'english',
        'ur-Roman': 'roman_urdu'
    };
    
    return languageMap[value] || 'english';
}

// Language selector HTML
<select id="languageSelector" onchange="changeLanguage(this.value)">
    <option value="ur-PK">🇵🇰 اردو (Urdu)</option>
    <option value="en-US">🇬🇧 English</option>
</select>
```

---

## 🎨 UI Integration

### Your Existing Chatbot Widget

Based on your website screenshots, you have a chatbot widget in the bottom-right corner. Here's how to integrate:

```html
<!-- Your existing chatbot widget -->
<div id="chatbot-widget">
    <div id="chatbot-header">
        <span>Eastern Services AI Online</span>
        <select id="languageSelector">
            <option value="ur-PK">🇵🇰 اردو</option>
            <option value="en-US">🇬🇧 English</option>
        </select>
    </div>
    
    <div id="chatbot-messages">
        <!-- Messages will appear here -->
    </div>
    
    <div id="chatbot-input">
        <input type="text" id="messageInput" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>
        <input type="file" id="imageInput" accept="image/*" onchange="handleImageUpload(event.target.files[0])" style="display:none">
        <button onclick="document.getElementById('imageInput').click()">📷 Image</button>
    </div>
</div>
```

---

## 🧪 Testing

### Test Checklist:

1. **Text Chat:**
   - [ ] Send "termite treatment price"
   - [ ] Verify response contains treatment details
   - [ ] Test in Urdu and English

2. **Image Upload:**
   - [ ] Upload mosquito image
   - [ ] Verify detection and treatment recommendation
   - [ ] Test with different pest images

3. **Session Persistence:**
   - [ ] Send multiple messages
   - [ ] Verify conversation context is maintained
   - [ ] Test across page refreshes

4. **Error Handling:**
   - [ ] Test with invalid image
   - [ ] Test with network error
   - [ ] Verify error messages display correctly

---

## 🔧 Troubleshooting

### Issue: CORS Error

**Solution:** Ensure your domain is added to backend CORS configuration:
```python
# In app.py
origins = [
    "https://easternservices.pk",
    "https://www.easternservices.pk"
]
```

### Issue: Slow Response

**Solution:** HF Spaces has cold start delay (~30-60s). Consider:
- Using UptimeRobot to keep API warm
- Adding loading indicator
- Implementing timeout handling

### Issue: Image Upload Fails

**Solution:** Check:
- Image size < 10MB
- Image format is supported (JPG, PNG)
- Base64 encoding is correct

---

## 📊 Expected Response Times

| Operation | Time |
|-----------|------|
| Text Chat | 1-3 seconds |
| Image Analysis | 3-5 seconds |
| TTS Generation | 1-2 seconds |
| Cold Start | 30-60 seconds (first request) |

---

## 🚀 Go Live

Once integrated and tested:

1. Update API_BASE_URL to production
2. Test on staging environment
3. Monitor error logs
4. Deploy to production

---

## 📞 Support

For integration support:
- Email: abdullahnaseem4802@gmail.com
- API Docs: See `API_DOCUMENTATION.md`
- Backend Code: See `app.py`

---

**Your Eastern Services chatbot is ready to integrate!** 🎉
