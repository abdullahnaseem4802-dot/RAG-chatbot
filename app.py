"""
Eastern Services RAG Chatbot - FastAPI Application
FIXED VERSION - Faster startup, better error handling
"""

import os
import time
import json
import uuid
import re
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ============================================================================
# GLOBAL VARIABLES - Initialize as None
# ============================================================================
chatbot_initialized = False
initialization_error = None
redis_client = None
engine = None
SessionLocal = None
ConversationHistory = None
cohere_client = None
llm = None
pest_detector = None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    language: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    language: str

class WelcomeResponse(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str
    language: str = "ur"

class ImageAnalysisRequest(BaseModel):
    image_data: str
    question: Optional[str] = None
    session_id: Optional[str] = None
    language: Optional[str] = "english"

class ImageAnalysisResponse(BaseModel):
    analysis: str
    pest_identified: Optional[str] = None
    treatment_recommendation: str
    session_id: str
    language: str

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================
def init_databases():
    """Initialize Redis and Supabase connections."""
    global redis_client, engine, SessionLocal, ConversationHistory
    
    SUPABASE_URI = os.getenv('SUPABASE_URI')
    REDIS_URI = os.getenv('REDIS_URI')
    
    print("[*] Connecting to databases...")
    
    # Redis - with timeout
    try:
        import redis
        redis_uri_clean = REDIS_URI.replace('REDIS_URI=', '').strip() if REDIS_URI else ""
        if redis_uri_clean:
            redis_client = redis.from_url(redis_uri_clean, decode_responses=True, socket_connect_timeout=5)
            redis_client.ping()
            print("[OK] Redis connected!")
        else:
            print("[WARN] Redis URI not set")
            redis_client = None
    except Exception as e:
        print(f"[WARN] Redis failed: {e}")
        redis_client = None
    
    # Supabase
    try:
        from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, text
        from sqlalchemy.orm import declarative_base, sessionmaker
        
        if SUPABASE_URI:
            engine = create_engine(SUPABASE_URI, pool_pre_ping=True, pool_recycle=3600, pool_timeout=10)
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1;"))
                print("[OK] Supabase connected!")
                
                # Enable PGVector
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    conn.commit()
                except:
                    pass
            
            Base = declarative_base()
            
            class ConversationHistoryModel(Base):
                __tablename__ = 'conversation_history'
                id = Column(Integer, primary_key=True, autoincrement=True)
                session_id = Column(String(100), index=True)
                user_question = Column(Text)
                bot_response = Column(Text)
                language = Column(String(50))
                timestamp = Column(DateTime, default=datetime.utcnow)
                meta_data = Column(Text)
            
            ConversationHistory = ConversationHistoryModel
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine)
            print("[OK] Database tables ready!")
        else:
            print("[WARN] Supabase URI not set")
            engine = None
            
    except Exception as e:
        print(f"[WARN] Supabase failed: {e}")
        engine = None

def init_ai_clients():
    """Initialize Cohere, Groq, and YOLOv8."""
    global cohere_client, llm, pest_detector
    
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    print("[*] Initializing AI clients...")
    
    # Cohere
    try:
        import cohere
        cohere_client = cohere.Client(COHERE_API_KEY)
        print("[OK] Cohere initialized!")
    except Exception as e:
        print(f"[WARN] Cohere failed: {e}")
        cohere_client = None
    
    # Groq LLM
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0.0)
        print("[OK] Groq LLM initialized!")
    except Exception as e:
        print(f"[WARN] Groq failed: {e}")
        llm = None
    
    # YOLOv8 Pest Detector
    try:
        from yolov8_detector import PestDetector
        pest_detector = PestDetector('models/best.pt')
        if pest_detector.is_available():
            print("[OK] YOLOv8 Pest Detector initialized!")
        else:
            print("[WARN] YOLOv8 model not loaded")
            pest_detector = None
    except Exception as e:
        print(f"[WARN] YOLOv8 failed: {e}")
        pest_detector = None

def check_embeddings():
    """Check embeddings count."""
    global engine
    if not engine:
        return 0
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1024),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """))
            conn.commit()
            result = conn.execute(text("SELECT COUNT(*) FROM document_embeddings;"))
            count = result.fetchone()[0]
            print(f"[OK] Found {count} embeddings")
            return count
    except Exception as e:
        print(f"[WARN] Embeddings check failed: {e}")
        return 0

# ============================================================================
# LANGUAGE DETECTION
# ============================================================================
def detect_language(text):
    """Detect language."""
    if not text:
        return 'english'
    text_lower = text.lower()
    
    # Urdu script
    if re.search(r'[\u0600-\u06FF]', text):
        return 'urdu'
    
    # Roman Urdu keywords
    urdu_keywords = ['aapka', 'hain', 'kia', 'hai', 'karna', 'hoga', 'meray', 'ghar', 
                     'deemak', 'rate', 'mein', 'ke', 'ka', 'ki', 'kitni', 'kitna']
    urdu_count = sum(1 for k in urdu_keywords if k in text_lower)
    
    if urdu_count >= 2:
        return 'roman_urdu'
    
    return 'english'

# ============================================================================
# RETRIEVAL FUNCTION
# ============================================================================
def retrieve_similar_chunks(query, k=3):
    """Retrieve similar chunks from PGVector."""
    global cohere_client, engine
    
    if not cohere_client or not engine:
        return []
    
    try:
        # Generate embedding
        query_response = cohere_client.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type='search_query'
        )
        query_embedding = query_response.embeddings[0]
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Search
        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            cursor.execute("""
                SELECT chunk_text, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM document_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding_str, query_embedding_str, k))
            
            rows = cursor.fetchall()
            cursor.close()
            
            documents = []
            for row in rows:
                documents.append({
                    'page_content': row[0],
                    'metadata': row[1] if isinstance(row[1], dict) else json.loads(row[1]) if row[1] else {},
                    'similarity': row[2]
                })
            
            return documents
        finally:
            raw_conn.close()
    except Exception as e:
        print(f"[FAIL] Retrieval error: {e}")
        return []

# ============================================================================
# RAG FUNCTION
# ============================================================================
def get_rag_response(question, language):
    """Get RAG response."""
    global llm
    
    if not llm:
        return get_fallback_response(language)
    
    docs = retrieve_similar_chunks(question, k=3)
    
    if not docs:
        return get_fallback_response(language)
    
    context = "\n\n".join([doc['page_content'] for doc in docs])
    
    # Prompts
    if language == 'urdu':
        prompt_text = f"""آپ Eastern Services pest control bot ہیں۔
Context: {context}
Question: {question}
صرف اردو میں جواب دیں۔ Contact: +92 336 1101234
جواب:"""
    elif language == 'roman_urdu':
        prompt_text = f"""Aap Eastern Services pest control bot hain.
Context: {context}
Question: {question}
Sirf Roman Urdu mein jawab do. Contact: +92 336 1101234
Jawab:"""
    else:
        prompt_text = f"""You are Eastern Services pest control bot.
Context: {context}
Question: {question}
Answer in English. Contact: +92 336 1101234
Answer:"""
    
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = PromptTemplate(template=prompt_text, input_variables=[])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        return response.strip()
    except Exception as e:
        print(f"[FAIL] RAG error: {e}")
        return get_fallback_response(language)

def get_fallback_response(language):
    """Fallback response."""
    if language == 'urdu':
        return """معذرت، میرے پاس اس بارے میں معلومات نہیں۔
ہماری خدمات: دیمک کنٹرول، کھٹمل کا علاج، کیڑوں کا کنٹرول
📞 +92 336 1101234"""
    elif language == 'roman_urdu':
        return """Maafi, mere paas is baray mein info nahi hai.
Hamari services: Termite control, Bed bug treatment, Pest control
📞 +92 336 1101234"""
    else:
        return """I don't have specific information about that.
Our services: Termite control, Bed bug treatment, General pest control
📞 +92 336 1101234"""

# ============================================================================
# MEMORY FUNCTIONS
# ============================================================================
def save_to_memory(question, answer, language, session_id):
    """Save conversation."""
    global redis_client, SessionLocal, ConversationHistory
    
    # Redis
    if redis_client:
        try:
            key = f"chat:{session_id}:history"
            entry = json.dumps({'question': question, 'answer': answer, 'language': language, 'timestamp': time.time()})
            redis_client.lpush(key, entry)
            redis_client.ltrim(key, 0, 9)
            redis_client.expire(key, 3600)
        except:
            pass
    
    # Supabase
    if SessionLocal and ConversationHistory:
        try:
            db = SessionLocal()
            conversation = ConversationHistory(
                session_id=session_id,
                user_question=question,
                bot_response=answer,
                language=language,
                meta_data=json.dumps({'source': 'hf_spaces'})
            )
            db.add(conversation)
            db.commit()
            db.close()
        except:
            pass

# ============================================================================
# LIFESPAN (STARTUP/SHUTDOWN)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global chatbot_initialized, initialization_error
    
    print("=" * 50)
    print("[*] Starting Eastern Services Chatbot...")
    print("=" * 50)
    
    try:
        # Check env vars
        required_vars = ['GROQ_API_KEY', 'COHERE_API_KEY']
        missing = [v for v in required_vars if not os.getenv(v)]
        
        if missing:
            print(f"[WARN] Missing optional vars: {missing}")
        
        # Initialize (with error handling for each)
        try:
            init_databases()
        except Exception as e:
            print(f"[WARN] Database init failed: {e}")
        
        try:
            init_ai_clients()
        except Exception as e:
            print(f"[WARN] AI clients init failed: {e}")
        
        try:
            check_embeddings()
        except Exception as e:
            print(f"[WARN] Embeddings check failed: {e}")
        
        # Mark as initialized even if some components failed
        chatbot_initialized = True
        print("=" * 50)
        print("[OK] Chatbot started successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"[FAIL] Startup error: {e}")
        initialization_error = str(e)
        chatbot_initialized = True  # Still allow health checks
    
    yield  # App runs here
    
    print("[*] Shutting down...")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Eastern Services RAG Chatbot API",
    description="Multilingual RAG chatbot for pest control",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow ALL origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False for wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Eastern Services Pest Control API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "welcome": "/welcome",
            "chat": "/chat",
            "image_analysis": "/api/analyze-image",
            "tts": "/api/tts"
        },
        "documentation": "See API_DOCUMENTATION.md"
    }

@app.get("/health")
async def health_check():
    """Health check - always responds quickly."""
    return JSONResponse(content={
        "status": "healthy",
        "initialized": chatbot_initialized,
        "error": initialization_error,
        "components": {
            "redis": redis_client is not None,
            "database": engine is not None,
            "cohere": cohere_client is not None,
            "llm": llm is not None,
            "yolov8": pest_detector is not None
        }
    })

@app.get("/welcome")
async def get_welcome():
    """Welcome message."""
    return {"message": "السلام وعلیکم! 👋\n\nWelcome to Eastern Services Pest Control AI Assistant.\n\nHow can I help you today?"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint."""
    session_id = request.session_id or str(uuid.uuid4())[:8]
    
    try:
        language = request.language or detect_language(request.question)
        response = get_rag_response(request.question, language)
        
        # Save async (don't wait)
        try:
            save_to_memory(request.question, response, language, session_id)
        except:
            pass
        
        return {
            "response": response,
            "session_id": session_id,
            "language": language
        }
        
    except Exception as e:
        print(f"[FAIL] Chat error: {e}")
        language = detect_language(request.question) if request.question else 'english'
        return {
            "response": get_fallback_response(language),
            "session_id": session_id,
            "language": language
        }

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """TTS endpoint."""
    try:
        from gtts import gTTS
        from fastapi.responses import StreamingResponse
        import io
        
        lang_map = {'ur-PK': 'ur', 'en-US': 'en', 'ur': 'ur', 'en': 'en'}
        lang = lang_map.get(request.language, 'ur')
        
        tts = gTTS(text=request.text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    """Image analysis endpoint."""
    global pest_detector
    
    session_id = request.session_id or str(uuid.uuid4())[:8]
    language = request.language or 'english'
    
    if not pest_detector:
        return {
            "analysis": "Image analysis is currently unavailable. Please describe the pest you're seeing.",
            "pest_identified": None,
            "treatment_recommendation": "Contact us: +92 336 1101234",
            "session_id": session_id,
            "language": language
        }
    
    try:
        result = pest_detector.detect_pest(request.image_data, confidence_threshold=0.5)
        
        if not result.get('success') or not result.get('pest_detected'):
            return {
                "analysis": "No pests detected. Please upload a clearer image.",
                "pest_identified": None,
                "treatment_recommendation": "",
                "session_id": session_id,
                "language": language
            }
        
        pest_name = result['pest_name']
        
        # Get treatment
        try:
            from mock_rag_local import get_mock_treatment
            treatment = get_mock_treatment(pest_name, language)
        except:
            treatment = f"Contact us for {pest_name} treatment: +92 336 1101234"
        
        analysis = f"🔎 Detected: {pest_name}\n\n{treatment}"
        
        return {
            "analysis": analysis,
            "pest_identified": pest_name,
            "treatment_recommendation": treatment,
            "session_id": session_id,
            "language": language
        }
        
    except Exception as e:
        print(f"[FAIL] Image analysis error: {e}")
        return {
            "analysis": f"Error analyzing image: {str(e)}",
            "pest_identified": None,
            "treatment_recommendation": "Contact us: +92 336 1101234",
            "session_id": session_id,
            "language": language
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)