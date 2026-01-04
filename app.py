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

# Import document loading and embedding functions from chatbot_core
from chatbot_core import (
    load_documents_from_github,
    create_chunks_from_documents,
    generate_and_store_embeddings,
    setup_pgvector_table,
    GITHUB_REPO_URL,
    DATA_FILES,
    transliterate_urdu_to_roman,
    URDU_TO_ROMAN
)

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
hf_embeddings = None  # HuggingFace embeddings model (replaces Cohere for queries)

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
    """Initialize HuggingFace Embeddings, Groq, and YOLOv8."""
    global cohere_client, llm, pest_detector, hf_embeddings
    
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    print("[*] Initializing AI clients...")
    
    # HuggingFace Embeddings (FREE, NO RATE LIMITS - runs locally)
    try:
        from sentence_transformers import SentenceTransformer
        hf_embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("[OK] HuggingFace Embeddings initialized (all-MiniLM-L6-v2)!")
    except Exception as e:
        print(f"[WARN] HuggingFace Embeddings failed: {e}")
        hf_embeddings = None
    
    # Cohere (kept as backup, but not used for queries anymore)
    try:
        import cohere
        cohere_client = cohere.Client(COHERE_API_KEY)
        print("[OK] Cohere initialized (backup only)!")
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

def check_embeddings_hf():
    """Check HuggingFace embeddings count (384 dimensions)."""
    global engine
    if not engine:
        return 0
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            # Check if HF embeddings table exists and has data
            result = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'document_embeddings_hf';
            """))
            table_exists = result.fetchone()[0] > 0
            
            if not table_exists:
                print("[*] Creating HuggingFace embeddings table (384 dimensions)...")
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS document_embeddings_hf (
                        id SERIAL PRIMARY KEY,
                        chunk_text TEXT NOT NULL,
                        embedding vector(384),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """))
                conn.commit()
                print("[OK] HuggingFace embeddings table created!")
                return 0
            
            result = conn.execute(text("SELECT COUNT(*) FROM document_embeddings_hf;"))
            count = result.fetchone()[0]
            print(f"[OK] Found {count} HuggingFace embeddings")
            return count
    except Exception as e:
        print(f"[WARN] HF Embeddings check failed: {e}")
        return 0

def load_and_generate_embeddings_hf():
    """Load documents from GitHub and generate embeddings using HuggingFace (FREE, NO RATE LIMITS)."""
    global engine, hf_embeddings
    
    print("[*] load_and_generate_embeddings_hf() called")
    print(f"[*] engine is None: {engine is None}")
    print(f"[*] hf_embeddings is None: {hf_embeddings is None}")
    
    if not engine or not hf_embeddings:
        print("[WARN] Cannot generate embeddings - missing database or HuggingFace model")
        return False
    
    try:
        print("[*] Loading documents from GitHub...")
        documents = load_documents_from_github(GITHUB_REPO_URL, DATA_FILES)
        
        if not documents:
            print("[WARN] No documents loaded from GitHub")
            return False
        
        print(f"[OK] Loaded {len(documents)} documents")
        
        # Create chunks
        print("[*] Creating chunks...")
        chunks = create_chunks_from_documents(documents)
        
        if not chunks:
            print("[WARN] No chunks created")
            return False
        
        print(f"[OK] Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Generate and store embeddings using HuggingFace
        print("[*] Generating embeddings with HuggingFace (this may take a minute)...")
        
        from sqlalchemy import text
        batch_size = 50
        total_embedded = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk['text'] for chunk in batch]
            
            print(f"   Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
            
            # Generate embeddings with HuggingFace (FREE, NO RATE LIMITS!)
            embeddings_batch = hf_embeddings.encode(batch_texts).tolist()
            
            # Store in database
            raw_conn = engine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                
                for chunk, embedding in zip(batch, embeddings_batch):
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    metadata_json = json.dumps(chunk['metadata'])
                    
                    cursor.execute("""
                        INSERT INTO document_embeddings_hf (chunk_text, embedding, metadata)
                        VALUES (%s, %s::vector, %s::jsonb)
                    """, (chunk['text'], embedding_str, metadata_json))
                    
                    total_embedded += 1
                
                raw_conn.commit()
                cursor.close()
            finally:
                raw_conn.close()
            
            print(f"   [OK] Embedded {total_embedded}/{len(chunks)} chunks")
        
        print(f"[OK] All {total_embedded} embeddings generated and stored with HuggingFace!")
        return True
            
    except Exception as e:
        import traceback
        print(f"[WARN] Error in load_and_generate_embeddings: {e}")
        print(f"[WARN] Traceback: {traceback.format_exc()}")
        return False

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
    """Retrieve similar chunks from PGVector using HuggingFace embeddings (FREE, NO RATE LIMITS)."""
    global hf_embeddings, engine
    
    if not hf_embeddings or not engine:
        print("[WARN] Missing hf_embeddings or engine for retrieval")
        return []
    
    try:
        # Transliterate Urdu to Roman/English for better matching
        search_query = transliterate_urdu_to_roman(query)
        
        # Always check for Urdu keywords and add English equivalents for better matching
        urdu_keywords_map = {
            'ریٹ': 'rate price cost',
            'قیمت': 'price rate cost',
            'دیمک': 'termite deemak treatment wood',
            'کھٹمل': 'bed bug bedbug treatment mattress',
            'مچھر': 'mosquito control dengue',
            'چوہے': 'rat rodent control mouse',
            'چوہا': 'rat rodent control mouse',
            'سروس': 'service services offer',
            'خدمات': 'service services offer',
            'علاج': 'treatment control service',
            'گارنٹی': 'guarantee warranty',
            'ضمانت': 'guarantee warranty',
            'بکنگ': 'booking appointment schedule',
            'کتنے': 'how many days time duration',
            'کتنا': 'how much price cost',
            'کتنی': 'how much price cost',
            'دن': 'days time duration',
            'وقت': 'time duration',
            'کب': 'when time',
            'کیسے': 'how process method',
            'کیا': 'what which',
            'کون': 'which what',
            'گھر': 'home house residential',
            'مکان': 'home house residential',
            'دفتر': 'office commercial',
            'کاروبار': 'business commercial',
            'چیونٹی': 'ant ants control',
            'کاکروچ': 'cockroach roach control',
            'کیڑے': 'pest insects bugs',
            'سپرے': 'spray treatment chemical',
            'کیمیکل': 'chemical safe treatment',
            'محفوظ': 'safe safety',
            'فیومیگیشن': 'fumigation treatment',
        }
        enhanced_terms = []
        for urdu_word, english_terms in urdu_keywords_map.items():
            if urdu_word in query:
                enhanced_terms.append(english_terms)
        
        # Add enhanced terms to search query
        if enhanced_terms:
            search_query = ' '.join(enhanced_terms) + ' ' + search_query
            print(f"[*] Enhanced with keywords: {' '.join(enhanced_terms)}")
        
        print(f"[*] Original query: {query}")
        print(f"[*] Transliterated query: {search_query}")
        
        # Generate embedding using HuggingFace (FREE, NO RATE LIMITS!)
        query_embedding = hf_embeddings.encode(search_query).tolist()
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        print(f"[OK] Generated embedding with HuggingFace (dim={len(query_embedding)})")
        
        # Retrieve more chunks for re-ranking
        retrieval_k = k + 2
        
        # Search
        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            cursor.execute("""
                SELECT chunk_text, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM document_embeddings_hf
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding_str, query_embedding_str, retrieval_k))
            
            rows = cursor.fetchall()
            cursor.close()
            
            documents = []
            for row in rows:
                doc = {
                    'page_content': row[0],
                    'metadata': row[1] if isinstance(row[1], dict) else json.loads(row[1]) if row[1] else {},
                    'similarity': row[2]
                }
                documents.append(doc)
            
            print(f"[*] Retrieved {len(documents)} documents")
            
            # Keyword-based re-ranking for better relevance
            query_lower = search_query.lower()
            
            pest_keywords = {
                'deemak': ['termite', 'deemak', 'white ant', 'wood'],
                'termite': ['termite', 'deemak', 'white ant', 'wood'],
                'khatmal': ['bed bug', 'khatmal', 'bedbug', 'mattress'],
                'bed': ['bed bug', 'khatmal', 'bedbug', 'mattress'],
                'bug': ['bed bug', 'khatmal', 'bedbug', 'mattress'],
                'cockroach': ['cockroach', 'roach', 'kitchen'],
                'mosquito': ['mosquito', 'malaria', 'dengue', 'machar', 'machhar'],
                'machar': ['mosquito', 'malaria', 'dengue', 'machar', 'machhar'],
                'rat': ['rat', 'rodent', 'mouse', 'chuhay', 'chooha'],
                'chuhay': ['rat', 'rodent', 'mouse', 'chuhay', 'chooha'],
                'ant': ['ant', 'chiti', 'sugar ant'],
                'guarantee': ['guarantee', 'warranty', 'guaranteed', 'assurance'],
                'rate': ['rate', 'price', 'cost', 'pricing', 'charges', 'rupees', 'rs', 'qeemat'],
                'price': ['rate', 'price', 'cost', 'pricing', 'charges', 'rupees', 'rs', 'qeemat'],
                'service': ['service', 'services', 'provide', 'offer', 'ilaj', 'treatment'],
                'discount': ['discount', 'offer', 'package', 'deal', 'special'],
                'booking': ['booking', 'book', 'appointment', 'schedule'],
                'appointment': ['booking', 'book', 'appointment', 'schedule'],
            }
            
            query_relevant_keywords = []
            for key, keywords in pest_keywords.items():
                if key in query_lower:
                    query_relevant_keywords.extend(keywords)
            
            query_relevant_keywords = list(set(query_relevant_keywords))
            
            # Re-rank documents based on keyword matches
            if query_relevant_keywords:
                for doc in documents:
                    doc_text_lower = doc['page_content'].lower()
                    keyword_matches = sum(1 for kw in query_relevant_keywords if kw in doc_text_lower)
                    
                    if keyword_matches > 0:
                        boost = min(0.5, keyword_matches * 0.2)
                        doc['similarity'] = min(1.0, doc['similarity'] + boost)
                        doc['keyword_matches'] = keyword_matches
                    else:
                        doc['similarity'] = doc['similarity'] * 0.8
                        doc['keyword_matches'] = 0
                
                documents.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Log top results for debugging
            if documents:
                print(f"[*] Top result similarity: {documents[0]['similarity']:.3f}")
            
            return documents[:k]
        finally:
            raw_conn.close()
    except Exception as e:
        import traceback
        print(f"[FAIL] Retrieval error: {e}")
        print(f"[FAIL] Traceback: {traceback.format_exc()}")
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
            print("[*] Checking HuggingFace embeddings (FREE, NO RATE LIMITS)...")
            # Check and generate HuggingFace embeddings if needed
            embedding_count = check_embeddings_hf()
            print(f"[*] HF Embedding count: {embedding_count}")
            
            if embedding_count == 0:
                print("[*] No HuggingFace embeddings found. Starting document loading and embedding generation...")
                # Try to load and generate embeddings with HuggingFace
                success = load_and_generate_embeddings_hf()
                if success:
                    print("[OK] HuggingFace embeddings generated successfully!")
                else:
                    print("[WARN] HuggingFace embedding generation returned False")
            else:
                print(f"[OK] {embedding_count} HuggingFace embeddings ready for RAG (NO RATE LIMITS!)")
        except Exception as e:
            import traceback
            print(f"[WARN] HF Embeddings check/generation failed: {e}")
            print(f"[WARN] Traceback: {traceback.format_exc()}")
        
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
            "hf_embeddings": hf_embeddings is not None,  # HuggingFace (FREE, NO RATE LIMITS)
            "cohere": cohere_client is not None,  # Backup only
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
