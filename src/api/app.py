"""FastAPI REST API for the RAG chatbot."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid
from datetime import datetime, timedelta

from ..embeddings.factory import EmbeddingFactory
from ..vector_store.factory import VectorStoreFactory
from ..llm.factory import LLMFactory
from ..rag.retriever import RAGRetriever
from ..rag.generator import RAGGenerator
from ..utils.config import get_settings
from ..utils.logger import setup_logging, logger
from ..auth.models import LoginRequest, LoginResponse
from ..auth.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_HOURS
from ..auth.dependencies import get_current_user


# Request/Response models
class QuestionRequest(BaseModel):
    """Question request model."""
    question: str = Field(..., description="The question to ask")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    include_sources: Optional[bool] = Field(True, description="Include source documents")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")


class SourceDocument(BaseModel):
    """Source document information."""
    document: str
    relevance_score: float
    chunks: List[Dict[str, Any]]


class AnswerResponse(BaseModel):
    """Answer response model."""
    answer: str
    confidence: float
    sources: Optional[List[SourceDocument]]
    session_id: str
    timestamp: float
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    document_count: int
    model_info: Dict[str, str]
    timestamp: float


class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str
    created_at: float
    questions: List[Dict[str, Any]]


# Initialize FastAPI app
app = FastAPI(
    title="Technical Documentation RAG API",
    description="API for querying technical documentation using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for components
chatbot_components = None
chat_sessions = {}  # In-memory session storage


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot components on startup."""
    global chatbot_components
    
    settings = get_settings()
    
    # Set up logging
    setup_logging(
        log_level=settings.log_level,
        log_file=settings.log_file
    )
    
    logger.info("Initializing RAG API components...")
    
    # Initialize components
    embedding_model = EmbeddingFactory.create(
        model_type=settings.embedding.type,
        model_name=settings.embedding.model_name,
        device=settings.embedding.device
    )
    
    vector_store = VectorStoreFactory.create(
        store_type=settings.vector_store.type,
        collection_name=settings.vector_store.collection_name,
        persist_directory=settings.vector_store.persist_directory
    )
    
    llm_client = LLMFactory.create(
        provider=settings.llm.provider,
        model=settings.llm.model
    )
    
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        rerank=settings.rag.rerank,
        include_neighbors=settings.rag.include_neighbors
    )
    
    generator = RAGGenerator(
        llm_client=llm_client,
        prompt_template=settings.rag.prompt_template,
        max_context_length=3000,
        include_sources=True
    )
    
    chatbot_components = {
        'settings': settings,
        'vector_store': vector_store,
        'retriever': retriever,
        'generator': generator,
        'embedding_model': embedding_model,
        'llm_client': llm_client
    }
    
    logger.info("RAG API initialized successfully")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Technical Documentation RAG API",
        "docs": "/docs",
        "health": "/health",
        "login": "/auth/login"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not chatbot_components:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    doc_count = chatbot_components['vector_store'].count()
    
    return HealthResponse(
        status="healthy" if doc_count > 0 else "no_documents",
        document_count=doc_count,
        model_info={
            "embedding_model": chatbot_components['settings'].embedding.model_name or "default",
            "llm_model": chatbot_components['settings'].llm.model or "default",
            "vector_store": chatbot_components['settings'].vector_store.type
        },
        timestamp=time.time()
    )


@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint."""
    if not authenticate_user(request.email, request.password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": request.email},
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_HOURS * 3600  # Convert to seconds
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    current_user: str = Depends(get_current_user)
):
    """Ask a question and get an answer."""
    if not chatbot_components:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if chatbot_components['vector_store'].count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    start_time = time.time()
    
    try:
        # Retrieve relevant documents
        retrieval_results = chatbot_components['retriever'].retrieve(
            query=request.question,
            k=request.top_k
        )
        
        # Generate answer
        answer = chatbot_components['generator'].generate(
            question=request.question,
            retrieval_results=retrieval_results
        )
        
        # Create session if needed
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatSession(
                session_id=session_id,
                created_at=time.time(),
                questions=[]
            )
        
        # Add to session history
        chat_sessions[session_id].questions.append({
            'question': request.question,
            'answer': answer.answer,
            'confidence': answer.confidence,
            'timestamp': time.time()
        })
        
        # Prepare response
        processing_time = time.time() - start_time
        
        response = AnswerResponse(
            answer=answer.answer,
            confidence=answer.confidence,
            sources=[SourceDocument(**source) for source in answer.sources] if request.include_sources else None,
            session_id=session_id,
            timestamp=time.time(),
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get chat session history."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return chat_sessions[session_id]


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: str = Depends(get_current_user)
):
    """Delete a chat session."""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    return {"message": "Session deleted"}


@app.get("/stats")
async def get_statistics(
    current_user: str = Depends(get_current_user)
):
    """Get API statistics."""
    if not chatbot_components:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    total_questions = sum(len(session.questions) for session in chat_sessions.values())
    
    avg_confidence = 0
    if total_questions > 0:
        all_confidences = []
        for session in chat_sessions.values():
            all_confidences.extend([q['confidence'] for q in session.questions])
        avg_confidence = sum(all_confidences) / len(all_confidences)
    
    return {
        "document_count": chatbot_components['vector_store'].count(),
        "active_sessions": len(chat_sessions),
        "total_questions": total_questions,
        "average_confidence": avg_confidence,
        "models": {
            "embedding": chatbot_components['settings'].embedding.model_name or "default",
            "llm": chatbot_components['settings'].llm.model or "default"
        }
    }


@app.post("/index/verify")
async def verify_index(
    current_user: str = Depends(get_current_user)
):
    """Verify the vector store index."""
    if not chatbot_components:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    doc_count = chatbot_components['vector_store'].count()
    
    # Test search
    test_results = None
    if doc_count > 0:
        try:
            test_query = "test query"
            query_embedding = chatbot_components['embedding_model'].embed_text(test_query)
            results = chatbot_components['vector_store'].search(
                query_embedding=query_embedding,
                k=1
            )
            test_results = {
                "success": True,
                "result_count": len(results)
            }
        except Exception as e:
            test_results = {
                "success": False,
                "error": str(e)
            }
    
    return {
        "document_count": doc_count,
        "status": "ready" if doc_count > 0 else "empty",
        "test_search": test_results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)