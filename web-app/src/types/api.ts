// API Request/Response Types

export interface QuestionRequest {
  question: string;
  top_k?: number;
  include_sources?: boolean;
  session_id?: string;
}

export interface SourceChunk {
  text: string;
  chunk_id: number;
  metadata?: Record<string, any>;
}

export interface SourceDocument {
  document: string;
  relevance_score: number;
  chunks: SourceChunk[];
}

export interface AnswerResponse {
  answer: string;
  confidence: number;
  sources?: SourceDocument[];
  session_id: string;
  timestamp: number;
  processing_time: number;
}

export interface HealthResponse {
  status: string;
  document_count: number;
  model_info: {
    embedding_model: string;
    llm_model: string;
    vector_store: string;
  };
  timestamp: number;
}

export interface ChatQuestion {
  question: string;
  answer: string;
  confidence: number;
  timestamp: number;
}

export interface ChatSession {
  session_id: string;
  created_at: number;
  questions: ChatQuestion[];
}

export interface StatsResponse {
  document_count: number;
  active_sessions: number;
  total_questions: number;
  average_confidence: number;
  models: {
    embedding: string;
    llm: string;
  };
}

export interface ApiError {
  detail: string;
  status?: number;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}