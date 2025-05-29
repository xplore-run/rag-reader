// API Client Service

import type { 
  QuestionRequest, 
  AnswerResponse, 
  HealthResponse, 
  ChatSession, 
  StatsResponse,
  ApiError,
  LoginRequest,
  LoginResponse
} from '../types/api';

const API_BASE = '/api';

class ApiClient {
  private authToken: string | null = null;

  setAuthToken(token: string | null) {
    this.authToken = token;
  }

  private async fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options?.headers,
    };

    // Add auth header if token exists
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    const response = await fetch(`${API_BASE}${url}`, {
      headers,
      ...options,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw {
        detail: error.detail || response.statusText,
        status: response.status,
      } as ApiError;
    }

    return response.json();
  }

  async checkHealth(): Promise<HealthResponse> {
    return this.fetchJson<HealthResponse>('/health');
  }

  async askQuestion(request: QuestionRequest): Promise<AnswerResponse> {
    return this.fetchJson<AnswerResponse>('/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getSession(sessionId: string): Promise<ChatSession> {
    return this.fetchJson<ChatSession>(`/sessions/${sessionId}`);
  }

  async deleteSession(sessionId: string): Promise<{ message: string }> {
    return this.fetchJson<{ message: string }>(`/sessions/${sessionId}`, {
      method: 'DELETE',
    });
  }

  async getStats(): Promise<StatsResponse> {
    return this.fetchJson<StatsResponse>('/stats');
  }

  async verifyIndex(): Promise<{
    document_count: number;
    status: string;
    test_search: { success: boolean; result_count?: number; error?: string } | null;
  }> {
    return this.fetchJson('/index/verify', {
      method: 'POST',
    });
  }

  async login(email: string, password: string): Promise<LoginResponse> {
    const response = await this.fetchJson<LoginResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    
    // Automatically set the token after successful login
    this.authToken = response.access_token;
    
    return response;
  }
}

export const apiClient = new ApiClient();

// Utility function to generate session ID
export function generateSessionId(): string {
  return crypto.randomUUID();
}