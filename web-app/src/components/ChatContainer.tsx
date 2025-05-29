import { FunctionalComponent } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { Message } from './Message';
import { MessageInput } from './MessageInput';
import { LoadingIndicator } from './LoadingIndicator';
import { StatusBar } from './StatusBar';
import { apiClient, generateSessionId } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import type { AnswerResponse, HealthResponse } from '../types/api';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  confidence?: number;
  sources?: AnswerResponse['sources'];
  timestamp: number;
}

export const ChatContainer: FunctionalComponent = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId] = useState(() => generateSessionId());
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [showSources, setShowSources] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { logout } = useAuth();

  // Check health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkHealth = async () => {
    try {
      const healthData = await apiClient.checkHealth();
      setHealth(healthData);
      if (healthData.document_count === 0) {
        setError('No documents indexed. Please run the indexing script first.');
      }
    } catch (err) {
      setError('Failed to connect to the API. Make sure the backend is running.');
    }
  };

  const sendMessage = async (content: string) => {
    setError(null);
    
    // Add user message
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now() / 1000,
    };
    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);
    
    try {
      const response = await apiClient.askQuestion({
        question: content,
        session_id: sessionId,
        include_sources: showSources,
        top_k: 5,
      });
      
      // Add assistant message
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.answer,
        confidence: response.confidence,
        sources: response.sources,
        timestamp: response.timestamp,
      };
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (err: any) {
      setError(err.detail || 'Failed to get response from the chatbot');
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    apiClient.deleteSession(sessionId).catch(() => {});
  };

  return (
    <div class="flex flex-col h-screen bg-white dark:bg-gray-900">
      {/* Header */}
      <header class="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
        <div class="px-4 py-3">
          <div class="flex items-center justify-between">
            <h1 class="text-xl font-semibold text-gray-900 dark:text-gray-100">
              RAG Chatbot
            </h1>
            <div class="flex items-center gap-4">
              <label class="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <input
                  type="checkbox"
                  checked={showSources}
                  onChange={(e) => setShowSources(e.currentTarget.checked)}
                  class="rounded"
                />
                Show sources
              </label>
              <button
                onClick={clearChat}
                class="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
              >
                Clear chat
              </button>
              <button
                onClick={logout}
                class="text-sm text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-200"
              >
                Logout
              </button>
            </div>
          </div>
          {health && <StatusBar health={health} />}
        </div>
      </header>

      {/* Messages */}
      <div class="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 && !error && (
          <div class="text-center text-gray-500 dark:text-gray-400 mt-12">
            <p class="text-lg mb-2">Welcome to RAG Chatbot!</p>
            <p class="text-sm">Ask questions about your indexed documents.</p>
          </div>
        )}
        
        {error && (
          <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
            <p class="text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}
        
        {messages.map(message => (
          <Message
            key={message.id}
            role={message.role}
            content={message.content}
            confidence={message.confidence}
            sources={message.sources}
            timestamp={message.timestamp}
            showSources={showSources && message.role === 'assistant'}
          />
        ))}
        
        {isLoading && <LoadingIndicator />}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <MessageInput
        onSubmit={sendMessage}
        disabled={isLoading || health?.document_count === 0}
        placeholder={
          health?.document_count === 0 
            ? "No documents indexed..." 
            : "Ask a question about your documents..."
        }
      />
    </div>
  );
};