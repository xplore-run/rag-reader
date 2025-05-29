import { FunctionalComponent } from 'preact';
import { SourceDocument } from '../types/api';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  confidence?: number;
  sources?: SourceDocument[];
  timestamp?: number;
  showSources?: boolean;
}

export const Message: FunctionalComponent<MessageProps> = ({ 
  role, 
  content, 
  confidence, 
  sources, 
  timestamp,
  showSources = false 
}) => {
  const isUser = role === 'user';
  
  return (
    <div class={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div class={`max-w-[80%] ${isUser ? 'ml-12' : 'mr-12'}`}>
        <div class={`rounded-lg px-4 py-3 ${
          isUser 
            ? 'bg-blue-600 text-white' 
            : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
        }`}>
          <div class="text-sm mb-1 opacity-70">
            {isUser ? 'You' : 'Assistant'}
            {confidence !== undefined && (
              <span class="ml-2">
                (Confidence: {(confidence * 100).toFixed(0)}%)
              </span>
            )}
          </div>
          <div class="whitespace-pre-wrap">{content}</div>
        </div>
        
        {showSources && sources && sources.length > 0 && (
          <details class="mt-2 text-sm">
            <summary class="cursor-pointer text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200">
              Sources ({sources.length})
            </summary>
            <div class="mt-2 space-y-2">
              {sources.map((source, idx) => (
                <div key={idx} class="bg-gray-50 dark:bg-gray-900 rounded p-3 border border-gray-200 dark:border-gray-700">
                  <div class="font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {source.document} (Score: {source.relevance_score.toFixed(3)})
                  </div>
                  <div class="text-gray-600 dark:text-gray-400 text-xs">
                    {source.chunks.map((chunk) => chunk.text).join(' ... ')}
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}
        
        {timestamp && (
          <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {new Date(timestamp * 1000).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
};