import { FunctionalComponent } from 'preact';
import type { HealthResponse } from '../types/api';

interface StatusBarProps {
  health: HealthResponse;
}

export const StatusBar: FunctionalComponent<StatusBarProps> = ({ health }) => {
  const isHealthy = health.status === 'healthy';
  
  return (
    <div class="flex items-center gap-4 mt-2 text-xs text-gray-600 dark:text-gray-400">
      <div class="flex items-center gap-1">
        <div class={`w-2 h-2 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-yellow-500'}`} />
        <span>{isHealthy ? 'Connected' : 'No Documents'}</span>
      </div>
      <div>Documents: {health.document_count}</div>
      <div>Model: {health.model_info.llm_model}</div>
      <div>Embeddings: {health.model_info.embedding_model}</div>
    </div>
  );
};