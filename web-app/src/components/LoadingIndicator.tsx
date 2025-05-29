import { FunctionalComponent } from 'preact';

export const LoadingIndicator: FunctionalComponent = () => {
  return (
    <div class="flex justify-start mb-4">
      <div class="max-w-[80%] mr-12">
        <div class="bg-gray-100 dark:bg-gray-800 rounded-lg px-4 py-3">
          <div class="text-sm mb-1 text-gray-600 dark:text-gray-400">Assistant</div>
          <div class="flex items-center gap-1">
            <div class="w-2 h-2 bg-gray-400 dark:bg-gray-600 rounded-full animate-bounce" style="animation-delay: 0ms" />
            <div class="w-2 h-2 bg-gray-400 dark:bg-gray-600 rounded-full animate-bounce" style="animation-delay: 150ms" />
            <div class="w-2 h-2 bg-gray-400 dark:bg-gray-600 rounded-full animate-bounce" style="animation-delay: 300ms" />
          </div>
        </div>
      </div>
    </div>
  );
};