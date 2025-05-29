# RAG Chatbot Web Application

A modern web interface for the RAG (Retrieval-Augmented Generation) chatbot, built with Preact, TypeScript, Vite, and Tailwind CSS.

## Features

- 🚀 **Fast & Modern**: Built with Preact for optimal performance
- 🎨 **Beautiful UI**: Tailwind CSS with dark mode support
- 🔐 **Authentication**: JWT-based login system
- 💬 **Real-time Chat**: Seamless conversation interface
- 📄 **Source Citations**: View relevant document sources for each answer
- 📊 **Status Monitoring**: Real-time API health and document count
- 🔄 **Session Management**: Automatic session tracking
- ⚡ **Hot Module Replacement**: Fast development with Vite

## Prerequisites

- Node.js 18+ and npm
- Python RAG API running on http://localhost:8000
- Indexed documents in the vector store

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The application will be available at http://localhost:3000

## Project Structure

```
web-app/
├── src/
│   ├── components/          # UI components
│   │   ├── ChatContainer.tsx    # Main chat interface
│   │   ├── Message.tsx          # Message display component
│   │   ├── MessageInput.tsx     # Input form component
│   │   ├── LoadingIndicator.tsx # Loading animation
│   │   └── StatusBar.tsx        # API status display
│   ├── services/            # API integration
│   │   └── api.ts              # API client service
│   ├── types/               # TypeScript definitions
│   │   └── api.ts              # API request/response types
│   ├── app.tsx              # Main app component
│   ├── main.tsx             # Application entry point
│   └── tailwind.css         # Global styles
├── public/                  # Static assets
├── index.html              # HTML template
├── vite.config.ts          # Vite configuration
├── tailwind.config.js      # Tailwind configuration
└── tsconfig.json           # TypeScript configuration
```

## Configuration

### API Proxy

The Vite development server is configured to proxy API requests to the Python backend. All requests to `/api/*` are forwarded to `http://localhost:8000`.

### Environment Variables

Create a `.env` file if you need to customize the API endpoint:

```env
VITE_API_URL=http://localhost:8000
```

## Usage

### Starting the Application

1. First, ensure the Python API is running:
```bash
cd /Users/vk/projects/chatbot-tech-rag
source venv/bin/activate
python -m src.api.run_api
```

2. Then start the web application:
```bash
cd web-app
npm run dev
```

### Authentication

The application requires authentication to access the chat interface. 

**Default Credentials:**
- Email: `Check the env file for chatbot project`
- Password: `Check the env file for chatbot project`

These credentials are configured in the `.env` file on the backend.

### Using the Chat Interface

1. Login with the provided credentials
2. Type your question in the input field at the bottom
3. Press Enter or click the send button
4. View the AI response with confidence score
5. Click "Sources" to see the relevant documents used
6. Toggle "Show sources" to hide/show source citations
7. Click "Clear chat" to start a new conversation
8. Click "Logout" to sign out

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Features in Detail

### Message Interface
- User messages appear on the right in blue
- Assistant messages appear on the left with gray background
- Confidence scores are displayed for each response
- Timestamps show when each message was sent

### Source Citations
- Each assistant response can include source documents
- Sources show the document name and relevance score
- Expandable view shows the actual text chunks used

### Error Handling
- Connection errors are displayed with helpful messages
- Loading states prevent duplicate submissions
- Automatic retry for failed health checks

### Dark Mode
- Automatically respects system preferences
- Smooth transitions between light and dark themes
- Optimized contrast for readability

## API Integration

The web app communicates with these API endpoints:

- `POST /auth/login` - User authentication
- `GET /health` - Check API status and document count (public)
- `POST /ask` - Submit questions and receive answers (protected)
- `GET /sessions/{id}` - Retrieve session history (protected)
- `DELETE /sessions/{id}` - Clear session data (protected)
- `GET /stats` - Get usage statistics (protected)

All protected endpoints require a valid JWT token in the Authorization header.

## Development

### Adding New Components

1. Create component in `src/components/`
2. Define props interface
3. Use Tailwind classes for styling
4. Import in parent component

### Modifying API Client

Edit `src/services/api.ts` to add new API methods or modify existing ones.

### TypeScript Types

All API types are defined in `src/types/api.ts`. Update these when API changes.

## Troubleshooting

### API Connection Failed
- Ensure Python API is running on port 8000
- Check console for CORS errors
- Verify proxy configuration in vite.config.ts

### No Documents Indexed
- Run the indexing script: `python scripts/index_documents.py`
- Check that PDFs exist in train-data/
- Verify vector store initialization

### Build Errors
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check TypeScript errors: `npx tsc --noEmit`
- Ensure all imports are correct

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is part of the RAG Chatbot system. See the main project README for license information.