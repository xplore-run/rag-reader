import { ChatContainer } from './components/ChatContainer';
import { LoginScreen } from './components/LoginScreen';
import { AuthProvider, useAuth } from './contexts/AuthContext';

function AppContent() {
  const { isAuthenticated, login, error } = useAuth();

  if (!isAuthenticated) {
    return <LoginScreen onLogin={login} error={error} />;
  }

  return <ChatContainer />;
}

export function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
