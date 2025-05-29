import { createContext, FunctionalComponent, ComponentChildren } from 'preact';
import { useContext, useState, useEffect } from 'preact/hooks';
import { apiClient } from '../services/api';

interface AuthContextType {
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ComponentChildren;
}

export const AuthProvider: FunctionalComponent<AuthProviderProps> = ({ children }) => {
  const [token, setToken] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check for saved token on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('auth_token');
    if (savedToken) {
      setToken(savedToken);
      apiClient.setAuthToken(savedToken);
    }
  }, []);

  const login = async (email: string, password: string) => {
    setError(null);
    
    try {
      const response = await apiClient.login(email, password);
      
      // Save token
      localStorage.setItem('auth_token', response.access_token);
      setToken(response.access_token);
      apiClient.setAuthToken(response.access_token);
      
    } catch (err: any) {
      setError(err.detail || 'Login failed');
      throw err;
    }
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    setToken(null);
    apiClient.setAuthToken(null);
    setError(null);
  };

  return (
    <AuthContext.Provider value={{
      token,
      isAuthenticated: !!token,
      login,
      logout,
      error
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};