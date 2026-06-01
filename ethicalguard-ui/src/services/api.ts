import axios, { AxiosError } from 'axios';
import type {
  UploadResponse,
  RagStatusResponse,
  AskResponse,
  AnalyzeDocumentResponse,
  RewriteResponse,
  GenerateResponse,
  CompareResponse,
} from '../types/api';

// ---------------------------------------------------------------------------
// Base URL resolution
// ---------------------------------------------------------------------------
// Priority:
//   1. VITE_API_URL environment variable  (set in .env.local for Colab/remote)
//   2. http://127.0.0.1:8000              (default for local development)
//
// To point at a Colab backend, create ethicalguard-ui/.env.local:
//   VITE_API_URL=https://<your-ngrok-or-colab-url>
//
// Vite automatically loads .env.local and exposes VITE_* variables.
// Never commit .env.local — it is already in .gitignore.

const BASE_URL: string =
  import.meta.env.VITE_API_URL?.replace(/\/$/, '') || 'http://127.0.0.1:8000';

// Axios instance — all components import from here, never hardcode URLs.
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2 min — generation can be slow on CPU
  headers: {
    // Colab's built-in proxy shows an auth/warning page for browser requests
    // unless this header is present. It tells the proxy to skip the interstitial
    // and forward the request directly to the FastAPI server.
    'ngrok-skip-browser-warning': 'true',
    // Some Colab proxy versions check for this instead
    'bypass-tunnel-reminder': 'true',
  },
});

// Human-readable error extractor.
export function getErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    if (!err.response) {
      // Network error — could be CORS, server down, or wrong URL
      const isCors = err.message?.toLowerCase().includes('network');
      return isCors
        ? `Cannot reach backend at ${BASE_URL}.\n\n` +
          `If using Colab's built-in proxy: open ${BASE_URL} in this browser first to authenticate, ` +
          `or switch to ngrok for reliable CORS support.\n\n` +
          `If running locally: uvicorn app.main:app --reload`
        : `Backend not reachable at ${BASE_URL}. Check that the server is running.`;
    }
    const detail = err.response.data?.detail;
    if (typeof detail === 'string') return detail;
    if (typeof detail === 'object') return JSON.stringify(detail);
    return `Server error ${err.response.status}`;
  }
  if (err instanceof Error) return err.message;
  return 'An unexpected error occurred.';
}

// ── Endpoints ──────────────────────────────────────────────────────────────

export async function getRagStatus(): Promise<RagStatusResponse> {
  const { data } = await api.get<RagStatusResponse>('/rag-status');
  return data;
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function askQuestion(
  question: string,
  topK: number,
  documentName?: string
): Promise<AskResponse> {
  const { data } = await api.post<AskResponse>('/ask', {
    question,
    top_k: topK,
    document_name: documentName || undefined,
  });
  return data;
}

export async function analyzeDocument(
  documentName: string
): Promise<AnalyzeDocumentResponse> {
  const { data } = await api.post<AnalyzeDocumentResponse>('/analyze-document', {
    document_name: documentName,
  });
  return data;
}

export async function rewriteText(
  text: string,
  beams = 3,
  maxTokens = 80
): Promise<RewriteResponse> {
  const { data } = await api.post<RewriteResponse>('/rewrite', {
    text,
    beams,
    max_tokens: maxTokens,
  });
  return data;
}

export async function generateText(
  text: string,
  beams = 3,
  maxTokens = 80,
  alpha = 0.7
): Promise<GenerateResponse> {
  const { data } = await api.post<GenerateResponse>('/generate', {
    text,
    beams,
    max_tokens: maxTokens,
    alpha,
  });
  return data;
}

export async function compareGeneration(
  text: string,
  beams = 3,
  maxTokens = 80,
  alpha = 0.7
): Promise<CompareResponse> {
  const { data } = await api.post<CompareResponse>('/compare', {
    text,
    beams,
    max_tokens: maxTokens,
    alpha,
  });
  return data;
}
