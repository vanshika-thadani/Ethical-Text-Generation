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

// Axios instance pointing at the FastAPI backend.
// All components import from here — never hardcode the base URL elsewhere.
const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  timeout: 120_000, // 2 min — generation can be slow on CPU
});

// Human-readable error extractor.
// If the backend is unreachable, show a helpful startup message.
export function getErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    if (!err.response) {
      return 'Backend server not running. Start it using: uvicorn app.main:app --reload';
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
