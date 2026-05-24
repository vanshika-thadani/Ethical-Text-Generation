// TypeScript interfaces matching the FastAPI backend response schemas exactly.

export interface CandidateScores {
  text: string;
  toxicity_score: number;
  sentiment_score: number;
  bias_score: number;
  coherence_score: number;
  ethics_score: number;
  fluency_score: number;
  manipulation_penalty: number;
  final_score: number;
}

// POST /upload
export interface UploadResponse {
  status: string;
  document_name: string;
  chunks_added: number;
}

// GET /rag-status
export interface RagStatusResponse {
  status: string;
  total_chunks: number;
  documents: string[];
}

// POST /ask
export interface RetrievedChunk {
  text: string;
  document: string;
  chunk_index: number;
  distance: number;
}

export interface AskResponse {
  question: string;
  retrieved_chunks: RetrievedChunk[];
  answer: string;
  ethical_scores: CandidateScores;
}

// POST /analyze-document
export interface ChunkAnalysis {
  chunk: string;
  chunk_index: number;
  toxicity_score: number;
  bias_score: number;
  manipulation_penalty: number;
  ethics_score: number;
  flagged: boolean;
}

export interface AnalyzeDocumentResponse {
  document_name: string;
  total_chunks: number;
  flagged_chunks: number;
  unsafe_chunks: ChunkAnalysis[];
  all_chunks: ChunkAnalysis[];
}

// POST /rewrite
export interface RewriteResponse {
  original: string;
  ethical_rewrite: string;
  scores_before: CandidateScores;
  scores_after: CandidateScores;
}

// POST /generate
export interface GenerateResponse {
  generated_text: string;
  best_candidate: CandidateScores;
  all_candidates: CandidateScores[];
}

// POST /compare
export interface ImprovementMetrics {
  toxicity_safety_gain: number;
  bias_safety_gain: number;
  final_score_gain: number;
}

export interface CompareResponse {
  prompt: string;
  prompt_risk: number;
  baseline_output: string;
  safety_ranked_output: string;
  baseline_scores: CandidateScores;
  safety_ranked_scores: CandidateScores;
  improvement: ImprovementMetrics;
}
