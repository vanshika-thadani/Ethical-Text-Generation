import { useRef, useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadDocument, getErrorMessage } from '../services/api';
import type { UploadResponse } from '../types/api';
import LoadingSpinner from './LoadingSpinner';

interface Props {
  onUploaded: (docName: string) => void;
}

export default function UploadDocument({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState('');

  async function handleFile(file: File) {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!['txt', 'pdf'].includes(ext ?? '')) {
      setError('Only .txt and .pdf files are supported.');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await uploadDocument(file);
      setResult(res);
      onUploaded(res.document_name);
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200
          ${dragging ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}
      >
        <Upload className="w-10 h-10 text-indigo-400 mx-auto mb-3" />
        <p className="font-medium text-slate-600">Drop a file here or <span className="text-indigo-600 underline">browse</span></p>
        <p className="text-xs text-slate-400 mt-1">Supports .txt and .pdf</p>
        <input ref={inputRef} type="file" accept=".txt,.pdf" className="hidden" onChange={onInputChange} />
      </div>

      {loading && <LoadingSpinner label="Uploading and ingesting document…" />}

      {/* Success */}
      {result && !loading && (
        <div className="flex items-start gap-3 bg-emerald-50 border border-emerald-200 rounded-2xl p-4">
          <CheckCircle className="w-5 h-5 text-emerald-600 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold text-emerald-700">Document ingested successfully</p>
            <div className="mt-2 space-y-1 text-sm text-emerald-600">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4" />
                <span>{result.document_name}</span>
              </div>
              <p><span className="font-medium">{result.chunks_added}</span> chunks added to vector DB</p>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="flex items-start gap-3 bg-rose-50 border border-rose-200 rounded-2xl p-4">
          <AlertCircle className="w-5 h-5 text-rose-600 mt-0.5 shrink-0" />
          <p className="text-sm text-rose-700">{error}</p>
        </div>
      )}
    </div>
  );
}
