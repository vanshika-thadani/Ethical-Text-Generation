import { useEffect, useState } from 'react';
import { Database, RefreshCw, FileText } from 'lucide-react';
import { getRagStatus, getErrorMessage } from '../services/api';
import type { RagStatusResponse } from '../types/api';

export default function RagStatus() {
  const [status, setStatus] = useState<RagStatusResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function fetch() {
    setLoading(true);
    setError('');
    try {
      setStatus(await getRagStatus());
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetch(); }, []);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-indigo-500" />
          <h2 className="font-semibold text-slate-700">Vector DB Status</h2>
        </div>
        <button
          onClick={fetch}
          disabled={loading}
          className="p-1.5 rounded-lg hover:bg-slate-100 transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 text-slate-500 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {error && <p className="text-sm text-rose-600 bg-rose-50 rounded-lg px-3 py-2">{error}</p>}

      {status && (
        <div className="space-y-3">
          {/* Status badge */}
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${status.status === 'ready' ? 'bg-emerald-500' : 'bg-amber-400'}`} />
            <span className="text-sm font-medium text-slate-600 capitalize">{status.status}</span>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-indigo-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-indigo-700">{status.total_chunks}</p>
              <p className="text-xs text-indigo-500 mt-0.5">Total Chunks</p>
            </div>
            <div className="bg-violet-50 rounded-xl p-3 text-center">
              <p className="text-2xl font-bold text-violet-700">{status.documents.length}</p>
              <p className="text-xs text-violet-500 mt-0.5">Documents</p>
            </div>
          </div>

          {/* Document list */}
          {status.documents.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Uploaded Documents</p>
              {status.documents.map(doc => (
                <div key={doc} className="flex items-center gap-2 text-sm text-slate-600 bg-slate-50 rounded-lg px-3 py-1.5">
                  <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  <span className="truncate">{doc}</span>
                </div>
              ))}
            </div>
          )}

          {status.documents.length === 0 && (
            <p className="text-sm text-slate-400 text-center py-2">No documents uploaded yet.</p>
          )}
        </div>
      )}
    </div>
  );
}
