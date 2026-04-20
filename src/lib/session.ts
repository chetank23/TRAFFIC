// In-memory store for the active analysis session.
// Files don't survive route changes via plain state, so we stash the object URL.

import type { Violation } from "./detection";

export interface AnalysisSession {
  fileName: string;
  fileUrl: string; // object URL
  isVideo: boolean;
  durationSeconds?: number;
  violations: Violation[];
}

let session: AnalysisSession | null = null;

export function setSession(s: AnalysisSession) {
  // Revoke previous URL to avoid leaks
  if (session?.fileUrl && session.fileUrl !== s.fileUrl) {
    try {
      URL.revokeObjectURL(session.fileUrl);
    } catch {
      /* ignore */
    }
  }
  session = s;
}

export function getSession(): AnalysisSession | null {
  return session;
}

export function clearSession() {
  if (session?.fileUrl) {
    try {
      URL.revokeObjectURL(session.fileUrl);
    } catch {
      /* ignore */
    }
  }
  session = null;
}
