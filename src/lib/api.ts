import type { Violation } from "@/lib/detection";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

interface ApiViolation {
  id: string;
  type: Violation["type"];
  label: string;
  confidence: number;
  timestamp?: number | null;
  box: { x: number; y: number; w: number; h: number };
  vehicle_id?: string | null;
}

interface AnalyzeResponse {
  file_name: string;
  is_video: boolean;
  duration_seconds?: number | null;
  violations: ApiViolation[];
}

export interface AnalyzeResult {
  fileName: string;
  isVideo: boolean;
  durationSeconds?: number;
  violations: Violation[];
}

function toViolation(v: ApiViolation): Violation {
  return {
    id: v.id,
    type: v.type,
    label: v.label,
    confidence: v.confidence,
    timestamp: v.timestamp ?? undefined,
    box: v.box,
    vehicleId: v.vehicle_id ?? undefined,
  };
}

export async function analyzeUpload(file: File): Promise<AnalyzeResult> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body,
  });

  if (!response.ok) {
    let message = "Analysis failed";
    try {
      const err = (await response.json()) as { detail?: string };
      if (err.detail) message = err.detail;
    } catch {
      // ignore JSON parse errors and use fallback message
    }
    throw new Error(message);
  }

  const data = (await response.json()) as AnalyzeResponse;

  return {
    fileName: data.file_name,
    isVideo: data.is_video,
    durationSeconds: data.duration_seconds ?? undefined,
    violations: data.violations.map(toViolation),
  };
}
