// Simulated AI detection engine — produces realistic violations with
// bounding boxes, timestamps and confidence scores for any uploaded file.

export type ViolationType =
  | "no_helmet"
  | "no_seatbelt"
  | "red_light"
  | "wrong_lane"
  | "mobile_usage"
  | "overspeeding";

export interface Violation {
  id: string;
  type: ViolationType;
  label: string;
  confidence: number; // 0..1
  timestamp?: number; // seconds (video only)
  // Bounding box as fractions of media dimensions (0..1)
  box: { x: number; y: number; w: number; h: number };
  vehicleId?: string;
}

export const VIOLATION_META: Record<
  ViolationType,
  { label: string; description: string }
> = {
  no_helmet: { label: "No Helmet", description: "Two-wheeler rider without helmet" },
  no_seatbelt: { label: "No Seatbelt", description: "Driver without seatbelt" },
  red_light: { label: "Red Light Violation", description: "Vehicle crossed on red signal" },
  wrong_lane: { label: "Wrong Lane", description: "Vehicle driving in wrong lane" },
  mobile_usage: { label: "Mobile Phone Usage", description: "Driver using phone" },
  overspeeding: { label: "Overspeeding", description: "Exceeded speed limit" },
};

// Deterministic pseudo-random based on filename so repeat uploads match.
function hashSeed(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 0xffffffff;
}

function rng(seed: number) {
  let s = seed * 1e9;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

const TYPES: ViolationType[] = [
  "no_helmet",
  "no_seatbelt",
  "red_light",
  "wrong_lane",
  "mobile_usage",
  "overspeeding",
];

export interface DetectionInput {
  fileName: string;
  isVideo: boolean;
  durationSeconds?: number;
}

export function runDetection({
  fileName,
  isVideo,
  durationSeconds = 12,
}: DetectionInput): Violation[] {
  const seed = hashSeed(fileName || "demo");
  const rand = rng(seed);

  const count = isVideo ? 4 + Math.floor(rand() * 5) : 1 + Math.floor(rand() * 3);
  const violations: Violation[] = [];

  for (let i = 0; i < count; i++) {
    const type = TYPES[Math.floor(rand() * TYPES.length)];
    const w = 0.12 + rand() * 0.18;
    const h = 0.18 + rand() * 0.22;
    const x = 0.05 + rand() * (0.9 - w);
    const y = 0.1 + rand() * (0.75 - h);
    violations.push({
      id: `v-${i}-${Math.floor(rand() * 1e6)}`,
      type,
      label: VIOLATION_META[type].label,
      confidence: 0.72 + rand() * 0.27,
      timestamp: isVideo ? Number((rand() * durationSeconds).toFixed(1)) : undefined,
      box: { x, y, w, h },
      vehicleId: `VH-${Math.floor(rand() * 9000 + 1000)}`,
    });
  }

  return violations.sort((a, b) =>
    isVideo ? (a.timestamp ?? 0) - (b.timestamp ?? 0) : b.confidence - a.confidence
  );
}

export function formatTimestamp(t?: number): string {
  if (t == null) return "—";
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1);
  return `${m.toString().padStart(2, "0")}:${s.padStart(4, "0")}`;
}
