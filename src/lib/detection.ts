export type ViolationType =
  | "no_helmet"
  | "no_seatbelt"
  | "red_light"
  | "wrong_lane"
  | "mobile_usage"
  | "overspeeding"
  | "drunk_driving"
  | "no_valid_license"
  | "triple_riding"
  | "no_parking"
  | "dangerous_driving";

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

export const VIOLATION_META: Record<ViolationType, { label: string; description: string }> = {
  no_helmet: {
    label: "No Helmet",
    description: "No helmet / pillion without helmet (Rs 500 - Rs 1,000)",
  },
  no_seatbelt: { label: "No Seatbelt", description: "Seatbelt non-compliance (~Rs 1,000)" },
  red_light: { label: "Red Light Violation", description: "Jumping red light (~Rs 1,000)" },
  wrong_lane: { label: "Wrong Lane", description: "Wrong-lane driving pattern" },
  mobile_usage: {
    label: "Mobile Phone Usage",
    description: "Phone use while driving (Rs 1,000 - Rs 5,000+)",
  },
  overspeeding: { label: "Overspeeding", description: "Overspeeding (Rs 1,000 - Rs 2,000)" },
  drunk_driving: {
    label: "Drunk Driving",
    description: "Drink and drive cue (Rs 10,000+ / possible jail)",
  },
  no_valid_license: {
    label: "No Valid License",
    description: "Driving without valid license (Rs 1,000 - Rs 5,000)",
  },
  triple_riding: { label: "Triple Riding", description: "3+ persons on a two-wheeler (Rs 500)" },
  no_parking: {
    label: "No Parking / Obstruction",
    description: "No-parking or middle-road obstruction (Rs 1,000 - Rs 2,000)",
  },
  dangerous_driving: {
    label: "Dangerous Driving / Racing",
    description: "Rash driving or racing behavior (Rs 5,000 - Rs 10,000)",
  },
};

export function formatTimestamp(t?: number): string {
  if (t == null) return "—";
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1);
  return `${m.toString().padStart(2, "0")}:${s.padStart(4, "0")}`;
}
