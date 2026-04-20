import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  RotateCcw,
  AlertTriangle,
  Filter as FilterIcon,
  CheckCircle2,
} from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { getSession, clearSession } from "@/lib/session";
import {
  formatTimestamp,
  VIOLATION_META,
  type Violation,
  type ViolationType,
} from "@/lib/detection";

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Results — Sentry Traffic AI" },
      { name: "description", content: "Detected traffic violations with annotated evidence." },
    ],
  }),
  component: ResultsPage,
});

type SortKey = "confidence" | "time";
const MIN_CONFIDENCE = Number(import.meta.env.VITE_MIN_CONFIDENCE ?? "0.6");
const MAX_OVERLAY_BOXES = 18;

function ResultsPage() {
  const navigate = useNavigate();
  const session = getSession();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [filterTypes, setFilterTypes] = useState<Set<ViolationType>>(new Set());
  const [sortKey, setSortKey] = useState<SortKey>("confidence");
  const mediaRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!session) navigate({ to: "/upload" });
  }, [session, navigate]);

  const filtered = useMemo(() => {
    if (!session) return [];
    let v = [...session.violations];
    v = v.filter((x) => x.confidence >= MIN_CONFIDENCE);
    if (filterTypes.size > 0) v = v.filter((x) => filterTypes.has(x.type));
    if (sortKey === "confidence") v.sort((a, b) => b.confidence - a.confidence);
    else v.sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0));
    return v;
  }, [session, filterTypes, sortKey]);

  const overlayViolations = useMemo(() => {
    const top = filtered.slice(0, MAX_OVERLAY_BOXES);
    if (!activeId) return top;
    const active = filtered.find((v) => v.id === activeId);
    if (!active) return top;
    return top.some((v) => v.id === active.id)
      ? top
      : [active, ...top.slice(0, MAX_OVERLAY_BOXES - 1)];
  }, [filtered, activeId]);

  useEffect(() => {
    if (!filtered.length) {
      setActiveId(null);
      return;
    }
    if (!activeId || !filtered.some((v) => v.id === activeId)) {
      setActiveId(filtered[0].id);
    }
  }, [filtered, activeId]);

  if (!session) return null;

  const types = Array.from(new Set(session.violations.map((v) => v.type)));

  const toggleFilter = (t: ViolationType) => {
    setFilterTypes((prev) => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  };

  const jumpTo = (v: Violation) => {
    setActiveId(v.id);
    if (session.isVideo && videoRef.current && v.timestamp != null) {
      videoRef.current.currentTime = v.timestamp;
      videoRef.current.pause();
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="mx-auto max-w-7xl px-6 py-10">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8">
          <div>
            <Link
              to="/upload"
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors mb-3"
            >
              <ArrowLeft className="h-3 w-3" />
              Back to upload
            </Link>
            <h1 className="text-3xl sm:text-4xl font-semibold tracking-[-0.03em]">
              Analysis complete
            </h1>
            <p className="mt-1.5 text-sm text-muted-foreground font-mono truncate max-w-md">
              {session.fileName}
            </p>
          </div>

          <button
            onClick={() => {
              clearSession();
              navigate({ to: "/upload" });
            }}
            className="inline-flex items-center gap-2 rounded-full border border-border bg-surface-elevated px-4 py-2 text-sm font-medium hover:bg-accent transition-colors self-start sm:self-auto"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Try another file
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-border rounded-2xl overflow-hidden border border-border mb-8">
          <Stat label="Violations" value={session.violations.length.toString()} accent />
          <Stat
            label="Avg. confidence"
            value={`${Math.round(
              (session.violations.reduce((s, v) => s + v.confidence, 0) /
                Math.max(1, session.violations.length)) *
                100,
            )}%`}
          />
          <Stat label="Categories" value={types.length.toString()} />
          <Stat
            label="Media"
            value={
              session.isVideo ? `Video · ${(session.durationSeconds ?? 0).toFixed(1)}s` : "Image"
            }
          />
        </div>

        {/* Main split */}
        <div className="grid lg:grid-cols-[1fr_380px] gap-6">
          {/* Media viewer */}
          <div className="rounded-3xl border border-border bg-surface-elevated p-2 shadow-soft">
            <div
              ref={mediaRef}
              className="relative aspect-video rounded-2xl overflow-hidden bg-foreground"
            >
              {session.fileUrl ? (
                session.isVideo ? (
                  <video
                    ref={videoRef}
                    src={session.fileUrl}
                    controls
                    className="absolute inset-0 h-full w-full object-contain"
                  />
                ) : (
                  <img
                    src={session.fileUrl}
                    alt="analysed"
                    className="absolute inset-0 h-full w-full object-contain"
                  />
                )
              ) : (
                // Demo placeholder
                <div className="absolute inset-0 bg-gradient-to-br from-[#1a1d24] via-[#2a2520] to-[#1f1a15] flex items-center justify-center">
                  <span className="text-white/40 text-xs font-mono">DEMO · CAM_07 · 1920×1080</span>
                </div>
              )}

              {/* Bounding boxes overlay */}
              <div className="absolute inset-0 pointer-events-none">
                {overlayViolations.map((v) => (
                  <BoundingBox
                    key={v.id}
                    violation={v}
                    active={activeId === v.id}
                    onClick={() => jumpTo(v)}
                  />
                ))}
              </div>

              {/* HUD */}
              <div className="absolute top-3 left-3 flex items-center gap-2 rounded-md bg-black/50 backdrop-blur px-2.5 py-1 text-[10px] font-mono text-white/80">
                <span className="h-1.5 w-1.5 rounded-full bg-violation animate-pulse" />
                ANNOTATED · showing {overlayViolations.length} of {filtered.length}
              </div>
            </div>
          </div>

          {/* Side panel */}
          <aside className="space-y-4">
            {/* Filters */}
            <div className="rounded-2xl border border-border bg-surface-elevated p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <FilterIcon className="h-3.5 w-3.5" />
                  Filter & sort
                </div>
                {filterTypes.size > 0 && (
                  <button
                    onClick={() => setFilterTypes(new Set())}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    Clear
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-1.5 mb-4">
                {types.map((t) => {
                  const active = filterTypes.has(t);
                  return (
                    <button
                      key={t}
                      onClick={() => toggleFilter(t)}
                      className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${
                        active
                          ? "bg-foreground text-background border-foreground"
                          : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/40"
                      }`}
                    >
                      {VIOLATION_META[t].label}
                    </button>
                  );
                })}
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-muted-foreground">Sort by</span>
                {(["confidence", "time"] as SortKey[]).map((k) => (
                  <button
                    key={k}
                    onClick={() => setSortKey(k)}
                    disabled={k === "time" && !session.isVideo}
                    className={`px-2 py-1 rounded-md transition-colors ${
                      sortKey === k
                        ? "bg-accent text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    } disabled:opacity-30 disabled:cursor-not-allowed`}
                  >
                    {k === "confidence" ? "Confidence" : "Timestamp"}
                  </button>
                ))}
              </div>
            </div>

            {/* List */}
            <div className="rounded-2xl border border-border bg-surface-elevated overflow-hidden">
              <div className="px-5 py-4 border-b border-border flex items-center justify-between">
                <span className="text-sm font-medium">Detections</span>
                <span className="text-xs font-mono text-muted-foreground">
                  {filtered.length} (≥{Math.round(MIN_CONFIDENCE * 100)}%)
                </span>
              </div>
              <div className="max-h-[520px] overflow-y-auto divide-y divide-border">
                {filtered.length === 0 ? (
                  <EmptyDetections />
                ) : (
                  filtered.map((v, i) => (
                    <ViolationRow
                      key={v.id}
                      violation={v}
                      index={i}
                      active={activeId === v.id}
                      isVideo={session.isVideo}
                      onClick={() => jumpTo(v)}
                    />
                  ))
                )}
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="bg-surface-elevated p-5">
      <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
      <p
        className={`mt-2 text-2xl sm:text-3xl font-semibold tracking-tight ${
          accent ? "text-violation" : ""
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function BoundingBox({
  violation,
  active,
  onClick,
}: {
  violation: Violation;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <motion.button
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: Math.random() * 0.3 }}
      onClick={onClick}
      className={`absolute pointer-events-auto rounded-md border-2 transition-all duration-200 ${
        active
          ? "border-violation z-20 shadow-[0_0_0_4px_color-mix(in_oklab,var(--color-violation)_25%,transparent)]"
          : "border-violation/80 hover:border-violation z-10"
      }`}
      style={{
        left: `${violation.box.x * 100}%`,
        top: `${violation.box.y * 100}%`,
        width: `${violation.box.w * 100}%`,
        height: `${violation.box.h * 100}%`,
      }}
    >
      {active ? (
        <span className="absolute -top-6 left-0 bg-violation text-violation-foreground text-[10px] font-mono font-medium px-1.5 py-0.5 rounded whitespace-nowrap">
          {violation.label} · {(violation.confidence * 100).toFixed(0)}%
        </span>
      ) : (
        <span className="absolute -top-1.5 -left-1.5 h-2.5 w-2.5 rounded-full bg-violation" />
      )}
    </motion.button>
  );
}

function ViolationRow({
  violation,
  index,
  active,
  isVideo,
  onClick,
}: {
  violation: Violation;
  index: number;
  active: boolean;
  isVideo: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-5 py-4 transition-colors ${
        active ? "bg-accent" : "hover:bg-accent/50"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="h-8 w-8 rounded-lg bg-violation/15 text-violation flex items-center justify-center shrink-0">
          <AlertTriangle className="h-3.5 w-3.5" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-medium tracking-tight truncate">{violation.label}</p>
            <span className="text-[10px] font-mono text-muted-foreground shrink-0">
              #{(index + 1).toString().padStart(2, "0")}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5 truncate">
            {VIOLATION_META[violation.type].description}
          </p>
          <div className="mt-2.5 flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
            {isVideo && <span>{formatTimestamp(violation.timestamp)}</span>}
            <span>{violation.vehicleId}</span>
            <span className="ml-auto inline-flex items-center gap-1.5">
              <span className="relative h-1 w-12 rounded-full bg-border overflow-hidden">
                <span
                  className="absolute inset-y-0 left-0 bg-violation"
                  style={{ width: `${violation.confidence * 100}%` }}
                />
              </span>
              {(violation.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
    </button>
  );
}

function EmptyDetections() {
  return (
    <div className="px-5 py-12 text-center">
      <div className="mx-auto h-10 w-10 rounded-full bg-success/15 text-success flex items-center justify-center mb-3">
        <CheckCircle2 className="h-5 w-5" />
      </div>
      <p className="text-sm font-medium">No matching detections</p>
      <p className="text-xs text-muted-foreground mt-1">Adjust filters to see more results.</p>
    </div>
  );
}
