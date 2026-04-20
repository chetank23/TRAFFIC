import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileVideo, FileImage, X, Play, Sparkles } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { analyzeUpload } from "@/lib/api";
import { setSession } from "@/lib/session";

interface UploadSearch {
  kind?: "image" | "video";
}

export const Route = createFileRoute("/upload")({
  validateSearch: (s: Record<string, unknown>): UploadSearch => ({
    kind: s.kind === "image" || s.kind === "video" ? s.kind : undefined,
  }),
  head: () => ({
    meta: [
      { title: "Upload — Sentry Traffic AI" },
      { name: "description", content: "Drop an image or video to analyse traffic violations." },
      { property: "og:title", content: "Upload — Sentry Traffic AI" },
      {
        property: "og:description",
        content: "Drop an image or video to analyse traffic violations.",
      },
    ],
  }),
  component: UploadPage,
});

const ACCEPT_IMAGE = "image/jpeg,image/jpg,image/png,image/webp";
const ACCEPT_VIDEO = "video/mp4,video/quicktime,video/webm";

function UploadPage() {
  const navigate = useNavigate();
  const search = Route.useSearch();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    setError(null);
    const isImage = f.type.startsWith("image/");
    const isVideo = f.type.startsWith("video/");
    if (!isImage && !isVideo) {
      setError("Unsupported file type. Please upload an image or video.");
      return;
    }
    if (f.size > 200 * 1024 * 1024) {
      setError("File too large. Maximum 200 MB.");
      return;
    }
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const reset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setError(null);
    setProgress(0);
  };

  const startAnalysis = async () => {
    if (!file || !previewUrl) return;
    setIsProcessing(true);
    setProgress(0);

    const timer = setInterval(() => {
      setProgress((current) => {
        if (current >= 90) return current;
        return Math.min(90, current + Math.max(3, Math.round(Math.random() * 10)));
      });
    }, 220);

    try {
      const result = await analyzeUpload(file);
      setProgress(98);

      setSession({
        fileName: result.fileName,
        fileUrl: previewUrl,
        isVideo: result.isVideo,
        durationSeconds: result.durationSeconds,
        violations: result.violations,
      });

      setProgress(100);
      await new Promise((r) => setTimeout(r, 180));
      navigate({ to: "/results" });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to process this file right now.";
      setError(message);
      setIsProcessing(false);
    } finally {
      clearInterval(timer);
    }
  };

  const accept =
    search.kind === "image"
      ? ACCEPT_IMAGE
      : search.kind === "video"
        ? ACCEPT_VIDEO
        : `${ACCEPT_IMAGE},${ACCEPT_VIDEO}`;

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <AnimatePresence mode="wait">
        {isProcessing ? (
          <ProcessingState key="processing" progress={progress} fileName={file?.name ?? ""} />
        ) : (
          <motion.main
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mx-auto max-w-4xl px-6 py-16 sm:py-24"
          >
            <div className="mb-10">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-3">
                Step 01 · Upload
              </p>
              <h1 className="text-4xl sm:text-5xl font-semibold tracking-[-0.03em]">
                Drop your media to begin.
              </h1>
              <p className="mt-3 text-muted-foreground max-w-xl">
                {search.kind === "image"
                  ? "JPG, PNG or WEBP. Up to 200 MB."
                  : search.kind === "video"
                    ? "MP4, MOV or WEBM. Up to 200 MB."
                    : "Images (JPG, PNG, WEBP) or video (MP4, MOV, WEBM). Up to 200 MB."}
              </p>
            </div>

            {!file ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                className={`relative cursor-pointer rounded-3xl border-2 border-dashed transition-all duration-300 ${
                  isDragging
                    ? "border-violation bg-violation/5 scale-[1.01]"
                    : "border-border bg-surface hover:bg-accent/40 hover:border-foreground/30"
                }`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept={accept}
                  className="sr-only"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFile(f);
                  }}
                />
                <div className="px-8 py-20 sm:py-28 flex flex-col items-center text-center">
                  <div className="h-14 w-14 rounded-2xl bg-foreground text-background flex items-center justify-center mb-6">
                    <UploadCloud className="h-6 w-6" strokeWidth={2} />
                  </div>
                  <p className="text-xl font-medium tracking-tight">
                    {isDragging ? "Drop to upload" : "Drag & drop your file"}
                  </p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    or <span className="text-foreground underline underline-offset-4">browse</span>{" "}
                    from your device
                  </p>
                  <div className="mt-8 flex items-center gap-2 text-xs font-mono text-muted-foreground">
                    <FileImage className="h-3.5 w-3.5" /> JPG · PNG · WEBP
                    <span className="mx-2 opacity-40">|</span>
                    <FileVideo className="h-3.5 w-3.5" /> MP4 · MOV · WEBM
                  </div>
                </div>
              </div>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-3xl border border-border bg-surface-elevated p-2 shadow-soft"
              >
                <div className="relative aspect-video rounded-2xl overflow-hidden bg-foreground/95">
                  {file.type.startsWith("image/") ? (
                    <img
                      src={previewUrl!}
                      alt="preview"
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  ) : (
                    <video
                      src={previewUrl!}
                      controls
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  )}
                  <button
                    onClick={reset}
                    className="absolute top-3 right-3 h-8 w-8 rounded-full bg-black/60 backdrop-blur text-white flex items-center justify-center hover:bg-black/80 transition-colors"
                    aria-label="Remove file"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                <div className="flex items-center justify-between gap-4 px-4 py-4">
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">{file.name}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {(file.size / 1024 / 1024).toFixed(2)} MB ·{" "}
                      {file.type.startsWith("image/") ? "Image" : "Video"}
                    </p>
                  </div>
                  <button
                    onClick={startAnalysis}
                    className="inline-flex items-center gap-2 rounded-full bg-foreground text-background px-5 py-2.5 text-sm font-medium hover:opacity-90 transition-opacity shadow-soft shrink-0"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" />
                    Analyse
                  </button>
                </div>
              </motion.div>
            )}

            {error && (
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 rounded-xl bg-destructive/10 border border-destructive/30 px-4 py-3 text-sm text-destructive"
              >
                {error}
              </motion.div>
            )}

            {!file && (
              <div className="mt-8 flex items-center gap-3 rounded-2xl bg-surface border border-border px-5 py-4">
                <Sparkles className="h-4 w-4 text-violation" />
                <p className="text-xs text-muted-foreground">
                  Backend-connected analysis is enabled. Upload real media to generate live results.
                </p>
              </div>
            )}
          </motion.main>
        )}
      </AnimatePresence>
    </div>
  );
}

function ProcessingState({ progress, fileName }: { progress: number; fileName: string }) {
  const messages = [
    "Decoding stream",
    "Detecting vehicles & riders",
    "Analysing traffic patterns",
    "Cross-referencing violations",
    "Compiling evidence",
  ];
  const idx = Math.min(messages.length - 1, Math.floor((progress / 100) * messages.length));

  return (
    <motion.main
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="mx-auto max-w-2xl px-6 py-24 sm:py-32"
    >
      <div className="text-center">
        <div className="relative mx-auto h-20 w-20 mb-10">
          <div className="absolute inset-0 rounded-full border-2 border-border" />
          <motion.div
            className="absolute inset-0 rounded-full border-2 border-foreground border-t-transparent"
            animate={{ rotate: 360 }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="font-mono text-sm font-medium">{progress}%</span>
          </div>
        </div>

        <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-3">Analysing</p>
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-[-0.03em]">{messages[idx]}…</h1>
        <p className="mt-3 text-sm text-muted-foreground font-mono truncate">{fileName}</p>

        <div className="mt-12 h-1 w-full rounded-full bg-border overflow-hidden">
          <motion.div
            className="h-full bg-foreground"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          />
        </div>
      </div>
    </motion.main>
  );
}
