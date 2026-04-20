import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowRight, Image as ImageIcon, Video, Shield, Zap, Eye } from "lucide-react";
import { Navbar } from "@/components/Navbar";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Sentry — AI Traffic Violation Detection" },
      {
        name: "description",
        content:
          "Upload a video or image and detect traffic violations instantly with AI — helmets, seatbelts, red lights, lane discipline and more.",
      },
      { property: "og:title", content: "Sentry — AI Traffic Violation Detection" },
      {
        property: "og:description",
        content: "Premium AI-powered traffic violation detection. Drop a clip, see the calls.",
      },
    ],
  }),
  component: HomePage,
});

const VIOLATIONS = [
  "No Helmet",
  "No Seatbelt",
  "Red Light Violation",
  "Wrong Lane Driving",
  "Mobile Phone Usage",
  "Overspeeding",
];

function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main>
        {/* HERO */}
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 grid-bg opacity-60 [mask-image:radial-gradient(ellipse_at_top,black_20%,transparent_70%)]" />
          <div className="relative mx-auto max-w-7xl px-6 pt-20 pb-28 lg:pt-28 lg:pb-36">
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              className="max-w-3xl"
            >
              <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface-elevated px-3 py-1 text-xs text-muted-foreground">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violation opacity-75" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-violation" />
                </span>
                Real-time detection engine · v1.4
              </div>

              <h1 className="mt-6 text-5xl sm:text-6xl lg:text-7xl font-semibold tracking-[-0.04em] leading-[0.95] text-balance">
                Spot every violation.
                <br />
                <span className="text-muted-foreground">In a single frame.</span>
              </h1>

              <p className="mt-6 max-w-xl text-lg text-muted-foreground leading-relaxed">
                Upload a video or image and Sentry analyses traffic patterns to surface infractions
                with confidence scores and timestamped evidence.
              </p>

              <div className="mt-10 flex flex-wrap items-center gap-3">
                <Link
                  to="/upload"
                  search={{ kind: "image" as const }}
                  className="inline-flex items-center gap-2 rounded-full bg-foreground text-background px-5 py-3 text-sm font-medium hover:opacity-90 transition-opacity shadow-soft"
                >
                  <ImageIcon className="h-4 w-4" />
                  Upload image
                </Link>
                <Link
                  to="/upload"
                  search={{ kind: "video" as const }}
                  className="inline-flex items-center gap-2 rounded-full border border-border bg-surface-elevated px-5 py-3 text-sm font-medium hover:bg-accent transition-colors"
                >
                  <Video className="h-4 w-4" />
                  Upload video
                </Link>
                <Link
                  to="/upload"
                  className="inline-flex items-center gap-1.5 px-3 py-3 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  Start with your file <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </motion.div>

            {/* Floating preview card */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
              className="mt-20 mx-auto max-w-5xl"
            >
              <div className="rounded-3xl border border-border bg-surface-elevated p-2 shadow-elevated">
                <div className="relative aspect-[16/9] rounded-2xl bg-foreground overflow-hidden">
                  {/* Mock CCTV scene */}
                  <div className="absolute inset-0 bg-gradient-to-br from-[#1a1d24] via-[#2a2520] to-[#1f1a15]" />
                  <div className="absolute inset-0 grid-bg opacity-20" />

                  {/* Mock bounding boxes */}
                  <BoundingBox
                    x="14%"
                    y="32%"
                    w="18%"
                    h="36%"
                    label="No Helmet · 0.94"
                    delay={0.4}
                  />
                  <BoundingBox
                    x="46%"
                    y="44%"
                    w="22%"
                    h="32%"
                    label="Mobile Usage · 0.87"
                    delay={0.7}
                  />
                  <BoundingBox
                    x="74%"
                    y="22%"
                    w="16%"
                    h="40%"
                    label="Wrong Lane · 0.81"
                    delay={1.0}
                  />

                  {/* Scanline */}
                  <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <div className="absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-violation to-transparent animate-scanline" />
                  </div>

                  {/* HUD label */}
                  <div className="absolute top-4 left-4 flex items-center gap-2 rounded-md bg-black/40 backdrop-blur px-2.5 py-1 text-[11px] font-mono text-white/80">
                    <span className="h-1.5 w-1.5 rounded-full bg-violation animate-pulse" />
                    LIVE · CAM_07
                  </div>
                  <div className="absolute bottom-4 right-4 rounded-md bg-black/40 backdrop-blur px-2.5 py-1 text-[11px] font-mono text-white/80">
                    3 violations detected
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* DETECTION CAPABILITIES */}
        <section className="border-t border-border bg-surface">
          <div className="mx-auto max-w-7xl px-6 py-24">
            <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-8 mb-14">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-3">
                  Detection coverage
                </p>
                <h2 className="text-3xl sm:text-4xl font-semibold tracking-[-0.03em]">
                  Six categories.
                  <br />
                  One unified pipeline.
                </h2>
              </div>
              <p className="max-w-md text-muted-foreground">
                Each detection includes a labelled bounding box, vehicle ID and confidence score,
                with a timestamp when analysing video.
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-px bg-border rounded-2xl overflow-hidden border border-border">
              {VIOLATIONS.map((v, i) => (
                <div
                  key={v}
                  className="bg-surface-elevated p-6 sm:p-8 flex flex-col justify-between min-h-[160px] hover:bg-accent transition-colors"
                >
                  <span className="text-xs font-mono text-muted-foreground">0{i + 1}</span>
                  <span className="text-lg font-medium tracking-tight mt-8">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section className="border-t border-border">
          <div className="mx-auto max-w-7xl px-6 py-24">
            <h2 className="text-3xl sm:text-4xl font-semibold tracking-[-0.03em] mb-14 max-w-2xl">
              From upload to evidence,
              <br />
              <span className="text-muted-foreground">in three steps.</span>
            </h2>

            <div className="grid md:grid-cols-3 gap-8">
              {[
                {
                  icon: ImageIcon,
                  title: "Upload",
                  desc: "Drop an image or video clip. JPG, PNG, MP4 and MOV up to 200 MB.",
                },
                {
                  icon: Zap,
                  title: "Analyse",
                  desc: "Frames are scanned for vehicles, riders, signals and lane markings.",
                },
                {
                  icon: Eye,
                  title: "Review",
                  desc: "Annotated media plus a structured list of every violation found.",
                },
              ].map(({ icon: Icon, title, desc }, i) => (
                <div key={title} className="group">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="h-9 w-9 rounded-lg bg-foreground text-background flex items-center justify-center">
                      <Icon className="h-4 w-4" />
                    </div>
                    <span className="text-xs font-mono text-muted-foreground">Step {i + 1}</span>
                  </div>
                  <h3 className="text-xl font-medium tracking-tight mb-2">{title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* FOOTER */}
        <footer className="border-t border-border">
          <div className="mx-auto max-w-7xl px-6 py-10 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <Shield className="h-3.5 w-3.5" />
              <span>Sentry · Traffic Violation Detection</span>
            </div>
            <span className="font-mono text-xs">v1.5.0 · API integrated</span>
          </div>
        </footer>
      </main>
    </div>
  );
}

function BoundingBox({
  x,
  y,
  w,
  h,
  label,
  delay,
}: {
  x: string;
  y: string;
  w: string;
  h: string;
  label: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay }}
      className="absolute border-2 border-violation rounded-md"
      style={{ left: x, top: y, width: w, height: h }}
    >
      <span className="absolute -top-6 left-0 bg-violation text-violation-foreground text-[10px] font-mono font-medium px-1.5 py-0.5 rounded whitespace-nowrap">
        {label}
      </span>
    </motion.div>
  );
}
