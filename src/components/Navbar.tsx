import { Link } from "@tanstack/react-router";
import { ScanLine } from "lucide-react";

export function Navbar() {
  return (
    <header className="sticky top-0 z-40 glass border-b border-border/60">
      <div className="mx-auto max-w-7xl px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="relative h-8 w-8 rounded-lg bg-foreground text-background flex items-center justify-center">
            <ScanLine className="h-4 w-4" strokeWidth={2.25} />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-[15px] font-semibold tracking-tight">Sentry</span>
            <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground mt-0.5">
              Traffic AI
            </span>
          </div>
        </Link>

        <nav className="flex items-center gap-1">
          <Link
            to="/"
            className="px-3.5 py-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
            activeOptions={{ exact: true }}
            activeProps={{ className: "text-foreground" }}
          >
            Home
          </Link>
          <Link
            to="/upload"
            className="px-3.5 py-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
            activeProps={{ className: "text-foreground" }}
          >
            Upload
          </Link>
          <Link
            to="/upload"
            className="ml-2 inline-flex items-center gap-1.5 rounded-full bg-foreground text-background px-4 py-1.5 text-sm font-medium hover:opacity-90 transition-opacity"
          >
            Start scan
          </Link>
        </nav>
      </div>
    </header>
  );
}
