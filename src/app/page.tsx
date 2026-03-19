"use client";

import {
  useState,
  useEffect,
  useRef,
  useCallback,
  type DragEvent,
  type ChangeEvent,
} from "react";
import { v4 as uuidv4 } from "uuid";

// ─── Types ────────────────────────────────────────────────────────────────────

type AppState = "home" | "loading" | "locked" | "unlocked";
type ScanMode = "person" | "car" | "item";

interface LocalUser {
  uuid: string;
  email?: string;
  credits: number;
}

interface VisualAnomaly {
  description: string;
  riskWeight: number;
}

interface Listing {
  title: string;
  price: string;
  thumbnail: string;
  link: string;
}

interface FullItem {
  id: number;
  category: string;
  brand: string;
  model: string;
  confidence: number;
  year?: string;
  visualAnomalies: VisualAnomaly[];
  priceRange: string;
  riskScore: number;
  riskLevel: "low" | "medium" | "high";
  listings: Listing[];
  minPrice: number;
  maxPrice: number;
  x?: number;
  y?: number;
}

interface BlurredItem {
  id: number;
  category: string;
  brand: string;
  model: string;
  priceRange: string;
  riskScore: number;
  riskLevel: "low" | "medium" | "high";
  listings: Listing[];
}

interface AnalyzeResponse {
  scanId: string;
  itemCount: number;
  totalValueBlurred: string;
  items: BlurredItem[];
}

interface CarHigh {
  confidence: number;
  brand: string;
  series?: string;
  model: string;
  possibleYears?: string[];
  limitingFactors?: string[];
}

interface CarLow {
  confidence: number;
  candidates: { brand: string; model: string; probability: number }[];
  limitingFactors?: string[];
}

interface UnlockData {
  scanId: string;
  uuid: string;
  mode: ScanMode;
  timestamp: number;
  claudeRaw: unknown;
  items: FullItem[];
}

// ─── Constants ────────────────────────────────────────────────────────────────

const GOLD = "#C9A84C";

const LOADING_MESSAGES = [
  "Scanning luxury items...",
  "Checking market prices...",
  "Calculating total value...",
  "Almost ready...",
];

const MODE_OPTIONS: { mode: ScanMode; emoji: string; label: string }[] = [
  { mode: "person", emoji: "👗", label: "Scan Person" },
  { mode: "car",    emoji: "🚗", label: "Scan Car"    },
  { mode: "item",   emoji: "👜", label: "Scan Item"   },
];

// ─── Utilities ────────────────────────────────────────────────────────────────

function initUser(): LocalUser {
  const raw = localStorage.getItem("wl_user");
  if (raw) {
    try { return JSON.parse(raw) as LocalUser; } catch { /* fall through */ }
  }
  const fresh: LocalUser = { uuid: uuidv4(), credits: 0 };
  localStorage.setItem("wl_user", JSON.stringify(fresh));
  return fresh;
}

function saveUser(user: LocalUser): void {
  localStorage.setItem("wl_user", JSON.stringify(user));
}

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

async function compressToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const MAX = 1600;
      let { width, height } = img;
      if (width > MAX || height > MAX) {
        if (width >= height) { height = Math.round((height * MAX) / width); width = MAX; }
        else                 { width = Math.round((width * MAX) / height); height = MAX; }
      }
      const canvas = document.createElement("canvas");
      canvas.width = width; canvas.height = height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, width, height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.92).split(",")[1]);
    };
    img.onerror = reject;
    img.src = url;
  });
}

function isCarHigh(raw: unknown): raw is CarHigh {
  return typeof raw === "object" && raw !== null && "brand" in raw;
}
function isCarLow(raw: unknown): raw is CarLow {
  return typeof raw === "object" && raw !== null && "candidates" in raw;
}

// ─── HOME STATE ───────────────────────────────────────────────────────────────

interface HomeProps {
  mode: ScanMode;
  onModeChange: (m: ScanMode) => void;
  onImageSelected: (file: File) => void;
  error: string | null;
}

function HomeState({ mode, onModeChange, onImageSelected, error }: HomeProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) onImageSelected(file);
  }, [onImageSelected]);

  const handleChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onImageSelected(file);
  }, [onImageSelected]);

  return (
    <div className="flex flex-col items-center gap-8 w-full max-w-md mx-auto px-4 py-12">
      {/* Logo */}
      <div className="text-center">
        <h1 className="text-4xl font-bold tracking-tight" style={{ color: GOLD }}>
          WealthLens
        </h1>
        <p className="mt-2 text-xl font-medium text-white">How much is that person worth?</p>
        <p className="mt-1 text-sm text-zinc-400">Upload any photo. AI scans everything.</p>
      </div>

      {/* Mode selector */}
      <div className="flex gap-3 w-full">
        {MODE_OPTIONS.map(({ mode: m, emoji, label }) => (
          <button
            key={m}
            onClick={() => onModeChange(m)}
            className="flex-1 flex flex-col items-center gap-1 py-3 px-2 rounded-xl text-sm font-medium transition-all"
            style={{
              border: mode === m ? `2px solid ${GOLD}` : "2px solid #2a2a2a",
              color: mode === m ? GOLD : "#888",
              background: mode === m ? "rgba(201,168,76,0.08)" : "#111",
            }}
          >
            <span className="text-xl">{emoji}</span>
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Upload area */}
      <div
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className="w-full rounded-2xl flex flex-col items-center justify-center gap-3 cursor-pointer transition-all"
        style={{
          border: dragging ? `2px dashed ${GOLD}` : "2px dashed #333",
          background: dragging ? "rgba(201,168,76,0.06)" : "#111",
          minHeight: 200,
          padding: "2rem",
        }}
      >
        <span className="text-5xl">📷</span>
        <p className="text-white font-medium text-center">Drop a photo here</p>
        <p className="text-zinc-500 text-sm text-center">or click to browse</p>
        <p className="text-zinc-600 text-xs text-center">JPG, PNG, WEBP — auto-compressed to 800px</p>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />
      </div>

      {error && (
        <div className="w-full rounded-xl p-3 text-sm text-red-300 bg-red-900/30 border border-red-800">
          {error}
        </div>
      )}
    </div>
  );
}

// ─── LOADING STATE ────────────────────────────────────────────────────────────

function LoadingState({ imageUrl }: { imageUrl: string }) {
  const [msgIdx, setMsgIdx] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setMsgIdx((i) => (i + 1) % LOADING_MESSAGES.length), 2000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex flex-col items-center gap-8 w-full max-w-md mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold" style={{ color: GOLD }}>WealthLens</h1>
      <div className="relative w-full rounded-2xl overflow-hidden" style={{ maxHeight: 320 }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl}
          alt="Analyzing"
          className="w-full object-cover"
          style={{ filter: "blur(8px)", opacity: 0.5, maxHeight: 320 }}
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="gold-spinner" />
        </div>
      </div>
      <div key={msgIdx} className="fade-in-up text-center">
        <p className="text-lg font-medium text-white">{LOADING_MESSAGES[msgIdx]}</p>
        <p className="text-sm text-zinc-500 mt-1">This takes 10–20 seconds</p>
      </div>
    </div>
  );
}

// ─── LOCKED STATE ─────────────────────────────────────────────────────────────

interface LockedProps {
  imageUrl: string;
  analyzeData: AnalyzeResponse;
  user: LocalUser;
  onUnlocked: (data: UnlockData) => void;
  onUserUpdate: (user: LocalUser) => void;
}

type LockedFlow = null | "register" | "waitlist";

function LockedState({ imageUrl, analyzeData, user, onUnlocked, onUserUpdate }: LockedProps) {
  const [flow, setFlow] = useState<LockedFlow>(null);
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ text: string; ok: boolean } | null>(null);

  const doUnlock = useCallback(async (currentUser: LocalUser) => {
    try {
      const res = await fetch("/api/unlock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uuid: currentUser.uuid, scanId: analyzeData.scanId }),
      });
      const data = await res.json() as UnlockData & { error?: string };
      if (!res.ok) { setMessage({ text: data.error ?? "Unlock failed.", ok: false }); return; }
      const updated = { ...currentUser, credits: currentUser.credits - 2 };
      onUserUpdate(updated);
      onUnlocked(data);
    } catch {
      setMessage({ text: "Network error. Please try again.", ok: false });
    }
  }, [analyzeData.scanId, onUnlocked, onUserUpdate]);

  const handleRegister = useCallback(async () => {
    if (!isValidEmail(email)) { setMessage({ text: "Please enter a valid email.", ok: false }); return; }
    setLoading(true); setMessage(null);
    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uuid: user.uuid, email }),
      });
      const data = await res.json() as { success?: boolean; credits?: number; error?: string };
      if (!res.ok) { setMessage({ text: data.error ?? "Registration failed.", ok: false }); setLoading(false); return; }
      const updated: LocalUser = { ...user, email, credits: data.credits ?? 4 };
      onUserUpdate(updated);
      setMessage({ text: "✅ 4 credits added! Unlocking...", ok: true });
      await doUnlock(updated);
    } catch {
      setMessage({ text: "Network error. Please try again.", ok: false });
    } finally {
      setLoading(false);
    }
  }, [email, user, onUserUpdate, doUnlock]);

  const handleWaitlist = useCallback(async () => {
    if (!isValidEmail(email)) { setMessage({ text: "Please enter a valid email.", ok: false }); return; }
    setLoading(true); setMessage(null);
    try {
      const res = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const data = await res.json() as { success?: boolean; message?: string; error?: string };
      if (!res.ok) { setMessage({ text: data.error ?? "Something went wrong.", ok: false }); }
      else         { setMessage({ text: data.message ?? "✅ You're on the list!", ok: true }); setFlow(null); }
    } catch {
      setMessage({ text: "Network error. Please try again.", ok: false });
    } finally {
      setLoading(false);
    }
  }, [email]);

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold" style={{ color: GOLD }}>WealthLens</h1>

      {/* Blurred image */}
      <div className="relative w-full rounded-2xl overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={imageUrl} alt="Scan result" className="w-full object-cover"
          style={{ filter: "blur(20px)", maxHeight: 260, transform: "scale(1.05)" }} />
        <div className="absolute inset-0 flex items-center justify-center"
          style={{ background: "rgba(0,0,0,0.45)" }}>
          <div className="text-center">
            <p className="text-4xl mb-1">🔒</p>
            <p className="text-white font-semibold text-lg">Results Locked</p>
          </div>
        </div>
      </div>

      {/* Gold banner */}
      <div className="w-full rounded-xl py-3 px-4 text-center font-semibold text-base"
        style={{ background: "rgba(201,168,76,0.15)", border: `1px solid ${GOLD}`, color: GOLD }}>
        🔥 Found {analyzeData.itemCount} {analyzeData.itemCount === 1 ? "item" : "items"} — Total value {analyzeData.totalValueBlurred}
      </div>

      {/* Blurred item cards */}
      <div className="w-full flex flex-col gap-3">
        {analyzeData.items.map((item) => (
          <div key={item.id} className="w-full rounded-xl p-4"
            style={{ background: "#111", border: "1px solid #222" }}>
            <div className="flex justify-between items-start" style={{ filter: "blur(4px)", userSelect: "none" }}>
              <div>
                <p className="font-semibold text-white text-sm">Item #{item.id}</p>
                <p className="text-zinc-400 text-xs mt-0.5">{item.brand} · {item.model}</p>
              </div>
              <p className="text-sm font-medium" style={{ color: GOLD }}>{item.priceRange}</p>
            </div>
            <div className="mt-2 flex gap-1" style={{ filter: "blur(3px)", userSelect: "none" }}>
              {item.listings.slice(0, 2).map((_, i) => (
                <div key={i} className="h-2 flex-1 rounded-full bg-zinc-700" />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Unlock wall */}
      <div className="w-full rounded-2xl p-5 flex flex-col gap-4"
        style={{ background: "#111", border: "1px solid #2a2a2a" }}>
        <p className="text-white font-semibold text-center text-lg">🔓 Unlock Full Analysis</p>

        {message && (
          <div className={`rounded-xl p-3 text-sm text-center ${message.ok ? "text-green-300 bg-green-900/30 border border-green-800" : "text-red-300 bg-red-900/30 border border-red-800"}`}>
            {message.text}
          </div>
        )}

        {flow === null && (
          <div className="flex flex-col gap-3">
            <button onClick={() => { setFlow("register"); setMessage(null); setEmail(""); }}
              className="w-full py-3 rounded-xl font-semibold text-sm transition-all"
              style={{ background: GOLD, color: "#0a0a0a" }}>
              Register Free — Get 4 Credits
            </button>
            <button onClick={() => { setFlow("waitlist"); setMessage(null); setEmail(""); }}
              className="w-full py-3 rounded-xl font-semibold text-sm transition-all"
              style={{ border: `1px solid ${GOLD}`, color: GOLD, background: "transparent" }}>
              Pay $1.99 to Unlock
            </button>
          </div>
        )}

        {flow === "register" && (
          <div className="flex flex-col gap-3">
            <p className="text-zinc-400 text-sm text-center">Enter your email to get 4 free credits</p>
            <input
              type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full rounded-xl px-4 py-3 text-sm text-white outline-none"
              style={{ background: "#1a1a1a", border: "1px solid #333" }}
              onKeyDown={(e) => e.key === "Enter" && handleRegister()}
            />
            <button onClick={handleRegister} disabled={loading}
              className="w-full py-3 rounded-xl font-semibold text-sm disabled:opacity-50"
              style={{ background: GOLD, color: "#0a0a0a" }}>
              {loading ? "Please wait..." : "Get 4 Free Credits"}
            </button>
            <button onClick={() => { setFlow(null); setMessage(null); }}
              className="text-zinc-500 text-xs text-center hover:text-zinc-300 transition-colors">
              ← Back
            </button>
          </div>
        )}

        {flow === "waitlist" && (
          <div className="flex flex-col gap-3">
            <p className="text-zinc-400 text-sm text-center">
              💳 Payment coming soon! Leave your email to be notified:
            </p>
            <input
              type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full rounded-xl px-4 py-3 text-sm text-white outline-none"
              style={{ background: "#1a1a1a", border: "1px solid #333" }}
              onKeyDown={(e) => e.key === "Enter" && handleWaitlist()}
            />
            <button onClick={handleWaitlist} disabled={loading}
              className="w-full py-3 rounded-xl font-semibold text-sm disabled:opacity-50"
              style={{ border: `1px solid ${GOLD}`, color: GOLD, background: "transparent" }}>
              {loading ? "Please wait..." : "Notify Me"}
            </button>
            <button onClick={() => { setFlow(null); setMessage(null); }}
              className="text-zinc-500 text-xs text-center hover:text-zinc-300 transition-colors">
              ← Back
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── ITEM CARD (shared by person + item mode) ────────────────────────────────

const CATEGORY_ICONS: Record<string, string> = {
  bag: "👜", handbag: "👜", watch: "⌚", shoes: "👟", sneakers: "👟",
  jacket: "🧥", coat: "🧥", sunglasses: "🕶️", jewelry: "💎",
  necklace: "💎", bracelet: "💎", ring: "💍", belt: "👔",
  shirt: "👕", pants: "👖", dress: "👗", car: "🚗", default: "✦",
};

function categoryIcon(cat: string): string {
  if (!cat) return CATEGORY_ICONS.default;
  return CATEGORY_ICONS[cat.toLowerCase()] ?? CATEGORY_ICONS.default;
}

function RiskBar({ score, level }: { score: number; level: FullItem["riskLevel"] }) {
  const color = level === "low" ? "#22c55e" : level === "medium" ? "#eab308" : "#ef4444";
  const pct = Math.min(100, score);
  return (
    <div className="w-full h-2 rounded-full bg-zinc-800 overflow-hidden">
      <div
        className="h-full rounded-full risk-bar-fill"
        style={{ width: `${pct}%`, background: color, ["--bar-width" as string]: `${pct}%` }}
      />
    </div>
  );
}

function ItemCard({ item, index }: { item: FullItem; index: number }) {
  const showRisk = item.riskLevel === "medium" || item.riskLevel === "high";
  return (
    <div className="w-full rounded-2xl p-5 flex flex-col gap-4"
      style={{ background: "#111", border: "1px solid #222" }}>
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="flex items-center justify-center w-9 h-9 rounded-full text-sm font-bold flex-shrink-0"
          style={{ background: "rgba(201,168,76,0.15)", color: GOLD, border: `1px solid ${GOLD}` }}>
          {index + 1}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-lg">{categoryIcon(item.category)}</span>
            <span className="text-xs text-zinc-500 uppercase tracking-wide">{item.category}</span>
            <span className="text-xs text-zinc-600 ml-auto">
              {item.confidence}% confidence
            </span>
          </div>
          <p className="text-base font-bold mt-0.5" style={{ color: GOLD }}>
            {item.brand} {item.model}
            {item.year ? ` (${item.year})` : ""}
          </p>
          <p className="text-sm text-white mt-0.5">{item.priceRange}</p>
        </div>
      </div>

      {/* Listings */}
      {item.listings.length > 0 && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-zinc-500 uppercase tracking-wide">Market Listings</p>
          {item.listings.map((l, i) => (
            <a key={i} href={l.link} target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-3 rounded-xl p-3 transition-all hover:border-zinc-600"
              style={{ background: "#1a1a1a", border: "1px solid #2a2a2a" }}>
              {l.thumbnail && l.thumbnail !== "blurred" ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={l.thumbnail} alt={l.title} className="w-12 h-12 rounded-lg object-cover flex-shrink-0" />
              ) : (
                <div className="w-12 h-12 rounded-lg bg-zinc-800 flex-shrink-0 flex items-center justify-center text-lg">
                  {categoryIcon(item.category)}
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs text-zinc-300 truncate">{l.title}</p>
                <p className="text-sm font-semibold mt-0.5" style={{ color: GOLD }}>{l.price}</p>
              </div>
              <span className="text-zinc-600 text-xs flex-shrink-0">↗</span>
            </a>
          ))}
        </div>
      )}

      {/* Risk module */}
      {showRisk && item.visualAnomalies.length > 0 && (
        <div className="rounded-xl p-4 flex flex-col gap-3"
          style={{ background: "#1a0f0f", border: "1px solid #3a1a1a" }}>
          <p className="text-sm font-semibold text-white">🔍 AI Condition Analysis</p>
          <div className="flex flex-col gap-2">
            {item.visualAnomalies.map((a, i) => (
              <div key={i} className="flex justify-between items-center gap-2">
                <p className="text-xs text-zinc-400 flex-1">{a.description}</p>
                <span className="text-xs font-medium text-red-400 flex-shrink-0">+{a.riskWeight}%</span>
              </div>
            ))}
          </div>
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-zinc-400">Overall Risk Score</span>
              <span className={item.riskLevel === "high" ? "text-red-400" : "text-yellow-400"}>
                {item.riskScore}% — {item.riskLevel.toUpperCase()}
              </span>
            </div>
            <RiskBar score={item.riskScore} level={item.riskLevel} />
          </div>
          <a href="https://luxverify.us" target="_blank" rel="noopener noreferrer"
            className="text-xs text-center py-2 rounded-lg transition-all hover:opacity-80"
            style={{ color: GOLD, border: `1px solid rgba(201,168,76,0.3)`, background: "rgba(201,168,76,0.06)" }}>
            👉 Get Expert Authentication at Luxverify →
          </a>
        </div>
      )}
    </div>
  );
}

// ─── UNLOCKED — PERSON MODE ───────────────────────────────────────────────────

// Fallback grid positions (% from top-left) used when GPT-4o doesn't return x/y.
const FALLBACK_POSITIONS = [
  { x: 25, y: 25 }, { x: 75, y: 25 }, { x: 25, y: 70 },
  { x: 75, y: 70 }, { x: 50, y: 48 }, { x: 50, y: 18 },
  { x: 15, y: 48 }, { x: 85, y: 48 }, { x: 50, y: 78 },
];

function PersonMode({ imageUrl, items }: { imageUrl: string; items: FullItem[] }) {
  const [activeId, setActiveId] = useState<number | null>(null);
  const totalMin = items.reduce((s, i) => s + i.minPrice, 0);
  const totalMax = items.reduce((s, i) => s + i.maxPrice, 0);

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Image with interactive hotspot dots */}
      <div className="relative w-full rounded-2xl overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={imageUrl} alt="Scan" className="w-full object-cover" style={{ maxHeight: 360 }} />

        {/* Dismiss tooltip when clicking image background */}
        {activeId !== null && (
          <div className="absolute inset-0 z-0" onClick={() => setActiveId(null)} />
        )}

        {items.map((item, i) => {
          const fallback = FALLBACK_POSITIONS[i] ?? { x: 20, y: 20 };
          const x = item.x ?? fallback.x;
          const y = item.y ?? fallback.y;
          const isActive = activeId === item.id;
          // Flip tooltip to stay inside image bounds
          const tipLeft = x > 55;
          const tipUp   = y > 55;

          return (
            <div
              key={item.id}
              className="absolute z-10"
              style={{ left: `${x}%`, top: `${y}%`, transform: "translate(-50%, -50%)" }}
            >
              {/* Clickable dot */}
              <button
                onClick={(e) => { e.stopPropagation(); setActiveId(isActive ? null : item.id); }}
                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold cursor-pointer transition-transform hover:scale-110 focus:outline-none"
                style={{
                  background: isActive ? "#fff" : GOLD,
                  color: isActive ? GOLD : "#000",
                  border: "2px solid #fff",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.6)",
                }}
              >
                {i + 1}
              </button>

              {/* Tooltip popup */}
              {isActive && (
                <div
                  className="absolute z-20 rounded-xl p-3 text-left shadow-2xl"
                  style={{
                    background: "#111",
                    border: `1px solid ${GOLD}`,
                    width: 190,
                    ...(tipLeft  ? { right: "calc(100% + 10px)" } : { left: "calc(100% + 10px)" }),
                    ...(tipUp    ? { bottom: 0 }                  : { top: 0 }),
                  }}
                >
                  <p className="text-xs text-zinc-400 uppercase tracking-wide mb-0.5">
                    {categoryIcon(item.category)} {item.category}
                  </p>
                  <p className="text-sm font-bold leading-tight" style={{ color: GOLD }}>
                    {item.brand} {item.model}
                    {item.year ? ` (${item.year})` : ""}
                  </p>
                  <p className="text-xs text-white mt-1">{item.priceRange}</p>
                  <p className="text-xs text-zinc-500 mt-0.5">{item.confidence}% confidence</p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Item cards */}
      {items.map((item, i) => <ItemCard key={item.id} item={item} index={i} />)}

      {/* Total value */}
      {(totalMin > 0 || totalMax > 0) && (
        <div className="w-full rounded-xl py-4 px-5 text-center"
          style={{ background: "rgba(201,168,76,0.12)", border: `1px solid ${GOLD}` }}>
          <p className="text-sm text-zinc-400">💰 Total Estimated Value</p>
          <p className="text-2xl font-bold mt-1" style={{ color: GOLD }}>
            ${totalMin.toLocaleString("en-US")} – ${totalMax.toLocaleString("en-US")}
          </p>
        </div>
      )}
    </div>
  );
}

// ─── UNLOCKED — CAR MODE ──────────────────────────────────────────────────────

function CarMode({ claudeRaw, items }: { claudeRaw: unknown; items: FullItem[] }) {
  const carItem = items[0];

  if (isCarHigh(claudeRaw)) {
    return (
      <div className="flex flex-col gap-5 w-full">
        {/* Car identity card */}
        <div className="rounded-2xl p-5" style={{ background: "#111", border: "1px solid #222" }}>
          <p className="text-xs text-zinc-500 uppercase tracking-wide mb-1">Vehicle Identified</p>
          <p className="text-2xl font-bold" style={{ color: GOLD }}>
            {claudeRaw.brand}{claudeRaw.series ? ` ${claudeRaw.series}` : ""} {claudeRaw.model}
          </p>
          {claudeRaw.limitingFactors && claudeRaw.limitingFactors.length > 0 && (
            <p className="text-xs text-zinc-500 mt-2">
              Note: {claudeRaw.limitingFactors.join(", ")}
            </p>
          )}
        </div>

        {/* Year table */}
        {claudeRaw.possibleYears && claudeRaw.possibleYears.length > 0 && (
          <div className="rounded-2xl overflow-hidden" style={{ border: "1px solid #222" }}>
            <div className="px-4 py-3" style={{ background: "#111" }}>
              <p className="text-xs text-zinc-500 uppercase tracking-wide">Possible Model Years</p>
            </div>
            {claudeRaw.possibleYears.map((yr, i) => (
              <div key={yr}
                className="flex justify-between items-center px-4 py-3 text-sm"
                style={{ background: i % 2 === 0 ? "#111" : "#0e0e0e", borderTop: "1px solid #1a1a1a" }}>
                <span className="text-white font-medium">{yr}</span>
                <span style={{ color: GOLD }}>{carItem?.priceRange ?? "N/A"}</span>
              </div>
            ))}
          </div>
        )}

        {/* Listings */}
        {carItem && carItem.listings.length > 0 && (
          <div className="flex flex-col gap-3">
            <p className="text-xs text-zinc-500 uppercase tracking-wide px-1">Market Listings</p>
            {carItem.listings.map((l, i) => (
              <a key={i} href={l.link} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-3 rounded-xl p-3"
                style={{ background: "#111", border: "1px solid #222" }}>
                {l.thumbnail && l.thumbnail !== "blurred" ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={l.thumbnail} alt={l.title} className="w-14 h-14 rounded-lg object-cover flex-shrink-0" />
                ) : (
                  <div className="w-14 h-14 rounded-lg bg-zinc-800 flex-shrink-0 flex items-center justify-center text-2xl">🚗</div>
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-zinc-300 truncate">{l.title}</p>
                  <p className="text-base font-bold mt-0.5" style={{ color: GOLD }}>{l.price}</p>
                </div>
                <span className="text-zinc-600 text-xs">↗</span>
              </a>
            ))}
          </div>
        )}
        <p className="text-xs text-zinc-600 text-center px-2">
          💡 Photograph the rear badge for exact year confirmation
        </p>
      </div>
    );
  }

  if (isCarLow(claudeRaw)) {
    return (
      <div className="flex flex-col gap-5 w-full">
        <div className="rounded-2xl p-4" style={{ background: "#111", border: "1px solid #222" }}>
          <p className="text-sm font-semibold text-white mb-3">
            📷 Limited image data — top possibilities:
          </p>
          {claudeRaw.candidates.map((c) => (
            <div key={`${c.brand}-${c.model}`} className="mb-3">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-white">{c.brand} {c.model}</span>
                <span style={{ color: GOLD }}>{c.probability}%</span>
              </div>
              <div className="w-full h-2 rounded-full bg-zinc-800">
                <div className="h-full rounded-full" style={{ width: `${c.probability}%`, background: GOLD }} />
              </div>
            </div>
          ))}
          {claudeRaw.limitingFactors && claudeRaw.limitingFactors.length > 0 && (
            <p className="text-xs text-zinc-500 mt-3">
              Limiting factors: {claudeRaw.limitingFactors.join(", ")}
            </p>
          )}
        </div>
        <p className="text-xs text-zinc-500 text-center">
          💡 Try photographing from a different angle
        </p>
      </div>
    );
  }

  return null;
}

// ─── UNLOCKED STATE (wrapper) ─────────────────────────────────────────────────

interface UnlockedProps {
  imageUrl: string;
  unlockData: UnlockData;
  credits: number;
  onScanAnother: () => void;
}

function UnlockedState({ imageUrl, unlockData, credits, onScanAnother }: UnlockedProps) {
  const { mode, items, claudeRaw } = unlockData;

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-md mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold self-start" style={{ color: GOLD }}>WealthLens</h1>

      {/* Mode badge */}
      <div className="self-start flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
        style={{ background: "rgba(201,168,76,0.1)", border: `1px solid rgba(201,168,76,0.3)`, color: GOLD }}>
        {MODE_OPTIONS.find((m) => m.mode === mode)?.emoji} Results Unlocked
      </div>

      {/* Content based on mode */}
      {mode === "person" && <PersonMode imageUrl={imageUrl} items={items} />}
      {mode === "car"    && <CarMode claudeRaw={claudeRaw} items={items} />}
      {mode === "item"   && items.length > 0 && (
        <div className="w-full flex flex-col gap-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={imageUrl} alt="Item" className="w-full rounded-2xl object-cover" style={{ maxHeight: 320 }} />
          <ItemCard item={items[0]} index={0} />
          {items[0].minPrice > 0 || items[0].maxPrice > 0 ? (
            <div className="w-full rounded-xl py-4 px-5 text-center"
              style={{ background: "rgba(201,168,76,0.12)", border: `1px solid ${GOLD}` }}>
              <p className="text-sm text-zinc-400">💰 Estimated Value</p>
              <p className="text-2xl font-bold mt-1" style={{ color: GOLD }}>
                ${items[0].minPrice.toLocaleString("en-US")} – ${items[0].maxPrice.toLocaleString("en-US")}
              </p>
            </div>
          ) : null}
        </div>
      )}

      {/* Footer */}
      <div className="w-full flex flex-col gap-4 pt-2">
        {/* Credits */}
        <div className="flex justify-center">
          <span className="text-sm px-4 py-2 rounded-full"
            style={{ background: "#111", border: "1px solid #2a2a2a", color: credits > 0 ? GOLD : "#888" }}>
            💳 {credits} {credits === 1 ? "credit" : "credits"} remaining
          </span>
        </div>

        {/* Scan Another */}
        <button onClick={onScanAnother}
          className="w-full py-3 rounded-xl font-semibold text-sm transition-all hover:opacity-80"
          style={{ background: GOLD, color: "#0a0a0a" }}>
          🔄 Scan Another
        </button>

        {/* Disclaimer */}
        <p className="text-xs text-zinc-600 text-center leading-relaxed px-2">
          AI estimates for reference only. Not financial or authentication advice.
          WealthLens is not liable for decisions made based on these results.
        </p>
      </div>
    </div>
  );
}

// ─── MAIN STATE MACHINE ───────────────────────────────────────────────────────

export default function WealthLens() {
  const [appState, setAppState]       = useState<AppState>("home");
  const [mode, setMode]               = useState<ScanMode>("person");
  const [user, setUser]               = useState<LocalUser | null>(null);
  const [imageUrl, setImageUrl]       = useState<string>("");
  const [analyzeData, setAnalyzeData] = useState<AnalyzeResponse | null>(null);
  const [unlockData, setUnlockData]   = useState<UnlockData | null>(null);
  const [error, setError]             = useState<string | null>(null);

  // Initialise user from localStorage (client-side only)
  useEffect(() => { setUser(initUser()); }, []);

  const handleUserUpdate = useCallback((updated: LocalUser) => {
    saveUser(updated);
    setUser(updated);
  }, []);

  const handleImageSelected = useCallback(async (file: File) => {
    if (!user) return;
    setError(null);
    // Show preview URL immediately
    const previewUrl = URL.createObjectURL(file);
    setImageUrl(previewUrl);
    setAppState("loading");

    try {
      const base64 = await compressToBase64(file);
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64, mode, uuid: user.uuid }),
      });
      const data = await res.json() as AnalyzeResponse & { error?: string };
      if (!res.ok) {
        setError(data.error ?? "Analysis failed. Please try again.");
        setAppState("home");
        return;
      }
      setAnalyzeData(data);
      setAppState("locked");
    } catch {
      setError("Network error. Please check your connection and try again.");
      setAppState("home");
    }
  }, [user, mode]);

  const handleUnlocked = useCallback((data: UnlockData) => {
    setUnlockData(data);
    setAppState("unlocked");
  }, []);

  const handleScanAnother = useCallback(() => {
    setAppState("home");
    setAnalyzeData(null);
    setUnlockData(null);
    setImageUrl("");
    setError(null);
  }, []);

  return (
    <div style={{ background: "#0a0a0a", minHeight: "100vh" }}>
      {appState === "home" && (
        <HomeState
          mode={mode}
          onModeChange={setMode}
          onImageSelected={handleImageSelected}
          error={error}
        />
      )}

      {appState === "loading" && (
        <LoadingState imageUrl={imageUrl} />
      )}

      {appState === "locked" && analyzeData && user && (
        <LockedState
          imageUrl={imageUrl}
          analyzeData={analyzeData}
          user={user}
          onUnlocked={handleUnlocked}
          onUserUpdate={handleUserUpdate}
        />
      )}

      {appState === "unlocked" && unlockData && user && (
        <UnlockedState
          imageUrl={imageUrl}
          unlockData={unlockData}
          credits={user.credits}
          onScanAnother={handleScanAnother}
        />
      )}
    </div>
  );
}
