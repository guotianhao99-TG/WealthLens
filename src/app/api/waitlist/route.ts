import { NextRequest, NextResponse } from "next/server";
import { kv } from "@vercel/kv";

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Basic RFC-5322-inspired email format check. */
function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// ─── Route Handler ────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    // ── 1. Parse & validate request body ─────────────────────────────────────
    const body = (await req.json()) as { email?: string };
    const { email } = body;

    if (!email) {
      return NextResponse.json(
        { error: "Missing required field: email" },
        { status: 400 }
      );
    }

    // ── 1b. Validate email format ─────────────────────────────────────────────
    if (!isValidEmail(email)) {
      return NextResponse.json(
        { error: "Invalid email format" },
        { status: 400 }
      );
    }

    // ── 2. Store in KV ───────────────────────────────────────────────────────
    await kv.set(`waitlist_${email}`, {
      email,
      createdAt: Date.now(),
    });

    // ── 3. Return success ─────────────────────────────────────────────────────
    return NextResponse.json({ success: true, message: "You're on the list!" });
  } catch (error) {
    console.error("[/api/waitlist]", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
