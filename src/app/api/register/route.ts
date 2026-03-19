import { NextRequest, NextResponse } from "next/server";
import { kv } from "@vercel/kv";

// ─── Types ────────────────────────────────────────────────────────────────────

interface UserRecord {
  credits: number;
  email: string;
  scans: number;
  createdAt: number;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Basic RFC-5322-inspired email format check. */
function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// ─── Route Handler ────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    // ── 1. Parse & validate request body ─────────────────────────────────────
    const body = (await req.json()) as { uuid?: string; email?: string };
    const { uuid, email } = body;

    if (!uuid || !email) {
      return NextResponse.json(
        { error: "Missing required fields: uuid, email" },
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

    // ── 2. Check for duplicate email via index ────────────────────────────────
    const existingUuid = await kv.get<string>(`email_index_${email}`);

    // ── 3. Email already registered ───────────────────────────────────────────
    if (existingUuid) {
      return NextResponse.json(
        { error: "Email already registered" },
        { status: 409 }
      );
    }

    // ── 4. Write user record + email index to KV ──────────────────────────────
    const newUser: UserRecord = {
      credits: 4,
      email,
      scans: 0,
      createdAt: Date.now(),
    };

    await Promise.all([
      kv.set(`user_${uuid}`, newUser),
      kv.set(`email_index_${email}`, uuid),
    ]);

    // ── 5. Return success with starting credits ───────────────────────────────
    return NextResponse.json({ success: true, credits: 4 });
  } catch (error) {
    console.error("[/api/register]", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
