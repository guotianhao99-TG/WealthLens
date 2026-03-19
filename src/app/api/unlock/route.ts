import { NextRequest, NextResponse } from "next/server";
import { kv } from "@vercel/kv";

// ─── Types ────────────────────────────────────────────────────────────────────

interface UserRecord {
  credits: number;
  email: string;
  scans: number;
  createdAt: number;
}

// ─── Route Handler ────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    // ── 1. Parse & validate request body ─────────────────────────────────────
    const body = (await req.json()) as { uuid?: string; scanId?: string };
    const { uuid, scanId } = body;

    if (!uuid || !scanId) {
      return NextResponse.json(
        { error: "Missing required fields: uuid, scanId" },
        { status: 400 }
      );
    }

    // ── 2. Fetch user from KV ────────────────────────────────────────────────
    const user = await kv.get<UserRecord>(`user_${uuid}`);

    // ── 3. User not found ────────────────────────────────────────────────────
    if (!user) {
      return NextResponse.json(
        { error: "User not found. Please register first." },
        { status: 404 }
      );
    }

    // ── 4. Insufficient credits ──────────────────────────────────────────────
    if (user.credits < 2) {
      return NextResponse.json(
        { error: "Insufficient credits", credits: user.credits },
        { status: 402 }
      );
    }

    // ── 5. Fetch scan from KV ────────────────────────────────────────────────
    const scan = await kv.get(`scan_${scanId}`);

    // ── 6. Scan not found / expired ──────────────────────────────────────────
    if (!scan) {
      return NextResponse.json(
        { error: "Scan expired or not found. Please scan again." },
        { status: 404 }
      );
    }

    // ── 7. Deduct 2 credits ───────────────────────────────────────────────────
    // ── 8. Increment scans counter ────────────────────────────────────────────
    const updatedUser: UserRecord = {
      ...user,
      credits: user.credits - 2,
      scans: user.scans + 1,
    };
    await kv.set(`user_${uuid}`, updatedUser);

    // ── 9. Return full unmasked scan data ─────────────────────────────────────
    console.log("KV scan data:", JSON.stringify(scan, null, 2));
    return NextResponse.json(scan);
  } catch (error) {
    console.error("[/api/unlock]", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
