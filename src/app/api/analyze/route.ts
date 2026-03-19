import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { kv } from "@vercel/kv";
import { v4 as uuidv4 } from "uuid";

// ─── Types ────────────────────────────────────────────────────────────────────

type Mode = "person" | "car" | "item";

type ImageMediaType = "image/jpeg" | "image/png" | "image/gif" | "image/webp";

interface VisualAnomaly {
  description: string;
  riskWeight: number;
}

interface ClaudeItem {
  id?: number;
  category: string;
  brand: string;
  model: string;
  year?: string;
  confidence: number;
  visualAnomalies?: VisualAnomaly[];
  searchQuery: string;
  x?: number;
  y?: number;
}

interface ClaudeCarHigh {
  confidence: number;
  brand: string;
  series?: string;
  model: string;
  possibleYears?: string[];
  limitingFactors?: string[];
}

interface ClaudeCarLow {
  confidence: number;
  candidates: { brand: string; model: string; probability: number }[];
  limitingFactors?: string[];
}

interface SerperListing {
  title: string;
  price: string;
  thumbnail: string;
  link: string;
}

interface EnrichedItem {
  id: number;
  category: string;
  brand: string;
  model: string;
  confidence: number;
  visualAnomalies: VisualAnomaly[];
  searchQuery: string;
  priceRange: string;
  riskScore: number;
  riskLevel: "low" | "medium" | "high";
  listings: SerperListing[];
  minPrice: number;
  maxPrice: number;
  x?: number;
  y?: number;
}

// ─── System Prompts ───────────────────────────────────────────────────────────

const SYSTEM_PROMPTS: Record<Mode, string> = {
  person: `You are a fashion and goods expert.
Analyze this image and identify every visible item in the photo including clothing, shoes, bags, accessories, watches, and jewelry regardless of brand or price point.
Focus on the items themselves, not who is wearing them.
If the brand is not identifiable, use "Unknown" for the brand field.
Only identify what is clearly visible, do not guess hidden items.
Return ONLY a valid JSON array with this exact structure:
[{
  "id": 1,
  "category": "bag",
  "brand": "Louis Vuitton",
  "model": "Neverfull MM",
  "confidence": 90,
  "visualAnomalies": [
    {"description": "stitching pattern inconsistent", "riskWeight": 25}
  ],
  "searchQuery": "Louis Vuitton Neverfull MM resale price 2024",
  "x": 35,
  "y": 60
}]
x and y are the estimated percentage position (0-100) of the item's center within the image, where x=0 is the left edge, x=100 is the right edge, y=0 is the top edge, y=100 is the bottom edge.
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`,

  car: `You are an automotive expert.
Analyze this image and identify the vehicle.
Do NOT guess the exact year. List all possible production years for this model.
If confidence >= 70, return:
{
  "confidence": 85,
  "brand": "BMW",
  "series": "4 Series",
  "model": "428i",
  "possibleYears": ["2014","2015","2016","2017","2018"],
  "limitingFactors": ["side angle only"]
}
If confidence < 70, return:
{
  "confidence": 55,
  "candidates": [
    {"brand": "BMW", "model": "428i", "probability": 60},
    {"brand": "BMW", "model": "430i", "probability": 25},
    {"brand": "Mercedes", "model": "C300", "probability": 15}
  ],
  "limitingFactors": ["rear angle only"]
}
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`,

  item: `You are a luxury goods expert.
Analyze this single item image.
Return ONLY valid JSON with this exact structure:
{
  "category": "handbag",
  "brand": "Chanel",
  "model": "Classic Flap Medium",
  "year": "2022",
  "confidence": 88,
  "visualAnomalies": [
    {"description": "logo font spacing abnormal", "riskWeight": 25},
    {"description": "hardware color inconsistent", "riskWeight": 20}
  ],
  "searchQuery": "Chanel Classic Flap Medium 2022 resale price"
}
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Strips the data-URL prefix (e.g. "data:image/png;base64,") if present,
 * and returns the raw base64 string together with a detected media type.
 */
function parseBase64Image(raw: string): { data: string; mediaType: ImageMediaType } {
  const dataUrlMatch = raw.match(/^data:(image\/[\w+]+);base64,(.+)$/);
  if (dataUrlMatch) {
    return {
      mediaType: dataUrlMatch[1] as ImageMediaType,
      data: dataUrlMatch[2],
    };
  }
  // Detect from first bytes of raw base64
  let mediaType: ImageMediaType = "image/jpeg";
  if (raw.startsWith("iVBORw0KGgo")) mediaType = "image/png";
  else if (raw.startsWith("R0lGODlh")) mediaType = "image/gif";
  else if (raw.startsWith("UklGR")) mediaType = "image/webp";
  return { data: raw, mediaType };
}

/** Sum riskWeight values across all visual anomalies. */
function sumRiskScore(anomalies: VisualAnomaly[]): number {
  return (anomalies ?? []).reduce((acc, a) => acc + (a.riskWeight ?? 0), 0);
}

/** Map numeric risk score to a label. */
function toRiskLevel(score: number): "low" | "medium" | "high" {
  if (score <= 20) return "low";
  if (score <= 50) return "medium";
  return "high";
}

/** Format a dollar total as an approximate blurred string, e.g. "$15,000+". */
function formatTotalBlurred(value: number): string {
  if (value <= 0) return "N/A";
  // Round down to the nearest 100 to avoid false precision
  const floored = Math.floor(value / 100) * 100;
  return `$${floored.toLocaleString("en-US")}+`;
}

/** Pull numeric dollar values out of Serper listing price strings. */
function extractPrices(listings: SerperListing[]): number[] {
  return listings
    .map((l) => parseFloat((l.price ?? "").replace(/[^0-9.]/g, "")))
    .filter((n) => n > 0);
}

/** Fetch top-3 Serper shopping results for a search query. */
async function fetchSerperListings(query: string): Promise<SerperListing[]> {
  try {
    const res = await fetch("https://google.serper.dev/shopping", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-KEY": process.env.SERPER_API_KEY!,
      },
      body: JSON.stringify({ q: query }),
    });
    if (!res.ok) return [];
    const data = (await res.json()) as { shopping?: Record<string, string>[] };
    return (data.shopping ?? []).slice(0, 3).map((r) => ({
      title: r.title ?? "",
      price: r.price ?? "",
      thumbnail: r.imageUrl ?? r.thumbnailUrl ?? r.thumbnail ?? "",
      link: `https://www.google.com/search?tbm=shop&q=${encodeURIComponent(query)}`,
    }));
  } catch {
    return [];
  }
}

// Extract JSON from Claude response - handle markdown code blocks and extra text
function extractJSON(text: string): string {
  // Remove markdown code blocks
  text = text.replace(/```json\n?/g, '').replace(/```\n?/g, '');

  // Try to find JSON array or object
  const arrayMatch = text.match(/\[[\s\S]*\]/);
  const objectMatch = text.match(/\{[\s\S]*\}/);

  // Return whichever outermost structure appears first in the text.
  // This prevents the array regex from matching a nested array (e.g.
  // visualAnomalies) inside a top-level object, which would cause the
  // object's fields to be lost and its array indices spread as "0", "1", etc.
  if (arrayMatch && objectMatch) {
    return arrayMatch.index! < objectMatch.index! ? arrayMatch[0] : objectMatch[0];
  }
  if (arrayMatch) return arrayMatch[0];
  if (objectMatch) return objectMatch[0];

  return text.trim();
}

// ─── Route Handler ────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    console.log("KEY CHECK:", process.env.OPENAI_API_KEY?.slice(0, 12));

    // ── 1. Parse & validate request body ─────────────────────────────────────
    const body = (await req.json()) as {
      image?: string;
      mode?: string;
      uuid?: string;
    };
    const { image, mode, uuid } = body;

    if (!image || !mode || !uuid) {
      return NextResponse.json(
        { error: "Missing required fields: image, mode, uuid" },
        { status: 400 }
      );
    }
    if (!["person", "car", "item"].includes(mode)) {
      return NextResponse.json(
        { error: "Invalid mode. Must be one of: person, car, item" },
        { status: 400 }
      );
    }

    const typedMode = mode as Mode;

    // ── 2. Call GPT-5.4 API with the image ───────────────────────────────────
    const { data: imageData, mediaType } = parseBase64Image(image);
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 2048,
      messages: [
        {
          role: "system",
          content: SYSTEM_PROMPTS[typedMode],
        },
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: {
                url: `data:${mediaType};base64,${imageData}`,
                detail: "high",
              },
            },
            {
              type: "text",
              text: "Analyze this image and return the JSON as instructed.",
            },
          ],
        },
      ],
    });

    // ── 3. Parse GPT-5.4's JSON response ─────────────────────────────────────
    const rawText = response.choices[0].message.content || "";
    console.log("Raw GPT-4o text:", rawText);
    const cleanedText = extractJSON(rawText);

    let claudeParsed: unknown;
    try {
      claudeParsed = JSON.parse(cleanedText);
      console.log("Claude parsed result:", JSON.stringify(claudeParsed, null, 2));
    } catch {
      return NextResponse.json(
        { error: "AI returned non-JSON response", raw: rawText },
        { status: 502 }
      );
    }

    // ── 4. Normalise to a flat array of items + extract searchQueries ─────────
    let normalizedItems: ClaudeItem[];
    let carRaw: ClaudeCarHigh | ClaudeCarLow | null = null;

    if (typedMode === "car") {
      const car = claudeParsed as ClaudeCarHigh | ClaudeCarLow;
      carRaw = car;
      const isHighConfidence = "brand" in car;
      const brand = isHighConfidence
        ? (car as ClaudeCarHigh).brand
        : (car as ClaudeCarLow).candidates[0]?.brand ?? "Unknown";
      const carModel = isHighConfidence
        ? (car as ClaudeCarHigh).model
        : (car as ClaudeCarLow).candidates[0]?.model ?? "Unknown";
      const series = isHighConfidence ? (car as ClaudeCarHigh).series : undefined;

      normalizedItems = [
        {
          id: 1,
          category: "car",
          brand,
          model: carModel,
          confidence: car.confidence,
          visualAnomalies: [],
          searchQuery: `${brand}${series ? " " + series : ""} ${carModel} market value resale price`,
        },
      ];
    } else if (typedMode === "item") {
      const item = claudeParsed as ClaudeItem;
      normalizedItems = [{ ...item, id: 1 }];
    } else {
      // person — GPT-4o returns an array
      normalizedItems = claudeParsed as ClaudeItem[];
    }

    // ── 5. Call Serper in parallel for every item ─────────────────────────────
    const allListings = await Promise.all(
      normalizedItems.map((item) => fetchSerperListings(item.searchQuery))
    );

    // ── 6. Assemble enriched items ────────────────────────────────────────────
    const enrichedItems: EnrichedItem[] = normalizedItems.map((item, i) => {
      const listings = allListings[i];
      const prices = extractPrices(listings);
      const minPrice = prices.length ? Math.min(...prices) : 0;
      const maxPrice = prices.length ? Math.max(...prices) : 0;
      const score = sumRiskScore(item.visualAnomalies ?? []);

      return {
        ...item,
        id: item.id ?? i + 1,
        visualAnomalies: item.visualAnomalies ?? [],
        priceRange:
          prices.length
            ? `$${minPrice.toLocaleString("en-US")} - $${maxPrice.toLocaleString("en-US")}`
            : "N/A",
        riskScore: score,
        riskLevel: toRiskLevel(score),
        listings,
        minPrice,
        maxPrice,
      };
    });

    // ── 7. Generate scanId ────────────────────────────────────────────────────
    const scanId = uuidv4();

    // ── 8. Persist full record in Vercel KV (TTL = 24 h) ─────────────────────
    const fullRecord = {
      scanId,
      uuid,
      mode: typedMode,
      timestamp: Date.now(),
      claudeRaw: carRaw ?? claudeParsed,
      items: enrichedItems,
    };
    await kv.set(`scan_${scanId}`, fullRecord, { ex: 86400 });

    // ── 9. Calculate aggregate risk score ─────────────────────────────────────
    // (Each item already has its own riskScore computed above.)
    // The totalValue is the sum of each item's highest observed listing price.
    const totalValue = enrichedItems.reduce((sum, item) => sum + item.maxPrice, 0);

    // ── 10. Build blurred response for the frontend ───────────────────────────
    const blurredItems = enrichedItems.map((item) => ({
      id: item.id,
      category: "***",
      brand: "***",
      model: "***",
      priceRange: "*** - ***",
      riskScore: item.riskScore,
      riskLevel: item.riskLevel,
      listings: item.listings.map((l) => ({
        title: "***",
        price: "***",
        thumbnail: "blurred",
        link: "***",
      })),
    }));

    return NextResponse.json({
      scanId,
      itemCount: enrichedItems.length,
      totalValueBlurred: formatTotalBlurred(totalValue),
      items: blurredItems,
    });
  } catch (error) {
    console.error("[/api/analyze]", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
