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

interface FaceRegion {
  x: number; // % from left edge
  y: number; // % from top edge
  w: number; // % of image width
  h: number; // % of image height
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
Analyze this image and identify every visible item including clothing, shoes, bags, accessories, watches, and jewelry regardless of brand or price point.
Focus on the items themselves, not who is wearing them.
If the brand is not identifiable, use "Unknown" for the brand field and describe the item visually for the model field (e.g. "black leather crossbody bag"). Never leave model empty.
Only identify what is clearly visible, do not guess hidden items.
Also detect any human faces and return their bounding boxes.
Return ONLY a valid JSON object with this exact structure:
{
  "items": [{
    "id": 1,
    "category": "bag",
    "brand": "Louis Vuitton",
    "model": "Neverfull MM",
    "confidence": 90,
    "visualAnomalies": [
      {"description": "stitching pattern inconsistent", "riskWeight": 25}
    ],
    "searchQuery": "Louis Vuitton Neverfull MM new price buy 2024 site:nordstrom.com OR site:net-a-porter.com OR site:farfetch.com OR site:ssense.com",
    "x": 35,
    "y": 60
  }],
  "faces": [
    {"x": 42, "y": 5, "w": 18, "h": 22}
  ]
}
Return x and y as the exact center coordinates of each item as a percentage of the total image dimensions. Be as precise as possible. x=0 is left, x=100 is right, y=0 is top, y=100 is bottom.
Never place a hotspot on a person's face. Only place hotspots on clothing, bags, shoes, accessories, watches, and jewelry.
For faces: x and y are the top-left corner of the face bounding box as a percentage, w and h are the width and height as a percentage of the image dimensions.
If no faces are detected, return an empty array for faces.
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`,

  car: `You are an automotive expert.
Only identify real full-size vehicles. Ignore toys, scale models, posters, and miniatures.
If the image contains only a toy or model car, return { "error": "No real vehicle found" }
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
  "searchQuery": "Chanel Classic Flap Medium new price buy 2024 site:nordstrom.com OR site:net-a-porter.com OR site:farfetch.com OR site:ssense.com"
}
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`,
};

// ─── Search query helpers ─────────────────────────────────────────────────────

/**
 * Brands that are discontinued or exist primarily on the vintage/secondhand
 * market. Items from these brands will use a resale search query instead of
 * a new-retail one. Extend this list as needed.
 */
const VINTAGE_RESALE_BRANDS = new Set([
  "biba", "ossie clark", "halston", "bill blass", "geoffrey beene",
  "courreges", "courrèges", "pierre cardin", "ungaro",
  "gianni versace", "azzedine alaïa", "alaia",
  "helmut lang vintage", "martin margiela vintage",
]);

/** Returns true when a brand is discontinued / primarily a resale brand. */
function isVintageBrand(brand: string): boolean {
  return VINTAGE_RESALE_BRANDS.has(brand.toLowerCase().trim());
}

/**
 * Build the appropriate Serper search query for a fashion/goods item.
 * - Active brands   → new retail query targeting major retail sites.
 * - Unknown brand   → generic new-retail query.
 * - Vintage/discontinued brands → resale query targeting secondhand sites.
 */
function buildItemSearchQuery(brand: string, model: string): string {
  const b = (brand ?? "").trim();
  const m = (model ?? "").trim();
  if (!b || b === "Unknown") {
    return `${m} buy new price 2024 site:nordstrom.com OR site:net-a-porter.com OR site:farfetch.com OR site:ssense.com`;
  }
  if (isVintageBrand(b)) {
    return `${b} ${m} resale secondhand price site:vestiairecollective.com OR site:therealreal.com OR site:1stdibs.com`;
  }
  return `${b} ${m} new price buy 2024 site:nordstrom.com OR site:net-a-porter.com OR site:farfetch.com OR site:ssense.com`;
}

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

/**
 * Fetch automotive pricing results from Serper web search.
 * Uses the /search endpoint so site: operators (cars.com, autotrader, edmunds)
 * are respected — the Serper /shopping endpoint does not index those sites.
 */
async function fetchSerperAutomotiveListings(query: string): Promise<SerperListing[]> {
  try {
    const res = await fetch("https://google.serper.dev/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-KEY": process.env.SERPER_API_KEY!,
      },
      body: JSON.stringify({ q: query, num: 6, gl: "us", hl: "en" }),
    });
    if (!res.ok) return [];
    const data = (await res.json()) as {
      organic?: Array<{ title?: string; snippet?: string; link?: string; imageUrl?: string }>;
    };
    return (data.organic ?? []).slice(0, 3).map((r) => ({
      title: r.title ?? "",
      price: r.snippet ?? "",
      thumbnail: r.imageUrl ?? "",
      link: r.link ?? `https://www.google.com/search?q=${encodeURIComponent(query)}`,
    }));
  } catch {
    return [];
  }
}

// Extract JSON from GPT response — handle markdown code blocks and extra text
function extractJSON(text: string): string {
  text = text.replace(/```json\n?/g, "").replace(/```\n?/g, "");
  const arrayMatch = text.match(/\[[\s\S]*\]/);
  const objectMatch = text.match(/\{[\s\S]*\}/);
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
    const { data: imageData, mediaType } = parseBase64Image(image);
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    // ── 2. Single GPT-4o call for all modes ───────────────────────────────────
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 2048,
      messages: [
        { role: "system", content: SYSTEM_PROMPTS[typedMode] },
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: { url: `data:${mediaType};base64,${imageData}`, detail: "high" },
            },
            { type: "text", text: "Analyze this image and return the JSON as instructed." },
          ],
        },
      ],
    });

    const rawText = response.choices[0].message.content ?? "";
    console.log("Raw GPT-4o response:", rawText);
    const cleanedText = extractJSON(rawText);

    let claudeParsed: unknown;
    try {
      claudeParsed = JSON.parse(cleanedText);
      console.log("Parsed result:", JSON.stringify(claudeParsed, null, 2));
    } catch {
      return NextResponse.json(
        { error: "AI returned non-JSON response", raw: rawText },
        { status: 502 }
      );
    }

    // ── 3. Normalise to a flat array of items + extract faces / carRaw ────────
    let normalizedItems: ClaudeItem[] = [];
    let carRaw: ClaudeCarHigh | ClaudeCarLow | null = null;
    let faces: FaceRegion[] = [];

    if (typedMode === "person") {
      type PersonResponse = { items: ClaudeItem[]; faces?: FaceRegion[] };
      if (Array.isArray(claudeParsed)) {
        normalizedItems = claudeParsed as ClaudeItem[];
      } else {
        const personData = claudeParsed as PersonResponse;
        normalizedItems = personData.items ?? [];
        faces = personData.faces ?? [];
      }
      // Override searchQuery with our retail/resale helper
      normalizedItems = normalizedItems.map((item) => ({
        ...item,
        searchQuery: buildItemSearchQuery(item.brand, item.model),
      }));

    } else if (typedMode === "car") {
      // Handle toy/model car rejection
      const maybeError = claudeParsed as { error?: string };
      if (maybeError.error) {
        return NextResponse.json({ error: maybeError.error }, { status: 422 });
      }

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
      const possibleYears = isHighConfidence ? ((car as ClaudeCarHigh).possibleYears ?? []) : [];
      const yearToken = possibleYears.length
        ? possibleYears[Math.floor(possibleYears.length / 2)]
        : "";
      const seriesToken = series ? ` ${series}` : "";
      const searchQuery = `${yearToken ? yearToken + " " : ""}${brand}${seriesToken} ${carModel} new price OR used price site:cars.com OR site:autotrader.com OR site:edmunds.com`;

      normalizedItems = [{
        id: 1,
        category: "car",
        brand,
        model: carModel,
        confidence: car.confidence,
        visualAnomalies: [],
        searchQuery,
      }];

    } else {
      // item mode
      const item = claudeParsed as ClaudeItem;
      normalizedItems = [{
        ...item,
        id: 1,
        searchQuery: buildItemSearchQuery(item.brand, item.model),
      }];
    }

    // ── 4. Call Serper in parallel for every item ─────────────────────────────
    const allListings = await Promise.all(
      normalizedItems.map((item) =>
        item.category === "car"
          ? fetchSerperAutomotiveListings(item.searchQuery)
          : fetchSerperListings(item.searchQuery)
      )
    );

    // ── 5. Assemble enriched items ────────────────────────────────────────────
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

    // ── 6. Persist to KV (TTL = 24 h) ────────────────────────────────────────
    const scanId = uuidv4();
    const fullRecord = {
      scanId,
      uuid,
      mode: typedMode,
      timestamp: Date.now(),
      claudeRaw: carRaw ?? claudeParsed,
      items: enrichedItems,
      faces,
    };
    await kv.set(`scan_${scanId}`, fullRecord, { ex: 86400 });

    // ── 7. Calculate aggregate value ──────────────────────────────────────────
    const totalValue = enrichedItems.reduce((sum, item) => sum + item.maxPrice, 0);

    // ── 8. Build blurred response for the frontend ────────────────────────────
    const blurredItems = enrichedItems.map((item) => ({
      id: item.id,
      category: "***",
      brand: "***",
      model: "***",
      priceRange: "*** - ***",
      riskScore: item.riskScore,
      riskLevel: item.riskLevel,
      listings: item.listings.map(() => ({
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
      faces,
    });
  } catch (error) {
    console.error("[/api/analyze]", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
