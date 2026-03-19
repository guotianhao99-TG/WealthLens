import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { kv } from "@vercel/kv";
import { v4 as uuidv4 } from "uuid";
import sharp from "sharp";

// ─── Types ────────────────────────────────────────────────────────────────────

type Mode = "person" | "car" | "item";

type ImageMediaType = "image/jpeg" | "image/png" | "image/gif" | "image/webp";

interface VisualAnomaly {
  description: string;
  riskWeight: number;
}

// Stage-1 detection result
interface DetectedItem {
  id: number;
  category: string;
  bbox: { top: number; left: number; width: number; height: number };
}

// Stage-2 brand detection result
interface BrandDetectionResult {
  hasIdentifiableFeature: boolean;
  brand: string;
  identifiedFeature: string;
  confidence: number;
}

// Stage-3 model matching result
interface ModelMatchResult {
  model: string;
  year?: string;
  visualAnomalies?: VisualAnomaly[];
  searchQuery: string;
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

// Stage-1: Detect all items + faces, return bboxes only
const PERSON_STAGE1_PROMPT = `Identify all visible items in this image (clothing, bags, shoes, accessories, watches, jewelry).
Also detect any human faces and return their bounding boxes.
Return ONLY valid JSON with this exact structure:
{
  "items": [
    { "id": 1, "category": "bag", "bbox": { "top": 20, "left": 30, "width": 15, "height": 20 } }
  ],
  "faces": [
    { "x": 42, "y": 5, "w": 18, "h": 22 }
  ]
}
Bounding box values are percentages of image dimensions (0-100).
For items: bbox.top/left are the top-left corner; bbox.width/height are the size.
For faces: x/y are the top-left corner; w/h are the size.
Never place a hotspot on a person's face. Only detect clothing, bags, shoes, accessories, watches, and jewelry.
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`;

// Stage-2: Brand detection from cropped item image
function personStage2Prompt(category: string): string {
  return `Look for any logo, label, hardware, pattern, or distinctive feature on this ${category}.
Return ONLY valid JSON with this exact structure:
{
  "hasIdentifiableFeature": true,
  "brand": "Louis Vuitton",
  "identifiedFeature": "monogram canvas pattern",
  "confidence": 85
}
If no identifiable brand feature exists, return:
{
  "hasIdentifiableFeature": false,
  "brand": "Unknown",
  "identifiedFeature": "",
  "confidence": 0
}
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`;
}

// Stage-3: Model matching from cropped item image, given a known brand
function personStage3Prompt(brand: string, category: string): string {
  return `This is a ${brand} ${category}. Based on the overall shape, size, hardware, and details, identify the specific model.
Return ONLY valid JSON with this exact structure:
{
  "model": "Neverfull MM",
  "year": "2022",
  "visualAnomalies": [
    { "description": "stitching pattern inconsistent", "riskWeight": 25 }
  ],
  "searchQuery": "Louis Vuitton Neverfull MM resale price 2024"
}
If no visual anomalies are detected, return an empty array for visualAnomalies.
Return ONLY valid JSON without any markdown formatting, code blocks, or preambles.`;
}

const SYSTEM_PROMPTS: Record<Exclude<Mode, "person">, string> = {
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

/**
 * Crop a region out of a base64-encoded image using sharp.
 * bbox values are percentages of image dimensions (0-100).
 * Returns a base64-encoded JPEG of the cropped region.
 */
async function cropImage(
  base64Data: string,
  mediaType: ImageMediaType,
  bbox: { top: number; left: number; width: number; height: number }
): Promise<string> {
  const buffer = Buffer.from(base64Data, "base64");
  const img = sharp(buffer);
  const meta = await img.metadata();
  const imgW = meta.width ?? 800;
  const imgH = meta.height ?? 800;

  // Clamp to image bounds
  const left   = Math.max(0, Math.round((bbox.left   / 100) * imgW));
  const top    = Math.max(0, Math.round((bbox.top    / 100) * imgH));
  const width  = Math.min(imgW - left, Math.max(1, Math.round((bbox.width  / 100) * imgW)));
  const height = Math.min(imgH - top,  Math.max(1, Math.round((bbox.height / 100) * imgH)));

  const cropped = await img
    .extract({ left, top, width, height })
    .jpeg({ quality: 90 })
    .toBuffer();

  return cropped.toString("base64");
}

// ─── Person mode — three-stage pipeline ──────────────────────────────────────

async function runPersonPipeline(
  openai: OpenAI,
  imageData: string,
  mediaType: ImageMediaType
): Promise<{ items: ClaudeItem[]; faces: FaceRegion[] }> {

  // ── Stage 1: detect all items + faces ──────────────────────────────────────
  const stage1Response = await openai.chat.completions.create({
    model: "gpt-4o",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: `data:${mediaType};base64,${imageData}`, detail: "high" },
          },
          { type: "text", text: PERSON_STAGE1_PROMPT },
        ],
      },
    ],
  });

  const stage1Raw = stage1Response.choices[0].message.content ?? "";
  console.log("Stage-1 raw:", stage1Raw);

  let detectedItems: DetectedItem[] = [];
  let faces: FaceRegion[] = [];

  try {
    const parsed = JSON.parse(extractJSON(stage1Raw)) as {
      items?: DetectedItem[];
      faces?: FaceRegion[];
    };
    detectedItems = parsed.items ?? [];
    faces = parsed.faces ?? [];
  } catch {
    console.error("Stage-1 JSON parse failed:", stage1Raw);
    return { items: [], faces: [] };
  }

  console.log(`Stage-1 detected ${detectedItems.length} items, ${faces.length} faces`);

  // ── Stages 2 & 3: run per-item pipeline in parallel ────────────────────────
  const results = await Promise.all(
    detectedItems.map(async (detected): Promise<ClaudeItem> => {
      // Dot sits at the bbox centre
      const x = detected.bbox.left + detected.bbox.width  / 2;
      const y = detected.bbox.top  + detected.bbox.height / 2;

      try {
        const croppedBase64 = await cropImage(imageData, mediaType, detected.bbox);
        const croppedUrl = `data:image/jpeg;base64,${croppedBase64}`;

        // ── Stage 2: brand detection ──────────────────────────────────────────
        const stage2Response = await openai.chat.completions.create({
          model: "gpt-4o",
          max_tokens: 256,
          messages: [
            {
              role: "user",
              content: [
                { type: "image_url", image_url: { url: croppedUrl, detail: "high" } },
                { type: "text", text: personStage2Prompt(detected.category) },
              ],
            },
          ],
        });

        const stage2Raw = stage2Response.choices[0].message.content ?? "";
        console.log(`Stage-2 item ${detected.id} (${detected.category}):`, stage2Raw);

        const brandResult = JSON.parse(extractJSON(stage2Raw)) as BrandDetectionResult;

        // ── Stage 3: model matching ───────────────────────────────────────────
        if (brandResult.hasIdentifiableFeature && brandResult.brand !== "Unknown") {
          const stage3Response = await openai.chat.completions.create({
            model: "gpt-4o",
            max_tokens: 384,
            messages: [
              {
                role: "user",
                content: [
                  { type: "image_url", image_url: { url: croppedUrl, detail: "high" } },
                  { type: "text", text: personStage3Prompt(brandResult.brand, detected.category) },
                ],
              },
            ],
          });

          const stage3Raw = stage3Response.choices[0].message.content ?? "";
          console.log(`Stage-3 item ${detected.id}:`, stage3Raw);

          const modelResult = JSON.parse(extractJSON(stage3Raw)) as ModelMatchResult;

          return {
            id: detected.id,
            category: detected.category,
            brand: brandResult.brand,
            model: modelResult.model ?? "Unknown",
            year: modelResult.year,
            confidence: brandResult.confidence,
            visualAnomalies: modelResult.visualAnomalies ?? [],
            searchQuery: modelResult.searchQuery ??
              `${brandResult.brand} ${modelResult.model} resale price`,
            x,
            y,
          };
        }

        // No identifiable brand — skip stage 3, build a descriptive search query
        const featureHint = brandResult.identifiedFeature
          ? ` ${brandResult.identifiedFeature}` : "";
        return {
          id: detected.id,
          category: detected.category,
          brand: "Unknown",
          model: "Similar style",
          confidence: 40,
          visualAnomalies: [],
          searchQuery: `${detected.category}${featureHint} similar style price`,
          x,
          y,
        };

      } catch (err) {
        console.error(`Pipeline failed for item ${detected.id}:`, err);
        return {
          id: detected.id,
          category: detected.category,
          brand: "Unknown",
          model: "Unknown",
          confidence: 50,
          visualAnomalies: [],
          searchQuery: `${detected.category} resale price`,
          x,
          y,
        };
      }
    })
  );

  return { items: results, faces };
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

    // ── 2. Call GPT-4o (mode-specific logic) ─────────────────────────────────
    let normalizedItems: ClaudeItem[] = [];
    let carRaw: ClaudeCarHigh | ClaudeCarLow | null = null;
    let faces: FaceRegion[] = [];

    if (typedMode === "person") {
      // Three-stage pipeline: detect → brand → model
      const result = await runPersonPipeline(openai, imageData, mediaType);
      normalizedItems = result.items;
      faces = result.faces;

    } else {
      // Car / item: single-shot as before
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
      } else {
        // item mode
        const item = claudeParsed as ClaudeItem;
        normalizedItems = [{ ...item, id: 1 }];
      }
    }

    // ── 3. Call Serper in parallel for every item ─────────────────────────────
    const allListings = await Promise.all(
      normalizedItems.map((item) => fetchSerperListings(item.searchQuery))
    );

    // ── 4. Assemble enriched items ────────────────────────────────────────────
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

    // ── 5. Generate scanId and persist to KV (TTL = 24 h) ────────────────────
    const scanId = uuidv4();
    const fullRecord = {
      scanId,
      uuid,
      mode: typedMode,
      timestamp: Date.now(),
      claudeRaw: carRaw ?? (typedMode === "person" ? normalizedItems : undefined),
      items: enrichedItems,
      faces,
    };
    await kv.set(`scan_${scanId}`, fullRecord, { ex: 86400 });

    // ── 6. Calculate aggregate value ──────────────────────────────────────────
    const totalValue = enrichedItems.reduce((sum, item) => sum + item.maxPrice, 0);

    // ── 7. Build blurred response for the frontend ────────────────────────────
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
