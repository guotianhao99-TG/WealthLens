// One-time script: delete all keys from the KV store
// Usage: node scripts/flush-kv.mjs
import { readFileSync } from 'fs';
import { join } from 'path';

// Load .env.local (falls back to already-set env vars via ??=)
try {
  const raw = readFileSync(join(process.cwd(), '.env.local'), 'utf-8');
  for (const line of raw.split('\n')) {
    const m = line.match(/^([^#=][^=]*)=(.*)$/);
    if (m) process.env[m[1].trim()] ??= m[2].trim().replace(/^["']|["']$/g, '');
  }
} catch { /* env vars may already be set in environment */ }

const { KV_REST_API_URL: url, KV_REST_API_TOKEN: token } = process.env;
if (!url || !token) {
  console.error('Error: KV_REST_API_URL and KV_REST_API_TOKEN must be set.');
  process.exit(1);
}

async function cmd(...args) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(args),
  });
  const { result, error } = await r.json();
  if (error) throw new Error(error);
  return result;
}

// SCAN all keys (cursor loop)
const allKeys = [];
let cursor = '0';
do {
  const [nextCursor, keys] = await cmd('SCAN', cursor, 'COUNT', '100');
  cursor = nextCursor;
  allKeys.push(...keys);
} while (cursor !== '0');

if (allKeys.length === 0) {
  console.log('KV store is already empty. 0 keys deleted.');
} else {
  const deleted = await cmd('DEL', ...allKeys);
  console.log(`Deleted ${deleted} key(s):`);
  for (const k of allKeys) console.log(`  - ${k}`);
}
