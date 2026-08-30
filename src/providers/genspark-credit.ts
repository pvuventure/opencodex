/**
 * Optional Genspark credit telemetry (Phase 2 of Genspark key failover).
 *
 * A pool entry MAY carry a Genspark browser-session cookie used EXCLUSIVELY to query
 * `GET https://www.genspark.ai/api/payment/get_credit_balance`. The API key alone does
 * not authenticate that payment endpoint (it answers `{"status":-5,"message":"not login"}`
 * without a session), so:
 *
 *   API key = inference authentication (always)
 *   cookie  = optional credit telemetry (never sent to the LLM proxy or any other host)
 *
 * Results are cached in memory with a short TTL so the payment endpoint is never hit on
 * the inference path. A cookie failure of any kind degrades to `unknown`, which keeps the
 * key fully eligible — telemetry can only ever REMOVE a known-drained key from rotation,
 * never disable inference.
 *
 * SECURITY: raw cookie values never appear in logs, management responses, or errors.
 */
import type { OcxProviderConfig } from "../types";
import { isGensparkProvider } from "./genspark";

/** Cached credit state is trusted for this long (spec recommends 30–60 s). */
export const GENSPARK_CREDIT_TTL_MS = 45_000;
/** One bounded probe; a slow payment endpoint must never stall anything. */
const CREDIT_FETCH_TIMEOUT_MS = 5_000;
export const GENSPARK_CREDIT_BALANCE_URL = "https://www.genspark.ai/api/payment/get_credit_balance";
/** Defensive cap when reading the payment response body. */
const CREDIT_BODY_MAX_BYTES = 16_384;

export type GensparkCreditState = "healthy" | "exhausted" | "unknown";

export interface GensparkCreditEntry {
  /** Known numeric balance, or null when the endpoint gave no usable number. */
  balance: number | null;
  /** When this entry was recorded (ms epoch). */
  checkedAt: number;
  /** `exhausted` only on a KNOWN zero/negative balance; anything unclear is `unknown`. */
  state: GensparkCreditState;
}

/** Map<`${providerName}\0${keyId}`, GensparkCreditEntry> */
const creditCache = new Map<string, GensparkCreditEntry>();
/** De-dupes concurrent probes per key so a burst of failures fires one fetch. */
const inflight = new Map<string, Promise<GensparkCreditEntry>>();

function cacheKey(providerName: string, keyId: string): string {
  return `${providerName}\0${keyId}`;
}

/**
 * Fresh cached credit entry for a key, or null when nothing fresh is known.
 * Stale entries are treated as unknown (and lazily evicted), never as authority.
 */
export function getGensparkCreditEntry(
  providerName: string,
  keyId: string,
  now = Date.now(),
): GensparkCreditEntry | null {
  const key = cacheKey(providerName, keyId);
  const entry = creditCache.get(key);
  if (!entry) return null;
  if (now - entry.checkedAt > GENSPARK_CREDIT_TTL_MS) {
    creditCache.delete(key);
    return null;
  }
  return entry;
}

/**
 * True only when a FRESH probe reported a zero/negative balance. Unknown, stale, or
 * never-probed keys return false — the API key stays eligible without a cookie.
 */
export function isGensparkCreditExhausted(
  providerName: string,
  keyId: string,
  now = Date.now(),
): boolean {
  return getGensparkCreditEntry(providerName, keyId, now)?.state === "exhausted";
}

/**
 * Extract a numeric balance from the payment endpoint's `data` payload. The success shape
 * is not publicly documented, so accept the plausible field names defensively; the
 * documented failure (`{"status":-5,"message":"not login","data":{}}`) yields null here
 * and therefore `unknown` — exactly the required degradation.
 */
function extractBalance(data: unknown): number | null {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;
  const record = data as Record<string, unknown>;
  for (const field of ["balance", "credit_balance", "credits", "credit", "total_credits", "remaining_credits"]) {
    const value = record[field];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

interface RefreshOptions {
  now?: number;
  /** Injectable for tests; defaults to global fetch. */
  fetchImpl?: typeof fetch;
}

/**
 * Probe the payment endpoint with this key's cookie and cache the result. NEVER throws:
 * every failure path (network, timeout, non-2xx, "not login", unparseable body) records
 * and returns an `unknown` entry, which also negatively caches so a broken cookie is not
 * re-probed until the TTL lapses. Concurrent calls for the same key share one probe.
 */
export async function refreshGensparkCredit(
  providerName: string,
  keyId: string,
  cookie: string,
  options: RefreshOptions = {},
): Promise<GensparkCreditEntry> {
  const key = cacheKey(providerName, keyId);
  const existing = inflight.get(key);
  if (existing) return existing;
  const probe = (async (): Promise<GensparkCreditEntry> => {
    let entry: GensparkCreditEntry = {
      balance: null,
      checkedAt: options.now ?? Date.now(),
      state: "unknown",
    };
    try {
      const fetchImpl = options.fetchImpl ?? fetch;
      const response = await fetchImpl(GENSPARK_CREDIT_BALANCE_URL, {
        method: "GET",
        headers: { cookie, accept: "application/json" },
        // Never follow a redirect while carrying the session cookie.
        redirect: "error",
        signal: AbortSignal.timeout(CREDIT_FETCH_TIMEOUT_MS),
      });
      if (response.ok) {
        const text = await response.text();
        if (text.length <= CREDIT_BODY_MAX_BYTES) {
          const body = JSON.parse(text) as { data?: unknown };
          const balance = extractBalance(body?.data);
          if (balance !== null) {
            entry = {
              balance,
              checkedAt: options.now ?? Date.now(),
              state: balance <= 0 ? "exhausted" : "healthy",
            };
          }
        }
      } else {
        // Consume defensively; the status alone means "unknown".
        try { void response.body?.cancel().catch(() => {}); } catch { /* closed */ }
      }
    } catch {
      // Cookie expired / network error / timeout / bad JSON: telemetry unknown, key eligible.
    }
    creditCache.set(key, entry);
    return entry;
  })();
  inflight.set(key, probe);
  try {
    return await probe;
  } finally {
    inflight.delete(key);
  }
}

/**
 * Fire-and-forget refresh for every cookie-carrying pool entry whose cache went stale.
 * Called from the failure classifier on Genspark failures — it must never block, throw,
 * or touch the inference path. No cookies in the pool means it does nothing at all.
 */
export function scheduleGensparkCreditRefresh(
  providerName: string,
  provider: Pick<OcxProviderConfig, "baseUrl" | "apiKeyPool">,
  now = Date.now(),
): void {
  if (!isGensparkProvider(providerName, provider)) return;
  for (const entry of provider.apiKeyPool ?? []) {
    if (!entry.cookie) continue;
    if (getGensparkCreditEntry(providerName, entry.id, now)) continue; // still fresh
    void refreshGensparkCredit(providerName, entry.id, entry.cookie).catch(() => {});
  }
}

/**
 * Refresh credit state for every cookie-carrying key of a provider (management/GUI path).
 * Keys without a cookie are reported as `unknown` without probing anything.
 */
export async function refreshGensparkProviderCredits(
  providerName: string,
  provider: Pick<OcxProviderConfig, "baseUrl" | "apiKeyPool">,
  options: RefreshOptions = {},
): Promise<Record<string, GensparkCreditEntry>> {
  const result: Record<string, GensparkCreditEntry> = {};
  const pool = provider.apiKeyPool ?? [];
  await Promise.all(pool.map(async entry => {
    if (entry.cookie && isGensparkProvider(providerName, provider)) {
      result[entry.id] = await refreshGensparkCredit(providerName, entry.id, entry.cookie, options);
    } else {
      result[entry.id] = { balance: null, checkedAt: options.now ?? Date.now(), state: "unknown" };
    }
  }));
  return result;
}

/** Clear cached credit state (all, per provider, or one key) — key management + tests. */
export function clearGensparkCreditCache(providerName?: string, keyId?: string): void {
  if (!providerName) {
    creditCache.clear();
    return;
  }
  if (keyId) {
    creditCache.delete(cacheKey(providerName, keyId));
    return;
  }
  const prefix = `${providerName}\0`;
  for (const key of creditCache.keys()) {
    if (key.startsWith(prefix)) creditCache.delete(key);
  }
}

/** Visible-for-testing: seed a cache entry without a network probe. */
export function setGensparkCreditEntryForTests(
  providerName: string,
  keyId: string,
  entry: GensparkCreditEntry,
): void {
  creditCache.set(cacheKey(providerName, keyId), entry);
}
