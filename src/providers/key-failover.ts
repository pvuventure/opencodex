/**
 * Multi-key failover for non-OpenAI providers.
 *
 * When a provider's upstream fails in a way attributable to the API key that sent it
 * (429 rate limit, 402 payment/quota, 401/403 bad credential, or a provider-specific
 * quota error — see `classifyKeyPoolFailure`), this module picks the next available key
 * from `apiKeyPool`, puts the failed key into cooldown (respecting Retry-After),
 * and returns a fresh provider config with the swapped key so the caller can replay the
 * SAME in-flight request. If all keys are in cooldown, returns null so the caller
 * surfaces the upstream error to the client.
 *
 * Modelled after src/codex/routing.ts cooldown logic but scoped to plain API-key pools.
 */
import { saveConfigPreservingClaudeCode } from "../config";
import type { OcxConfig, OcxProviderConfig, RateLimitRetryPolicy } from "../types";
import { resolveProviderTransport, type OcxProviderTransport } from "./xai-transport";
import { sweepExpiredOnWrite } from "../lib/state-store-sweeper";
import { readBoundedResponseBody } from "../lib/bounded-body";
import { isGensparkProvider, isGensparkQuotaErrorText } from "./genspark";
import { isGensparkCreditExhausted, scheduleGensparkCreditRefresh } from "./genspark-credit";

// ---- cooldown state (in-memory, same as codex/routing.ts) ----

interface KeyCooldown {
  cooldownUntil: number;
}

const DEFAULT_COOLDOWN_MS = 60_000;
const MAX_COOLDOWN_MS = 10 * 60_000; // cap at 10 min for api-key rotation
/**
 * A key that reported quota/payment exhaustion (402 or a provider quota error body) is
 * unlikely to recover within the 429 window — keep it out of rotation for longer so a
 * drained key does not re-enter every minute. In-memory only, so a restart re-probes.
 */
const QUOTA_COOLDOWN_MS = 30 * 60_000;
/** Invalid/unauthorized keys (401, key-attributable 403) also cool for the long window. */
const INVALID_AUTH_COOLDOWN_MS = 30 * 60_000;
/** Error bodies are peeked only for classification; a few KiB is plenty. */
const CLASSIFY_BODY_MAX_BYTES = 8_192;

/**
 * Default same-target 429 retry policy used when a provider opts in via a bare
 * `retryOn429: {}` (presence = opt-in with these defaults).
 */
const DEFAULT_RATE_LIMIT_RETRY = {
  enabled: true,
  attempts: 3,
  intervalMs: 5_000,
  maxIntervalMs: 60_000,
  respectRetryAfter: true,
} as const satisfies Required<RateLimitRetryPolicy>;

/** Map<`${providerName}\0${keyId}`, KeyCooldown> */
const keyCooldowns = new Map<string, KeyCooldown>();

function cooldownKey(providerName: string, keyId: string): string {
  return `${providerName}\0${keyId}`;
}

/**
 * Parse an upstream `Retry-After` header: numeric seconds (including `0`) or an HTTP-date.
 * Returns a bounded delay in ms (1..MAX_COOLDOWN_MS), or undefined when the value is
 * malformed. An HTTP-date already in the past yields an immediate (1 ms) retry.
 */
function parseRetryAfterMs(value: string | null | undefined, now = Date.now()): number | undefined {
  const text = value?.trim();
  if (!text) return undefined;
  if (/^\d+(?:\.\d+)?$/.test(text)) {
    const seconds = Number(text);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return Math.min(Math.max(Math.ceil(seconds * 1000), 1), MAX_COOLDOWN_MS);
    }
  }
  const timestamp = Date.parse(text);
  if (!Number.isFinite(timestamp)) return undefined;
  const delay = timestamp - now;
  // A valid HTTP-date whose retry time has already passed is an immediate retry, exactly like
  // numeric `Retry-After: 0` — never a malformed-header fallback to the fixed interval.
  return Math.min(Math.max(delay, 1), MAX_COOLDOWN_MS);
}

/**
 * True while the given key is inside its 429 cooldown window (lazily evicting the entry once the
 * window expires). Used to skip keys that the upstream just rate-limited during failover.
 */
function isKeyInCooldown(providerName: string, keyId: string, now = Date.now()): boolean {
  const entry = keyCooldowns.get(cooldownKey(providerName, keyId));
  if (!entry) return false;
  if (entry.cooldownUntil <= now) {
    keyCooldowns.delete(cooldownKey(providerName, keyId));
    return false;
  }
  return true;
}

// ---- failure classification ----

/**
 * How an upstream failure relates to the API key that sent it. Only these kinds
 * trigger key rotation; everything else (5xx, 4xx request errors, success) is
 * handled by the ordinary transport/recovery paths.
 */
export type KeyPoolFailureKind = "rate-limit" | "quota-exhausted" | "invalid-auth";

export interface KeyPoolFailure {
  kind: KeyPoolFailureKind;
  /** Upstream HTTP status that produced this classification. */
  status: number;
  /** Raw Retry-After header (rate-limit cooldowns honor it). */
  retryAfter: string | null;
}

/**
 * 403 bodies are only attributed to the KEY when they explicitly name the credential.
 * A bare "permission denied" / "model not allowed" 403 may be provider-wide or
 * model-level, and rotating keys on it would burn the whole pool for nothing.
 */
const KEY_ATTRIBUTABLE_AUTH_PATTERNS: readonly RegExp[] = [
  /api[\s_-]?key/i,
  /invalid[\s_-]+(?:key|token|credential)/i,
  /(?:expired|revoked|disabled)[\s_-]+(?:key|token|credential)/i,
  /credential/i,
  /authentication[\s_-]+(?:failed|error|invalid)/i,
];

/**
 * Classify an upstream ERROR response as a key-attributable failure, or null when the
 * failure must not rotate keys. Reads at most a small bounded prefix of the body, and
 * only when the status alone is ambiguous (403, or a Genspark 4xx that may carry a
 * quota message under a non-429 status). Never called on 2xx by contract — and returns
 * null for them anyway, so a 200 with `usageStatus: "unreported"` can never rotate.
 */
export async function classifyKeyPoolFailure(
  response: Response,
  providerName: string,
  provider: Pick<OcxProviderConfig, "baseUrl" | "apiKeyPool">,
  options: { signal?: AbortSignal } = {},
): Promise<KeyPoolFailure | null> {
  const status = response.status;
  const retryAfter = response.headers.get("retry-after");
  if (status < 400 || status >= 500) return null;
  // A key failure is the moment credit telemetry pays off: kick a fire-and-forget refresh
  // for cookie-carrying Genspark pool entries (cached, TTL-gated, never blocks this path).
  scheduleGensparkCreditRefresh(providerName, provider);
  if (status === 429) return { kind: "rate-limit", status, retryAfter };
  if (status === 402) return { kind: "quota-exhausted", status, retryAfter };
  if (status === 401) return { kind: "invalid-auth", status, retryAfter };

  // Remaining 4xx: sniff the error body only where a provider is known to hide
  // key-scoped failures behind other statuses (Genspark quota), or where the status
  // is ambiguous between key-level and provider-level (403).
  const genspark = isGensparkProvider(providerName, provider);
  if (status !== 403 && !genspark) return null;
  let text = "";
  try {
    // Clone so the original body stays readable for downstream error reporting when we
    // decide NOT to rotate. Bounded read: classification needs a prefix, not the payload.
    const body = await readBoundedResponseBody(response.clone(), {
      signal: options.signal,
      maxBytes: CLASSIFY_BODY_MAX_BYTES,
    });
    if (body.displaySafe) text = body.text;
  } catch {
    return null; // unreadable body: fail closed (no rotation on ambiguous evidence)
  }
  if (genspark && isGensparkQuotaErrorText(text)) {
    return { kind: "quota-exhausted", status, retryAfter };
  }
  if (status === 403 && KEY_ATTRIBUTABLE_AUTH_PATTERNS.some(p => p.test(text))) {
    return { kind: "invalid-auth", status, retryAfter };
  }
  return null;
}

/**
 * Attempt-log recovery kind for a classified key failure. String literals match the
 * `AttemptRecoveryKind` union in src/usage/log.ts (not imported: usage/log sits above
 * providers in the dependency graph).
 */
export function keyFailureRecoveryKind(failure: KeyPoolFailure): "key-429" | "key-quota" | "key-auth" {
  switch (failure.kind) {
    case "rate-limit": return "key-429";
    case "quota-exhausted": return "key-quota";
    case "invalid-auth": return "key-auth";
  }
}

/** Cooldown window for a classified key failure. Rate limits honor Retry-After. */
function cooldownMsForFailure(failure: KeyPoolFailure, now: number): number {
  switch (failure.kind) {
    case "rate-limit":
      return parseRetryAfterMs(failure.retryAfter, now) ?? DEFAULT_COOLDOWN_MS;
    case "quota-exhausted":
      return QUOTA_COOLDOWN_MS;
    case "invalid-auth":
      return INVALID_AUTH_COOLDOWN_MS;
  }
}

// ---- public API ----

/**
 * Check whether a provider has multiple keys available for failover.
 * Returns true only for key-auth providers with 2+ pool entries.
 */
export function hasKeyPoolFailover(provider: OcxProviderConfig): boolean {
  if (provider.authMode === "oauth" || provider.authMode === "forward") return false;
  return (provider.apiKeyPool?.length ?? 0) >= 2;
}

// ---- proactive pool strategy (Phase 3: apiKeyPoolStrategy) ----

/**
 * In-memory round-robin cursors, keyed by provider name. Deliberately NOT persisted:
 * the spec forbids a disk write per request, and a restart simply restarts the cycle.
 */
const roundRobinCursors = new Map<string, number>();

/**
 * Pick the key a NEW request should use, per the provider's `apiKeyPoolStrategy`.
 * Reactive failover (rotateKeyOnFailure) still applies on top of whatever this picks.
 *
 * - "failover" (default / unknown values): keep the active key — a no-op, so existing
 *   configs behave exactly as before this knob existed.
 * - "round-robin": advance an in-memory cursor across keys that are neither cooling
 *   nor telemetry-known-drained; fall back to the active key when none qualify.
 * - "quota-aware": keep the active key unless its Genspark credit telemetry reports a
 *   fresh zero balance, in which case swap to the first non-drained, non-cooling key.
 *
 * Pure selection: mutates only the in-memory cursor, never the config on disk (the
 * active `apiKey` mirror is only persisted by reactive rotation and manual management).
 * Returns the provider unchanged when no better key exists — the request then proceeds
 * on the active key and reactive failover remains the authority.
 */
export function applyKeyPoolStrategy(
  providerName: string,
  provider: OcxProviderConfig,
  now = Date.now(),
): OcxProviderConfig {
  const strategy = provider.apiKeyPoolStrategy;
  if (strategy !== "round-robin" && strategy !== "quota-aware") return provider;
  if (!hasKeyPoolFailover(provider)) return provider;
  const pool = provider.apiKeyPool!;

  const usable = (id: string): boolean =>
    !isKeyInCooldown(providerName, id, now) && !isGensparkCreditExhausted(providerName, id, now);

  if (strategy === "round-robin") {
    const start = roundRobinCursors.get(providerName) ?? 0;
    for (let i = 0; i < pool.length; i++) {
      const index = (start + i) % pool.length;
      const candidate = pool[index]!;
      if (!usable(candidate.id)) continue;
      roundRobinCursors.set(providerName, index + 1);
      return candidate.key === provider.apiKey ? provider : { ...provider, apiKey: candidate.key };
    }
    return provider; // everything cooling/drained: stay reactive
  }

  // quota-aware: only move OFF a key that telemetry says is drained.
  const activeEntry = pool.find(e => e.key === provider.apiKey);
  if (!activeEntry || usable(activeEntry.id)) return provider;
  const activeIndex = pool.indexOf(activeEntry);
  for (let i = 1; i < pool.length; i++) {
    const candidate = pool[(activeIndex + i) % pool.length]!;
    if (usable(candidate.id)) {
      console.warn(
        `[key-failover] ${providerName}: quota-aware strategy skipping drained/cooling key ${activeEntry.id}; using key ${candidate.id}`,
      );
      return { ...provider, apiKey: candidate.key };
    }
  }
  return provider;
}

/** Test-only: reset in-memory round-robin cursors. */
export function clearKeyPoolStrategyState(providerName?: string): void {
  if (!providerName) {
    roundRobinCursors.clear();
    return;
  }
  roundRobinCursors.delete(providerName);
}

/**
 * Normalize a provider's `retryOn429` policy, or return null when the knob is absent,
 * explicitly disabled, or the provider is not key-auth (OAuth/forward credentials must not be
 * replayed on the same token, forward passthrough never reaches the recovery loop anyway, and
 * local runtimes have no remote key to preserve). The returned policy is fully defaulted so
 * callers never re-check fields.
 */
export function rateLimitRetryPolicyFor(
  provider: Pick<OcxProviderConfig, "retryOn429" | "authMode">,
): Required<RateLimitRetryPolicy> | null {
  const policy = provider.retryOn429;
  if (!policy || policy.enabled === false) return null;
  // Fail closed: only explicit key auth or the documented omitted-default (undefined == key for
  // custom API-key providers) may use same-key replays. OAuth/forward are never replayed on the
  // same token, local runtimes have no remote key to preserve, and unknown/custom values are
  // rejected rather than guessed at.
  if (provider.authMode !== undefined && provider.authMode !== "key") return null;
  return {
    enabled: policy.enabled ?? DEFAULT_RATE_LIMIT_RETRY.enabled,
    attempts: policy.attempts ?? DEFAULT_RATE_LIMIT_RETRY.attempts,
    intervalMs: policy.intervalMs ?? DEFAULT_RATE_LIMIT_RETRY.intervalMs,
    maxIntervalMs: policy.maxIntervalMs ?? DEFAULT_RATE_LIMIT_RETRY.maxIntervalMs,
    respectRetryAfter: policy.respectRetryAfter ?? DEFAULT_RATE_LIMIT_RETRY.respectRetryAfter,
  };
}

/**
 * Wait before the next same-target replay: upstream Retry-After (seconds or HTTP-date) when
 * `respectRetryAfter` is on and the header parses, capped at `maxIntervalMs`; otherwise the
 * fixed `intervalMs`, also capped at `maxIntervalMs` (a single wait never exceeds the cap).
 * Malformed headers fall back to the fixed interval.
 */
export function rateLimitRetryDelayMs(
  policy: Required<RateLimitRetryPolicy>,
  retryAfterHeader: string | null | undefined,
  now = Date.now(),
): number {
  const raw = retryAfterHeader?.trim();
  if (policy.respectRetryAfter && raw) {
    const parsed = parseRetryAfterMs(raw, now);
    if (parsed !== undefined) return Math.min(parsed, policy.maxIntervalMs);
  }
  return Math.min(policy.intervalMs, policy.maxIntervalMs);
}

/**
 * Record a key-attributable failure for the current key and attempt to switch to the
 * next available one.
 *
 * @param failure   Classified failure (kind decides the cooldown window; `rate-limit`
 *                  honors Retry-After).
 * @param attemptedKeyIds Request-scoped set of pool-entry ids this failover chain has
 *                  already tried. The failed and returned keys are recorded into it, and
 *                  members are never selected again for the same chain — even if their
 *                  global cooldown has lapsed — so one request's chain is strictly
 *                  bounded by the pool size.
 *
 * @returns A new OcxProviderConfig with the swapped key (and mutated config on disk),
 *          or `null` when no alternative key is available (all in cooldown or pool < 2).
 *
 * The returned object is a snapshot of the PERSISTED config — it carries none of the
 * registry backfills `routedProviderConfig` merges in at request time. Request paths must
 * not assign it to an active route wholesale; use `rotateProviderTransportOnFailure`,
 * which takes only the swapped key and keeps the routed provider intact.
 */
export function rotateKeyOnFailure(
  config: OcxConfig,
  providerName: string,
  failure: KeyPoolFailure,
  now = Date.now(),
  attemptedKey?: string,
  attemptedKeyIds?: Set<string>,
): OcxProviderConfig | null {
  const provider = config.providers[providerName];
  if (!provider) return null;
  if (provider.authMode === "oauth" || provider.authMode === "forward") return null;

  const pool = provider.apiKeyPool;
  if (!pool || pool.length < 2) return null;

  // Cool the key that ACTUALLY failed. Under concurrent failures another request may already
  // have rotated provider.apiKey — cooling the live key would punish an innocent replacement and
  // can exhaust a 2-key pool from a single bad key. CAS semantics: callers pass the key they used.
  const failedKey = attemptedKey ?? provider.apiKey;
  const currentEntry = pool.find(e => e.key === failedKey);
  if (currentEntry) {
    keyCooldowns.set(cooldownKey(providerName, currentEntry.id), {
      cooldownUntil: now + cooldownMsForFailure(failure, now),
    });
    sweepExpiredOnWrite(now);
    attemptedKeyIds?.add(currentEntry.id);
  }

  const eligible = (id: string): boolean =>
    !isKeyInCooldown(providerName, id, now) && !attemptedKeyIds?.has(id);

  // Lost the race: someone already rotated away from the failed key. If the live key is healthy
  // (and this chain has not already burned it), retry with it as-is instead of rotating again.
  if (attemptedKey !== undefined && provider.apiKey !== attemptedKey) {
    const liveEntry = pool.find(e => e.key === provider.apiKey);
    if (liveEntry && eligible(liveEntry.id)) {
      attemptedKeyIds?.add(liveEntry.id);
      return { ...provider };
    }
  }

  const select = (candidate: NonNullable<OcxProviderConfig["apiKeyPool"]>[number]): OcxProviderConfig => {
    // Swap active key
    provider.apiKey = candidate.key;
    saveConfigPreservingClaudeCode(config);
    console.warn(
      // Log ids only — labels are user-supplied free text and could carry secret material.
      `[key-failover] ${providerName}: ${failure.kind} (${failure.status}) on key ${currentEntry?.id ?? "?"}; rotating to key ${candidate.id}`,
    );
    attemptedKeyIds?.add(candidate.id);
    return { ...provider };
  };

  // Pick the next key that is NOT in cooldown and not yet tried by this chain. Two passes:
  // pass 1 also skips keys whose Genspark credit telemetry (optional per-key cookie) reports
  // a fresh zero balance — no point burning an attempt on a key that will 402. Pass 2 drops
  // that soft preference so telemetry (stale, wrong, or cookie-less) can never make inference
  // WORSE than pure reactive failover: a "known-drained" key still gets tried when it is the
  // only remaining option.
  const currentIndex = currentEntry ? pool.indexOf(currentEntry) : -1;
  for (let i = 1; i < pool.length; i++) {
    const candidate = pool[(currentIndex + i) % pool.length]!;
    if (eligible(candidate.id) && !isGensparkCreditExhausted(providerName, candidate.id, now)) {
      return select(candidate);
    }
  }
  for (let i = 1; i < pool.length; i++) {
    const candidate = pool[(currentIndex + i) % pool.length]!;
    if (eligible(candidate.id)) {
      return select(candidate);
    }
  }

  // All keys in cooldown / already attempted
  console.warn(`[key-failover] ${providerName}: all ${pool.length} keys unavailable (cooldown or already attempted); returning upstream error to client`);
  return null;
}

/**
 * Back-compat 429 entry point: classify as a plain rate limit and rotate.
 * Prefer `rotateKeyOnFailure` for new call sites.
 */
export function rotateKeyOn429(
  config: OcxConfig,
  providerName: string,
  retryAfterHeader: string | null | undefined,
  now = Date.now(),
  attemptedKey?: string,
): OcxProviderConfig | null {
  return rotateKeyOnFailure(
    config,
    providerName,
    { kind: "rate-limit", status: 429, retryAfter: retryAfterHeader ?? null },
    now,
    attemptedKey,
  );
}

export function sweepExpiredApiKeyCooldowns(now = Date.now()): number {
  let removed = 0;
  for (const [key, cooldown] of keyCooldowns) {
    if (cooldown.cooldownUntil > now) continue;
    keyCooldowns.delete(key);
    removed += 1;
  }
  return removed;
}

interface RotateProviderTransportOptions {
  retryAfter?: string | null;
  now?: number;
  attemptedKey?: string;
  promptCacheKey?: string;
  /** Request-scoped chain tracking; see rotateKeyOnFailure. */
  attemptedKeyIds?: Set<string>;
}

interface RotateProviderTransportOnFailureOptions extends Omit<RotateProviderTransportOptions, "retryAfter"> {
  failure: KeyPoolFailure;
}

/**
 * Rotate a failed key and re-apply provider-specific transport metadata to the replacement.
 *
 * `routedProvider` is the request's active provider (the `routedProviderConfig` output the
 * route was built with). The result inherits it and swaps ONLY the API key: the persisted
 * config that `rotateKeyOnFailure` snapshots predates registry backfill, so building the
 * retry provider from that snapshot would silently drop every field the registry merged in
 * at routing time (scalar flags like `promptCacheKey`/`parallelToolCalls`, merged model
 * metadata such as `noTemperatureModels`, a pinned baseUrl). Mirrors the OAuth-401 replay
 * path in src/server/responses/core.ts, which spreads `route.provider` for the same reason.
 */
export function rotateProviderTransportOnFailure(
  config: OcxConfig,
  providerName: string,
  routedProvider: OcxProviderTransport,
  options: RotateProviderTransportOnFailureOptions,
): OcxProviderTransport | null {
  const rotated = rotateKeyOnFailure(
    config,
    providerName,
    options.failure,
    options.now,
    options.attemptedKey,
    options.attemptedKeyIds,
  );
  return rotated
    ? resolveProviderTransport(
        providerName,
        { ...routedProvider, apiKey: rotated.apiKey },
        options.promptCacheKey,
      )
    : null;
}

/**
 * Back-compat 429 transport rotation. Prefer `rotateProviderTransportOnFailure` for new
 * call sites; existing 429-only loops keep working unchanged through this wrapper.
 */
export function rotateProviderTransportOn429(
  config: OcxConfig,
  providerName: string,
  routedProvider: OcxProviderTransport,
  options: RotateProviderTransportOptions = {},
): OcxProviderTransport | null {
  return rotateProviderTransportOnFailure(config, providerName, routedProvider, {
    failure: { kind: "rate-limit", status: 429, retryAfter: options.retryAfter ?? null },
    ...(options.now !== undefined ? { now: options.now } : {}),
    ...(options.attemptedKey !== undefined ? { attemptedKey: options.attemptedKey } : {}),
    ...(options.promptCacheKey !== undefined ? { promptCacheKey: options.promptCacheKey } : {}),
    ...(options.attemptedKeyIds !== undefined ? { attemptedKeyIds: options.attemptedKeyIds } : {}),
  });
}

/** Clear cooldown state for a provider (e.g. after manual key management). */
export function clearKeyCooldowns(providerName?: string): void {
  if (!providerName) {
    keyCooldowns.clear();
    return;
  }
  const prefix = `${providerName}\0`;
  for (const key of keyCooldowns.keys()) {
    if (key.startsWith(prefix)) keyCooldowns.delete(key);
  }
}

/** Visible-for-testing: get the cooldown-until timestamp for a key. */
export function getKeyCooldownUntil(providerName: string, keyId: string, now = Date.now()): number | null {
  const entry = keyCooldowns.get(cooldownKey(providerName, keyId));
  if (!entry) return null;
  return entry.cooldownUntil > now ? entry.cooldownUntil : null;
}
