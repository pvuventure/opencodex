/**
 * Genspark-specific upstream failure classification.
 *
 * Genspark's LLM proxy (https://www.genspark.ai/api/llm_proxy/v1) reports credit/quota
 * exhaustion with statuses other than 429 (e.g. 400/403 with a human-readable message),
 * so the generic status-based key-failover classifier cannot see it. This module is the
 * ONLY place that knows Genspark's wording; the generic failover machinery consumes just
 * the classification result (src/providers/key-failure.ts).
 *
 * Safety: only structurally failed responses (non-2xx) may be sniffed by the caller.
 * Successful assistant output is never substring-matched against these patterns.
 */
import type { OcxProviderConfig } from "../types";

/** True when a provider routes to Genspark (by conventional name or destination host). */
export function isGensparkProvider(
  providerName: string,
  provider: Pick<OcxProviderConfig, "baseUrl">,
): boolean {
  if (providerName.trim().toLowerCase() === "genspark") return true;
  if (typeof provider.baseUrl !== "string" || !provider.baseUrl) return false;
  try {
    const host = new URL(provider.baseUrl).hostname.toLowerCase();
    return host === "genspark.ai" || host.endsWith(".genspark.ai");
  } catch {
    return false;
  }
}

/**
 * Quota/credit-exhaustion phrases Genspark uses in upstream error bodies. Deliberately
 * narrow: generic rate-limit wording ("too many requests") is NOT here — that stays a
 * status-429 concern with its ordinary shorter cooldown.
 */
const GENSPARK_QUOTA_PATTERNS: readonly RegExp[] = [
  /insufficient[\s_-]+credits?/i,
  /insufficient[\s_-]+quota/i,
  /quota[\s_-]+(?:exceeded|exhausted)/i,
  /credits?[\s_-]+(?:balance[\s_-]+)?exhausted/i,
  /no[\s_-]+credits/i,
  /out[\s_-]+of[\s_-]+credits?/i,
  /credit[\s_-]+balance[\s_-]+exhausted/i,
];

/**
 * True when an upstream ERROR body clearly reports credit/quota exhaustion.
 * Callers must only pass text from structurally failed responses (non-2xx) —
 * never from successful completions, and never merely because usage metrics
 * were unreported on a 200.
 */
export function isGensparkQuotaErrorText(text: string): boolean {
  if (!text) return false;
  return GENSPARK_QUOTA_PATTERNS.some(pattern => pattern.test(text));
}
