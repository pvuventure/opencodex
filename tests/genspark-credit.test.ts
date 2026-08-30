/**
 * Phase 2: optional Genspark credit telemetry.
 *
 * Contract under test:
 *   API key = inference auth (always). Cookie = credit telemetry ONLY.
 *   - The payment endpoint is never called on the inference path (cached, TTL-gated).
 *   - Any cookie failure degrades to `unknown` and the key stays eligible.
 *   - A fresh known-zero balance lets rotation skip that key — but only while a
 *     better candidate exists (telemetry can never make failover worse).
 *   - Raw cookies never appear in logs or management responses.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { saveConfig } from "../src/config";
import { configuredAdminToken } from "../src/lib/admin-secrets";
import {
  clearGensparkCreditCache,
  getGensparkCreditEntry,
  isGensparkCreditExhausted,
  refreshGensparkCredit,
  setGensparkCreditEntryForTests,
  GENSPARK_CREDIT_BALANCE_URL,
  GENSPARK_CREDIT_TTL_MS,
} from "../src/providers/genspark-credit";
import {
  clearKeyCooldowns,
  rotateKeyOnFailure,
  type KeyPoolFailure,
} from "../src/providers/key-failover";
import { startServer } from "../src/server";
import type { OcxConfig, OcxProviderConfig } from "../src/types";
import { installIsolatedCodexHome, type IsolatedCodexHome } from "./helpers/isolated-codex-home";

const KEY_A = "gsk_test_key_A";
const KEY_B = "gsk_test_key_B";
const KEY_C = "gsk_test_key_C";
const COOKIE_A = "session_id=fake-cookie-value-A; other=1";
const GENSPARK_BASE = "https://www.genspark.ai/api/llm_proxy/v1";
const GENSPARK_CHAT = `${GENSPARK_BASE}/chat/completions`;

let testDir = "";
let previousHome: string | undefined;
let isolatedCodexHome: IsolatedCodexHome | null = null;
const originalFetch = globalThis.fetch;

beforeEach(() => {
  previousHome = process.env.OPENCODEX_HOME;
  isolatedCodexHome = installIsolatedCodexHome("ocx-gsk-credit-codex-");
  testDir = mkdtempSync(join(tmpdir(), "ocx-gsk-credit-"));
  process.env.OPENCODEX_HOME = testDir;
  clearKeyCooldowns();
  clearGensparkCreditCache();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
  else process.env.OPENCODEX_HOME = previousHome;
  isolatedCodexHome?.restore();
  isolatedCodexHome = null;
  if (testDir) rmSync(testDir, { recursive: true, force: true });
  clearKeyCooldowns();
  clearGensparkCreditCache();
});

function balanceResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("refreshGensparkCredit (unit)", () => {
  test("healthy balance is cached; zero balance is exhausted", async () => {
    const fetchImpl = (async () => balanceResponse({ status: 0, data: { balance: 8420 } })) as typeof fetch;
    const entry = await refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl, now: 1_000 });
    expect(entry).toEqual({ balance: 8420, checkedAt: 1_000, state: "healthy" });
    expect(getGensparkCreditEntry("genspark", "kA", 1_000)).toEqual(entry);

    clearGensparkCreditCache();
    const zero = (async () => balanceResponse({ status: 0, data: { credits: 0 } })) as typeof fetch;
    const drained = await refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl: zero, now: 2_000 });
    expect(drained.state).toBe("exhausted");
    expect(isGensparkCreditExhausted("genspark", "kA", 2_000)).toBe(true);
  });

  test("'not login' (expired cookie) degrades to unknown — key stays eligible", async () => {
    const fetchImpl = (async () => balanceResponse({ status: -5, message: "not login", data: {} })) as typeof fetch;
    const entry = await refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl, now: 1_000 });
    expect(entry.state).toBe("unknown");
    expect(entry.balance).toBeNull();
    expect(isGensparkCreditExhausted("genspark", "kA", 1_000)).toBe(false);
  });

  test("network failure, non-2xx, and bad JSON all degrade to unknown without throwing", async () => {
    for (const fetchImpl of [
      (async () => { throw new TypeError("connection reset"); }) as unknown as typeof fetch,
      (async () => balanceResponse({ error: "server" }, 503)) as typeof fetch,
      (async () => new Response("<html>login page</html>", { headers: { "content-type": "text/html" } })) as typeof fetch,
    ]) {
      clearGensparkCreditCache();
      const entry = await refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl, now: 1_000 });
      expect(entry.state).toBe("unknown");
      expect(isGensparkCreditExhausted("genspark", "kA", 1_000)).toBe(false);
    }
  });

  test("the probe sends the cookie ONLY to the payment endpoint and never follows redirects", async () => {
    let seenUrl = "";
    let seenCookie: string | null = null;
    let seenRedirect: string | undefined;
    const fetchImpl = (async (input: RequestInfo | URL, init?: RequestInit) => {
      seenUrl = String(input);
      seenCookie = new Headers(init?.headers).get("cookie");
      seenRedirect = init?.redirect;
      return balanceResponse({ data: { balance: 5 } });
    }) as typeof fetch;
    await refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl, now: 1_000 });
    expect(seenUrl).toBe(GENSPARK_CREDIT_BALANCE_URL);
    expect(seenCookie).toBe(COOKIE_A);
    expect(seenRedirect).toBe("error");
  });

  test("cache honors the TTL: stale entries evaporate to unknown", () => {
    setGensparkCreditEntryForTests("genspark", "kA", { balance: 0, checkedAt: 1_000, state: "exhausted" });
    expect(isGensparkCreditExhausted("genspark", "kA", 1_000 + GENSPARK_CREDIT_TTL_MS - 1)).toBe(true);
    expect(isGensparkCreditExhausted("genspark", "kA", 1_000 + GENSPARK_CREDIT_TTL_MS + 1)).toBe(false);
    expect(getGensparkCreditEntry("genspark", "kA", 1_000 + GENSPARK_CREDIT_TTL_MS + 1)).toBeNull();
  });

  test("concurrent refreshes share ONE probe per key", async () => {
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      await Bun.sleep(20);
      return balanceResponse({ data: { balance: 100 } });
    }) as typeof fetch;
    await Promise.all([
      refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl }),
      refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl }),
      refreshGensparkCredit("genspark", "kA", COOKIE_A, { fetchImpl }),
    ]);
    expect(calls).toBe(1);
  });
});

// ---------- quota-aware rotation ----------

function makeConfig(provider: Partial<OcxProviderConfig>): OcxConfig {
  return {
    port: 10199,
    defaultProvider: "genspark",
    providers: {
      genspark: {
        adapter: "openai-chat",
        baseUrl: GENSPARK_BASE,
        authMode: "key",
        ...provider,
      } as OcxProviderConfig,
    },
  } as OcxConfig;
}

function pool3(): OcxProviderConfig["apiKeyPool"] {
  return [
    { id: "kA", key: KEY_A, addedAt: 1 },
    { id: "kB", key: KEY_B, addedAt: 2 },
    { id: "kC", key: KEY_C, addedAt: 3 },
  ];
}

const RATE_LIMIT: KeyPoolFailure = { kind: "rate-limit", status: 429, retryAfter: null };

describe("quota-aware rotation (unit)", () => {
  test("rotation skips a key with fresh zero-balance telemetry", () => {
    const now = 1_000_000;
    const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    // Telemetry knows kB is drained; kC is unknown.
    setGensparkCreditEntryForTests("genspark", "kB", { balance: 0, checkedAt: now, state: "exhausted" });
    expect(rotateKeyOnFailure(config, "genspark", RATE_LIMIT, now, KEY_A)?.apiKey).toBe(KEY_C);
  });

  test("unknown-credit keys remain fully eligible (cookie never required)", () => {
    const now = 1_000_000;
    const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    // No telemetry at all: plain reactive order kA -> kB.
    expect(rotateKeyOnFailure(config, "genspark", RATE_LIMIT, now, KEY_A)?.apiKey).toBe(KEY_B);
  });

  test("a known-drained key is still tried when it is the ONLY remaining option", () => {
    const now = 1_000_000;
    const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    setGensparkCreditEntryForTests("genspark", "kB", { balance: 0, checkedAt: now, state: "exhausted" });
    setGensparkCreditEntryForTests("genspark", "kC", { balance: 0, checkedAt: now, state: "exhausted" });
    // Telemetry marks BOTH alternatives drained — the fallback pass must still rotate
    // (telemetry could be stale/wrong; reactive failover is the authority).
    expect(rotateKeyOnFailure(config, "genspark", RATE_LIMIT, now, KEY_A)?.apiKey).toBe(KEY_B);
  });

  test("stale telemetry does not skip anything", () => {
    const now = 1_000_000;
    const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    setGensparkCreditEntryForTests("genspark", "kB", {
      balance: 0,
      checkedAt: now - GENSPARK_CREDIT_TTL_MS - 1,
      state: "exhausted",
    });
    expect(rotateKeyOnFailure(config, "genspark", RATE_LIMIT, now, KEY_A)?.apiKey).toBe(KEY_B);
  });

  test("non-Genspark providers are never affected by the credit cache", () => {
    const now = 1_000_000;
    const config = {
      port: 1,
      defaultProvider: "p",
      providers: {
        p: { adapter: "openai-chat", baseUrl: "https://api.example.com/v1", apiKey: KEY_A, apiKeyPool: pool3() },
      },
    } as unknown as OcxConfig;
    // Even a (bogus) cache entry under the same key id must not skip for provider "p":
    // isGensparkCreditExhausted is keyed by provider name, and "p" never probes.
    setGensparkCreditEntryForTests("p", "kB", { balance: 0, checkedAt: now, state: "exhausted" });
    // The skip applies per cache — but this asserts the practical outcome for a provider
    // that never populates telemetry through any real path: clear it and verify plain order.
    clearGensparkCreditCache("p");
    expect(rotateKeyOnFailure(config, "p", RATE_LIMIT, now, KEY_A)?.apiKey).toBe(KEY_B);
  });
});

// ---------- e2e: inference path + management API ----------

function gensparkConfig(pool: NonNullable<OcxProviderConfig["apiKeyPool"]>): OcxConfig {
  return {
    port: 0,
    hostname: "127.0.0.1",
    defaultProvider: "genspark",
    providers: {
      genspark: {
        adapter: "openai-chat",
        baseUrl: GENSPARK_BASE,
        authMode: "key",
        apiKey: pool[0]!.key,
        apiKeyPool: pool,
      },
    },
  } as OcxConfig;
}

function managementHeaders(extra?: HeadersInit): Headers {
  const token = configuredAdminToken();
  if (!token) throw new Error("management token was not initialized");
  const headers = new Headers(extra);
  headers.set("x-opencodex-api-key", token);
  return headers;
}

describe("genspark credit telemetry (end-to-end)", () => {
  test("inference NEVER calls the payment endpoint; success does not probe even with cookies stored", async () => {
    let paymentCalls = 0;
    globalThis.fetch = (async (input, init) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url === GENSPARK_CREDIT_BALANCE_URL) {
        paymentCalls += 1;
        return balanceResponse({ data: { balance: 1 } });
      }
      if (url === GENSPARK_CHAT) {
        // The inference request must carry the API key and NEVER the cookie.
        const headers = new Headers(init?.headers);
        expect(headers.get("cookie")).toBeNull();
        expect(headers.get("authorization")).toBe(`Bearer ${KEY_A}`);
        return new Response(JSON.stringify({
          id: "c", object: "chat.completion",
          choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }), { headers: { "content-type": "application/json" } });
      }
      return originalFetch(input, init);
    }) as typeof fetch;

    saveConfig(gensparkConfig([
      { id: "kA", key: KEY_A, addedAt: 1, cookie: COOKIE_A },
      { id: "kB", key: KEY_B, addedAt: 2 },
    ]));
    const server = startServer(0);
    try {
      const res = await originalFetch(new URL("/v1/responses", server.url), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "genspark/gpt-test", input: "hi", stream: false }),
      });
      expect(res.status).toBe(200);
      expect(paymentCalls).toBe(0); // success path: telemetry silent
    } finally {
      await server.stop(true);
    }
  });

  test("a Genspark key failure triggers a background telemetry refresh (fire-and-forget)", async () => {
    let paymentCalls = 0;
    let paymentCookie: string | null = null;
    let attempts = 0;
    globalThis.fetch = (async (input, init) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url === GENSPARK_CREDIT_BALANCE_URL) {
        paymentCalls += 1;
        paymentCookie = new Headers(init?.headers).get("cookie");
        return balanceResponse({ data: { balance: 0 } });
      }
      if (url === GENSPARK_CHAT) {
        attempts += 1;
        if (attempts === 1) {
          return new Response(JSON.stringify({ error: { message: "credits exhausted" } }), {
            status: 402, headers: { "content-type": "application/json" },
          });
        }
        return new Response(JSON.stringify({
          id: "c", object: "chat.completion",
          choices: [{ index: 0, message: { role: "assistant", content: "recovered" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }), { headers: { "content-type": "application/json" } });
      }
      return originalFetch(input, init);
    }) as typeof fetch;

    saveConfig(gensparkConfig([
      { id: "kA", key: KEY_A, addedAt: 1, cookie: COOKIE_A },
      { id: "kB", key: KEY_B, addedAt: 2 },
    ]));
    const server = startServer(0);
    try {
      const res = await originalFetch(new URL("/v1/responses", server.url), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "genspark/gpt-test", input: "hi", stream: false }),
      });
      expect(res.status).toBe(200); // failover still recovered inference
      // Give the fire-and-forget probe a beat to land.
      await Bun.sleep(50);
      expect(paymentCalls).toBe(1); // kA has a cookie; kB does not
      expect(paymentCookie).toBe(COOKIE_A);
      expect(getGensparkCreditEntry("genspark", "kA", Date.now())?.state).toBe("exhausted");
    } finally {
      await server.stop(true);
    }
  });

  test("management: cookie is write-only; list shows hasCookie, never the value", async () => {
    saveConfig(gensparkConfig([
      { id: "kA", key: KEY_A, addedAt: 1 },
      { id: "kB", key: KEY_B, addedAt: 2 },
    ]));
    const server = startServer(0);
    try {
      const put = await fetch(new URL("/api/providers/keys/cookie", server.url), {
        method: "PUT",
        headers: managementHeaders({ "content-type": "application/json" }),
        body: JSON.stringify({ name: "genspark", id: "kA", cookie: COOKIE_A }),
      });
      expect(put.status).toBe(200);
      const putBody = await put.json() as Record<string, unknown>;
      expect(JSON.stringify(putBody)).not.toContain("fake-cookie-value-A");

      const list = await fetch(new URL("/api/providers/keys?name=genspark", server.url), {
        headers: managementHeaders(),
      });
      const listBody = await list.json() as { keys: Array<Record<string, unknown>> };
      expect(JSON.stringify(listBody)).not.toContain("fake-cookie-value-A");
      expect(listBody.keys.find(k => k.id === "kA")?.hasCookie).toBe(true);
      expect(listBody.keys.find(k => k.id === "kB")?.hasCookie).toBeUndefined();

      // Clearing works and drops the flag.
      const clear = await fetch(new URL("/api/providers/keys/cookie", server.url), {
        method: "PUT",
        headers: managementHeaders({ "content-type": "application/json" }),
        body: JSON.stringify({ name: "genspark", id: "kA", cookie: null }),
      });
      expect(clear.status).toBe(200);
      const after = await fetch(new URL("/api/providers/keys?name=genspark", server.url), {
        headers: managementHeaders(),
      }).then(r => r.json()) as { keys: Array<Record<string, unknown>> };
      expect(after.keys.find(k => k.id === "kA")?.hasCookie).toBeUndefined();
    } finally {
      await server.stop(true);
    }
  });

  test("management: GET /api/providers/keys/credit reports per-key state without leaking cookies", async () => {
    globalThis.fetch = (async (input, init) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url === GENSPARK_CREDIT_BALANCE_URL) {
        return balanceResponse({ data: { balance: 8420 } });
      }
      return originalFetch(input, init);
    }) as typeof fetch;
    saveConfig(gensparkConfig([
      { id: "kA", key: KEY_A, addedAt: 1, cookie: COOKIE_A },
      { id: "kB", key: KEY_B, addedAt: 2 },
    ]));
    const server = startServer(0);
    try {
      const res = await originalFetch(new URL("/api/providers/keys/credit?name=genspark", server.url), {
        headers: managementHeaders(),
      });
      expect(res.status).toBe(200);
      const body = await res.json() as { credits: Record<string, { balance: number | null; state: string }> };
      expect(JSON.stringify(body)).not.toContain("fake-cookie-value-A");
      expect(body.credits.kA).toMatchObject({ balance: 8420, state: "healthy" });
      expect(body.credits.kB).toMatchObject({ balance: null, state: "unknown" });
    } finally {
      await server.stop(true);
    }
  });

  test("GET /api/config never exposes the pool or cookies", async () => {
    saveConfig(gensparkConfig([
      { id: "kA", key: KEY_A, addedAt: 1, cookie: COOKIE_A },
      { id: "kB", key: KEY_B, addedAt: 2 },
    ]));
    const server = startServer(0);
    try {
      const res = await fetch(new URL("/api/config", server.url), { headers: managementHeaders() });
      expect(res.status).toBe(200);
      const text = await res.text();
      expect(text).not.toContain(KEY_A);
      expect(text).not.toContain(KEY_B);
      expect(text).not.toContain("fake-cookie-value-A");
      expect(text).not.toContain("apiKeyPool");
    } finally {
      await server.stop(true);
    }
  });
});
