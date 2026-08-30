/**
 * Custom-provider key failover: quota (402 / Genspark credit exhaustion), invalid auth
 * (401 / key-attributable 403), bounded chains, streaming safety, and concurrency.
 *
 * Spec: ONE Codex request → key A quota-exhausted → auto-select key B → auto-REPLAY the
 * SAME request → key B succeeds → caller receives the answer (never the quota error).
 *
 * Fake keys only (gsk_test_key_A / gsk_test_key_B / gsk_test_key_C) — never real credentials.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { saveConfig } from "../src/config";
import { isGensparkProvider, isGensparkQuotaErrorText } from "../src/providers/genspark";
import {
  classifyKeyPoolFailure,
  clearKeyCooldowns,
  getKeyCooldownUntil,
  keyFailureRecoveryKind,
  rotateKeyOnFailure,
  type KeyPoolFailure,
} from "../src/providers/key-failover";
import { clearReasoningReplayCacheForTests } from "../src/responses/reasoning-replay-cache";
import { startServer } from "../src/server";
import type { OcxConfig, OcxProviderConfig } from "../src/types";
import { installIsolatedCodexHome, type IsolatedCodexHome } from "./helpers/isolated-codex-home";

const KEY_A = "gsk_test_key_A";
const KEY_B = "gsk_test_key_B";
const KEY_C = "gsk_test_key_C";
const GENSPARK_BASE = "https://www.genspark.ai/api/llm_proxy/v1";
const GENSPARK_CHAT = `${GENSPARK_BASE}/chat/completions`;

let testDir = "";
let previousHome: string | undefined;
let isolatedCodexHome: IsolatedCodexHome | null = null;
const originalFetch = globalThis.fetch;

beforeEach(() => {
  previousHome = process.env.OPENCODEX_HOME;
  isolatedCodexHome = installIsolatedCodexHome("ocx-genspark-e2e-codex-");
  testDir = mkdtempSync(join(tmpdir(), "ocx-genspark-e2e-"));
  process.env.OPENCODEX_HOME = testDir;
  clearKeyCooldowns();
  clearReasoningReplayCacheForTests();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
  else process.env.OPENCODEX_HOME = previousHome;
  isolatedCodexHome?.restore();
  isolatedCodexHome = null;
  if (testDir) rmSync(testDir, { recursive: true, force: true });
  clearKeyCooldowns();
  clearReasoningReplayCacheForTests();
});

// ---------- unit: Genspark classifier ----------

describe("genspark classifier (unit)", () => {
  test("isGensparkProvider matches by name and by host", () => {
    expect(isGensparkProvider("genspark", { baseUrl: "https://elsewhere.example/v1" })).toBe(true);
    expect(isGensparkProvider("GenSpark ", { baseUrl: "" })).toBe(true);
    expect(isGensparkProvider("custom", { baseUrl: GENSPARK_BASE })).toBe(true);
    expect(isGensparkProvider("custom", { baseUrl: "https://api.genspark.ai/v1" })).toBe(true);
    expect(isGensparkProvider("custom", { baseUrl: "https://genspark.ai.evil.example/v1" })).toBe(false);
    expect(isGensparkProvider("custom", { baseUrl: "https://api.openai.com/v1" })).toBe(false);
    expect(isGensparkProvider("custom", { baseUrl: "not a url" })).toBe(false);
  });

  test("quota wording matches; generic rate-limit and success wording do not", () => {
    for (const text of [
      "Insufficient credits, please top up",
      '{"error":{"message":"insufficient credit"}}',
      "quota exceeded for this key",
      "Quota exhausted",
      "credit balance exhausted",
      "credits exhausted",
      "no credits remaining",
      "You are out of credits",
    ]) {
      expect(isGensparkQuotaErrorText(text)).toBe(true);
    }
    for (const text of [
      "",
      "too many requests",
      "internal server error",
      "the model wrote an essay about credit cards",
      '{"usageStatus":"unreported"}',
    ]) {
      expect(isGensparkQuotaErrorText(text)).toBe(false);
    }
  });
});

// ---------- unit: classifyKeyPoolFailure ----------

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

const GENERIC_PROVIDER = { baseUrl: "https://api.example.com/v1" };
const GENSPARK_PROVIDER = { baseUrl: GENSPARK_BASE };

describe("classifyKeyPoolFailure (unit)", () => {
  test("429 → rate-limit with Retry-After; 402 → quota; 401 → invalid-auth", async () => {
    const rl = await classifyKeyPoolFailure(
      jsonResponse(429, { error: { message: "slow down" } }, { "retry-after": "30" }),
      "p", GENERIC_PROVIDER,
    );
    expect(rl).toEqual({ kind: "rate-limit", status: 429, retryAfter: "30" });
    expect(keyFailureRecoveryKind(rl!)).toBe("key-429");

    const quota = await classifyKeyPoolFailure(jsonResponse(402, { error: { message: "payment required" } }), "p", GENERIC_PROVIDER);
    expect(quota).toEqual({ kind: "quota-exhausted", status: 402, retryAfter: null });
    expect(keyFailureRecoveryKind(quota!)).toBe("key-quota");

    const auth = await classifyKeyPoolFailure(jsonResponse(401, { error: { message: "unauthorized" } }), "p", GENERIC_PROVIDER);
    expect(auth).toEqual({ kind: "invalid-auth", status: 401, retryAfter: null });
    expect(keyFailureRecoveryKind(auth!)).toBe("key-auth");
  });

  test("2xx and 5xx never classify — a 200 with usageStatus unreported can never rotate", async () => {
    expect(await classifyKeyPoolFailure(
      jsonResponse(200, { choices: [], usageStatus: "unreported" }),
      "genspark", GENSPARK_PROVIDER,
    )).toBeNull();
    expect(await classifyKeyPoolFailure(jsonResponse(500, { error: { message: "insufficient credits" } }), "genspark", GENSPARK_PROVIDER)).toBeNull();
    expect(await classifyKeyPoolFailure(jsonResponse(503, {}), "p", GENERIC_PROVIDER)).toBeNull();
  });

  test("403 rotates only when the body names the credential", async () => {
    expect(await classifyKeyPoolFailure(
      jsonResponse(403, { error: { message: "Invalid API key provided" } }),
      "p", GENERIC_PROVIDER,
    )).toMatchObject({ kind: "invalid-auth", status: 403 });
    expect(await classifyKeyPoolFailure(
      jsonResponse(403, { error: { message: "API key expired or revoked" } }),
      "p", GENERIC_PROVIDER,
    )).toMatchObject({ kind: "invalid-auth" });
    // Provider/model-level 403s must NOT burn the pool.
    expect(await classifyKeyPoolFailure(
      jsonResponse(403, { error: { message: "This model is not available in your region" } }),
      "p", GENERIC_PROVIDER,
    )).toBeNull();
  });

  test("Genspark quota text under a non-429 status classifies as quota-exhausted — only for Genspark", async () => {
    expect(await classifyKeyPoolFailure(
      jsonResponse(400, { error: { message: "Insufficient credits. Please purchase more credits." } }),
      "genspark", GENSPARK_PROVIDER,
    )).toMatchObject({ kind: "quota-exhausted", status: 400 });
    // Same body on a non-Genspark provider: plain 400, no rotation.
    expect(await classifyKeyPoolFailure(
      jsonResponse(400, { error: { message: "Insufficient credits. Please purchase more credits." } }),
      "p", GENERIC_PROVIDER,
    )).toBeNull();
    // Genspark 400 without quota wording: plain request error, no rotation.
    expect(await classifyKeyPoolFailure(
      jsonResponse(400, { error: { message: "invalid request: messages missing" } }),
      "genspark", GENSPARK_PROVIDER,
    )).toBeNull();
  });

  test("classification clones — the original body stays readable for error reporting", async () => {
    const response = jsonResponse(400, { error: { message: "insufficient credits" } });
    await classifyKeyPoolFailure(response, "genspark", GENSPARK_PROVIDER);
    expect(response.bodyUsed).toBe(false);
    const body = await response.json() as { error: { message: string } };
    expect(body.error.message).toBe("insufficient credits");
  });
});

// ---------- unit: rotateKeyOnFailure chain bounding + concurrency ----------

function makeConfig(provider: Partial<OcxProviderConfig>): OcxConfig {
  return {
    port: 10199,
    defaultProvider: "p",
    providers: {
      p: { adapter: "openai-chat", baseUrl: "https://api.example.com/v1", ...provider } as OcxProviderConfig,
    },
  } as OcxConfig;
}

function pool3(): OcxProviderConfig["apiKeyPool"] {
  return [
    { id: "k1", key: KEY_A, addedAt: 1 },
    { id: "k2", key: KEY_B, addedAt: 2 },
    { id: "k3", key: KEY_C, addedAt: 3 },
  ];
}

const QUOTA_FAILURE: KeyPoolFailure = { kind: "quota-exhausted", status: 402, retryAfter: null };
const AUTH_FAILURE: KeyPoolFailure = { kind: "invalid-auth", status: 401, retryAfter: null };

describe("rotateKeyOnFailure (unit)", () => {
  test("quota and invalid-auth failures cool the key far longer than a rate limit", () => {
    const now = 1_000_000;
    const quotaConfig = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    rotateKeyOnFailure(quotaConfig, "p", QUOTA_FAILURE, now, KEY_A);
    expect(getKeyCooldownUntil("p", "k1", now)).toBe(now + 30 * 60_000);
    clearKeyCooldowns("p");
    const authConfig = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    rotateKeyOnFailure(authConfig, "p", AUTH_FAILURE, now, KEY_A);
    expect(getKeyCooldownUntil("p", "k1", now)).toBe(now + 30 * 60_000);
  });

  test("a quota-cooled key is not selected while a healthy key exists", () => {
    const now = 1_000_000;
    const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    // k1 drains its quota, chain rotates to k2.
    expect(rotateKeyOnFailure(config, "p", QUOTA_FAILURE, now, KEY_A)?.apiKey).toBe(KEY_B);
    // A LATER request's chain rate-limits on k2: it must skip quota-cooled k1 and land on k3.
    expect(
      rotateKeyOnFailure(config, "p", { kind: "rate-limit", status: 429, retryAfter: null }, now + 120_000, KEY_B)?.apiKey,
    ).toBe(KEY_C);
  });

  test("a chain never retries a key it already attempted, even after its cooldown lapses", () => {
    const now = 1_000_000;
    const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    const chain = new Set<string>();
    // Retry-After: 0 → ~1ms cooldowns, so only the chain set can bound the walk.
    const rl = (retryAfter: string): KeyPoolFailure => ({ kind: "rate-limit", status: 429, retryAfter });
    expect(rotateKeyOnFailure(config, "p", rl("0"), now, KEY_A, chain)?.apiKey).toBe(KEY_B);
    expect(rotateKeyOnFailure(config, "p", rl("0"), now + 10_000, KEY_B, chain)?.apiKey).toBe(KEY_C);
    // k1's 1ms cooldown lapsed long ago — the chain must still refuse to loop back.
    expect(rotateKeyOnFailure(config, "p", rl("0"), now + 20_000, KEY_C, chain)).toBeNull();
    expect(chain).toEqual(new Set(["k1", "k2", "k3"]));
  });

  test("concurrent chains stay independent; the race-lost request reuses the live key (CAS)", () => {
    const now = 1_000_000;
    const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
    const chainA = new Set<string>();
    const chainB = new Set<string>();
    // Request A (used key A) hits quota: rotates A → B.
    expect(rotateKeyOnFailure(config, "p", QUOTA_FAILURE, now, KEY_A, chainA)?.apiKey).toBe(KEY_B);
    expect(chainA).toEqual(new Set(["k1", "k2"]));
    // Request B also sent key A concurrently and fails AFTER A's rotation: it must not
    // punish the innocent live key — it retries with key B and records it in ITS OWN chain.
    expect(rotateKeyOnFailure(config, "p", QUOTA_FAILURE, now, KEY_A, chainB)?.apiKey).toBe(KEY_B);
    expect(chainB.has("k2")).toBe(true);
    expect(getKeyCooldownUntil("p", "k2", now)).toBeNull();
    // Request A's chain state never leaked into B: B can still rotate B → C on a real failure.
    expect(rotateKeyOnFailure(config, "p", QUOTA_FAILURE, now, KEY_B, chainB)?.apiKey).toBe(KEY_C);
    // A's chain, having already burned k1+k2, can only reach k3 — then nothing.
    expect(rotateKeyOnFailure(config, "p", QUOTA_FAILURE, now, KEY_B, chainA)?.apiKey).toBe(KEY_C);
    expect(rotateKeyOnFailure(config, "p", QUOTA_FAILURE, now, KEY_C, chainA)).toBeNull();
  });

  test("rotation logs key IDs only — never key material", () => {
    const warnings: string[] = [];
    const originalWarn = console.warn;
    console.warn = (...parts: unknown[]) => { warnings.push(parts.map(String).join(" ")); };
    try {
      const config = makeConfig({ apiKey: KEY_A, apiKeyPool: pool3() });
      rotateKeyOnFailure(config, "p", QUOTA_FAILURE, 1_000_000, KEY_A);
    } finally {
      console.warn = originalWarn;
    }
    const joined = warnings.join("\n");
    expect(joined).toContain("quota-exhausted");
    expect(joined).not.toContain(KEY_A);
    expect(joined).not.toContain(KEY_B);
    expect(joined).not.toContain(KEY_C);
  });
});

// ---------- e2e: /v1/responses through the Genspark chat path ----------

function gensparkConfig(pool: { id: string; key: string }[]): OcxConfig {
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
        apiKeyPool: pool.map((entry, index) => ({ ...entry, addedAt: index + 1 })),
      },
    },
  } as OcxConfig;
}

function chatCompletion(text: string, extra: Record<string, unknown> = {}): Response {
  return new Response(JSON.stringify({
    id: "chatcmpl-genspark",
    object: "chat.completion",
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
    usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
    ...extra,
  }), { headers: { "content-type": "application/json" } });
}

function sseSuccess(text: string): Response {
  const frames = [
    { id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] },
    { id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: text }, finish_reason: null }] },
    {
      id: "c", object: "chat.completion.chunk",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
    },
  ];
  const body = frames.map(frame => `data: ${JSON.stringify(frame)}\n\n`).join("") + "data: [DONE]\n\n";
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

interface UpstreamPlan {
  respond: (attempt: number) => Response;
}

function mockGensparkUpstream(plan: UpstreamPlan): { seenAuth: string[]; seenBodies: string[] } {
  const seenAuth: string[] = [];
  const seenBodies: string[] = [];
  globalThis.fetch = (async (input, init) => {
    const url = input instanceof Request ? input.url : String(input);
    if (url === GENSPARK_CHAT) {
      seenAuth.push(new Headers(init?.headers).get("authorization") ?? "");
      seenBodies.push(String(init?.body ?? ""));
      return plan.respond(seenAuth.length);
    }
    return originalFetch(input, init);
  }) as typeof fetch;
  return { seenAuth, seenBodies };
}

async function postResponses(serverUrl: URL | string, body: Record<string, unknown>): Promise<Response> {
  return originalFetch(new URL("/v1/responses", serverUrl), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("genspark key failover (end-to-end)", () => {
  test("ACCEPTANCE: 402 quota on key A → rotate to key B → SAME request replayed → caller gets the answer", async () => {
    const { clearRequestLogsForTests, getRequestLogEntries } = await import("../src/server/request-log");
    clearRequestLogsForTests();
    const { seenAuth, seenBodies } = mockGensparkUpstream({
      respond: attempt => attempt === 1
        ? jsonResponse(402, { error: { message: "Payment required: credits exhausted" } })
        : chatCompletion("answer from key B"),
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]));
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hello genspark", stream: false });
      expect(res.status).toBe(200);
      const json = await res.json() as { output?: { type: string; content?: { text?: string }[] }[] };
      expect(json.output?.find(o => o.type === "message")?.content?.[0]?.text).toBe("answer from key B");
      // Same request, two keys: attempt 1 = key A, attempt 2 = key B, identical payload.
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`, `Bearer ${KEY_B}`]);
      expect(seenBodies[1]).toBe(seenBodies[0]);
      // Attempt metadata records the quota rotation.
      const entry = getRequestLogEntries().at(-1);
      expect(entry?.status).toBe(200);
      expect(entry?.attempts?.flatMap(a => a.recoveryKinds ?? [])).toContain("key-quota");
      // The drained key is quota-cooled (30 min), not the short 429 window.
      const until = getKeyCooldownUntil("genspark", "kA", Date.now());
      expect(until).not.toBeNull();
      expect(until! - Date.now()).toBeGreaterThan(10 * 60_000);
    } finally {
      await server.stop(true);
      clearRequestLogsForTests();
    }
  });

  test("Genspark quota TEXT under a 400 status rotates and replays", async () => {
    const { seenAuth } = mockGensparkUpstream({
      respond: attempt => attempt === 1
        ? jsonResponse(400, { error: { message: "Insufficient credits. Please visit genspark.ai/pricing." } })
        : chatCompletion("recovered"),
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]));
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hi", stream: false });
      expect(res.status).toBe(200);
      const json = await res.json() as { output?: { type: string; content?: { text?: string }[] }[] };
      expect(json.output?.find(o => o.type === "message")?.content?.[0]?.text).toBe("recovered");
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`, `Bearer ${KEY_B}`]);
    } finally {
      await server.stop(true);
    }
  });

  test("MANDATORY: a 200 with usageStatus unreported is SUCCESS — exactly one attempt, no rotation", async () => {
    const { seenAuth } = mockGensparkUpstream({
      respond: () => chatCompletion("fine answer", { usageStatus: "unreported" }),
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]));
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hi", stream: false });
      expect(res.status).toBe(200);
      const json = await res.json() as { output?: { type: string; content?: { text?: string }[] }[] };
      expect(json.output?.find(o => o.type === "message")?.content?.[0]?.text).toBe("fine answer");
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`]);
      expect(getKeyCooldownUntil("genspark", "kA", Date.now())).toBeNull();
    } finally {
      await server.stop(true);
    }
  });

  test("401 invalid key rotates; the bad key cools for the long window", async () => {
    const { seenAuth } = mockGensparkUpstream({
      respond: attempt => attempt === 1
        ? jsonResponse(401, { error: { message: "Invalid API key" } })
        : chatCompletion("authed with B"),
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]));
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hi", stream: false });
      expect(res.status).toBe(200);
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`, `Bearer ${KEY_B}`]);
      const until = getKeyCooldownUntil("genspark", "kA", Date.now());
      expect(until).not.toBeNull();
      expect(until! - Date.now()).toBeGreaterThan(10 * 60_000);
    } finally {
      await server.stop(true);
    }
  });

  test("non-key-attributable 403 does NOT rotate — single attempt, error surfaced", async () => {
    const { seenAuth } = mockGensparkUpstream({
      respond: () => jsonResponse(403, { error: { message: "This model is not enabled for your workspace" } }),
    });
    // Non-Genspark provider name AND host so only the 403 heuristic applies.
    const config = gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]);
    saveConfig(config);
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hi", stream: false });
      expect(res.status).toBeGreaterThanOrEqual(400);
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`]);
      expect(getKeyCooldownUntil("genspark", "kA", Date.now())).toBeNull();
      expect(getKeyCooldownUntil("genspark", "kB", Date.now())).toBeNull();
    } finally {
      await server.stop(true);
    }
  });

  test("all keys exhausted: attempts bounded by pool size, one clear final error", async () => {
    const { seenAuth } = mockGensparkUpstream({
      respond: () => jsonResponse(402, { error: { message: "credits exhausted" } }),
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }, { id: "kC", key: KEY_C }]));
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hi", stream: false });
      expect(res.status).toBeGreaterThanOrEqual(400);
      const json = await res.json() as { error?: { message?: string } };
      expect(json.error?.message).toBeTruthy();
      // Exactly pool-size attempts: A, B, C — no key retried twice, no infinite loop.
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`, `Bearer ${KEY_B}`, `Bearer ${KEY_C}`]);
      // Full response never leaks a key.
      expect(JSON.stringify(json)).not.toContain(KEY_A);
      expect(JSON.stringify(json)).not.toContain(KEY_B);
      expect(JSON.stringify(json)).not.toContain(KEY_C);
    } finally {
      await server.stop(true);
    }
  });

  test("STREAMING: quota failure before any output rotates and the client sees only the successful stream", async () => {
    const { seenAuth } = mockGensparkUpstream({
      respond: attempt => attempt === 1
        ? jsonResponse(402, { error: { message: "credits exhausted" } })
        : sseSuccess("streamed from key B"),
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]));
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hi", stream: true });
      expect(res.status).toBe(200);
      const text = await res.text();
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`, `Bearer ${KEY_B}`]);
      expect(text).toContain("streamed from key B");
      expect(text).toContain("event: response.completed");
      // The quota error never leaks into the client stream.
      expect(text).not.toContain("event: response.failed");
      expect(text).not.toContain("credits exhausted");
    } finally {
      await server.stop(true);
    }
  });

  test("PARTIAL-STREAM SAFETY: a mid-stream failure after output began is NOT blindly replayed on another key", async () => {
    const encoder = new TextEncoder();
    const { seenAuth } = mockGensparkUpstream({
      respond: () => new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            id: "c", object: "chat.completion.chunk",
            choices: [{ index: 0, delta: { role: "assistant", content: "partial output " }, finish_reason: null }],
          })}\n\n`));
          controller.error(new Error("upstream connection reset mid-stream"));
        },
      }), { headers: { "content-type": "text/event-stream" } }),
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]));
    const server = startServer(0);
    try {
      const res = await postResponses(server.url, { model: "genspark/gpt-test", input: "hi", stream: true });
      await res.text().catch(() => "");
      // Key failover must never fire after bytes were emitted: key B is never contacted.
      expect(seenAuth).toEqual([`Bearer ${KEY_A}`]);
      expect(getKeyCooldownUntil("genspark", "kA", Date.now())).toBeNull();
    } finally {
      await server.stop(true);
    }
  });

  test("CONCURRENCY: two simultaneous requests each replay once and both succeed", async () => {
    let firstAttemptSeen = false;
    const { seenAuth } = mockGensparkUpstream({
      respond: () => {
        // The first physical attempt (whichever request wins) drains key A;
        // every subsequent attempt succeeds regardless of key.
        if (!firstAttemptSeen) {
          firstAttemptSeen = true;
          return jsonResponse(402, { error: { message: "credits exhausted" } });
        }
        return chatCompletion("concurrent ok");
      },
    });
    saveConfig(gensparkConfig([{ id: "kA", key: KEY_A }, { id: "kB", key: KEY_B }]));
    const server = startServer(0);
    try {
      const [res1, res2] = await Promise.all([
        postResponses(server.url, { model: "genspark/gpt-test", input: "first", stream: false }),
        postResponses(server.url, { model: "genspark/gpt-test", input: "second", stream: false }),
      ]);
      expect(res1.status).toBe(200);
      expect(res2.status).toBe(200);
      const [json1, json2] = await Promise.all([res1.json(), res2.json()]) as Array<{ output?: { type: string; content?: { text?: string }[] }[] }>;
      expect(json1.output?.find(o => o.type === "message")?.content?.[0]?.text).toBe("concurrent ok");
      expect(json2.output?.find(o => o.type === "message")?.content?.[0]?.text).toBe("concurrent ok");
      // Bounded: at most one extra attempt beyond the two requests (the single 402 replay).
      expect(seenAuth.length).toBeLessThanOrEqual(3);
      // key B ends healthy — the CAS guard never cooled the replacement key.
      expect(getKeyCooldownUntil("genspark", "kB", Date.now())).toBeNull();
    } finally {
      await server.stop(true);
    }
  });
});
