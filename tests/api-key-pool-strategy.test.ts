/**
 * Phase 3: `apiKeyPoolStrategy` — proactive key selection per request.
 *
 * - "failover" (default / absent / unknown): active key sticks; nothing changes.
 * - "round-robin": requests cycle eligible keys via an in-memory cursor (never persisted);
 *   cooldown and telemetry-drained keys are skipped.
 * - "quota-aware": active key sticks UNLESS telemetry knows it is drained; unknown-credit
 *   keys stay eligible (cookie never required). Reactive failover still applies on top.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { saveConfig } from "../src/config";
import { clearGensparkCreditCache, setGensparkCreditEntryForTests } from "../src/providers/genspark-credit";
import {
  applyKeyPoolStrategy,
  clearKeyCooldowns,
  clearKeyPoolStrategyState,
  rotateKeyOn429,
} from "../src/providers/key-failover";
import { routeModel } from "../src/router";
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
  isolatedCodexHome = installIsolatedCodexHome("ocx-pool-strategy-codex-");
  testDir = mkdtempSync(join(tmpdir(), "ocx-pool-strategy-"));
  process.env.OPENCODEX_HOME = testDir;
  clearKeyCooldowns();
  clearKeyPoolStrategyState();
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
  clearKeyPoolStrategyState();
  clearGensparkCreditCache();
});

function provider(extra: Partial<OcxProviderConfig> = {}): OcxProviderConfig {
  return {
    adapter: "openai-chat",
    baseUrl: GENSPARK_BASE,
    authMode: "key",
    apiKey: KEY_A,
    apiKeyPool: [
      { id: "kA", key: KEY_A, addedAt: 1 },
      { id: "kB", key: KEY_B, addedAt: 2 },
      { id: "kC", key: KEY_C, addedAt: 3 },
    ],
    ...extra,
  } as OcxProviderConfig;
}

describe("applyKeyPoolStrategy (unit)", () => {
  test("default/absent/failover strategy is a strict no-op (same object, same key)", () => {
    for (const strategy of [undefined, "failover"] as const) {
      const p = provider(strategy ? { apiKeyPoolStrategy: strategy } : {});
      const out = applyKeyPoolStrategy("genspark", p);
      expect(out).toBe(p); // identity: zero-copy passthrough
      expect(out.apiKey).toBe(KEY_A);
    }
  });

  test("round-robin cycles A -> B -> C -> A across successive requests", () => {
    const p = provider({ apiKeyPoolStrategy: "round-robin" });
    const picks = [1, 2, 3, 4].map(() => applyKeyPoolStrategy("genspark", p).apiKey);
    expect(picks).toEqual([KEY_A, KEY_B, KEY_C, KEY_A]);
  });

  test("round-robin skips cooling keys", () => {
    const now = 1_000_000;
    const config = {
      port: 1, defaultProvider: "genspark",
      providers: { genspark: provider({ apiKeyPoolStrategy: "round-robin" }) },
    } as unknown as OcxConfig;
    // kB rate-limits: it cools for 60s.
    rotateKeyOn429(config, "genspark", null, now, KEY_B);
    const p = config.providers.genspark!;
    const picks = [1, 2, 3].map(() => applyKeyPoolStrategy("genspark", p, now).apiKey);
    expect(picks).toEqual([KEY_A, KEY_C, KEY_A]); // kB never picked while cooling
  });

  test("round-robin skips telemetry-drained keys and survives all-drained (stays reactive)", () => {
    const now = 1_000_000;
    const p = provider({ apiKeyPoolStrategy: "round-robin" });
    setGensparkCreditEntryForTests("genspark", "kB", { balance: 0, checkedAt: now, state: "exhausted" });
    const picks = [1, 2, 3].map(() => applyKeyPoolStrategy("genspark", p, now).apiKey);
    expect(picks).toEqual([KEY_A, KEY_C, KEY_A]);
    // Everything drained: return the provider unchanged (reactive failover will handle it).
    setGensparkCreditEntryForTests("genspark", "kA", { balance: 0, checkedAt: now, state: "exhausted" });
    setGensparkCreditEntryForTests("genspark", "kC", { balance: 0, checkedAt: now, state: "exhausted" });
    expect(applyKeyPoolStrategy("genspark", p, now).apiKey).toBe(KEY_A);
  });

  test("quota-aware sticks with the active key while its credit is healthy or unknown", () => {
    const now = 1_000_000;
    const p = provider({ apiKeyPoolStrategy: "quota-aware" });
    expect(applyKeyPoolStrategy("genspark", p, now)).toBe(p); // unknown: no move
    setGensparkCreditEntryForTests("genspark", "kA", { balance: 900, checkedAt: now, state: "healthy" });
    expect(applyKeyPoolStrategy("genspark", p, now)).toBe(p); // healthy: no move
  });

  test("quota-aware proactively moves off a telemetry-drained active key", () => {
    const now = 1_000_000;
    const p = provider({ apiKeyPoolStrategy: "quota-aware" });
    setGensparkCreditEntryForTests("genspark", "kA", { balance: 0, checkedAt: now, state: "exhausted" });
    const out = applyKeyPoolStrategy("genspark", p, now);
    expect(out.apiKey).toBe(KEY_B);
    // Pure selection: neither the input provider nor (implicitly) the config was mutated.
    expect(p.apiKey).toBe(KEY_A);
  });

  test("quota-aware keeps the drained key when NOTHING else is usable", () => {
    const now = 1_000_000;
    const p = provider({ apiKeyPoolStrategy: "quota-aware" });
    for (const id of ["kA", "kB", "kC"]) {
      setGensparkCreditEntryForTests("genspark", id, { balance: 0, checkedAt: now, state: "exhausted" });
    }
    expect(applyKeyPoolStrategy("genspark", p, now).apiKey).toBe(KEY_A);
  });

  test("single-key pools and oauth providers are never touched", () => {
    const single = provider({ apiKeyPoolStrategy: "round-robin", apiKeyPool: [{ id: "kA", key: KEY_A }] });
    expect(applyKeyPoolStrategy("genspark", single)).toBe(single);
    const oauth = provider({ apiKeyPoolStrategy: "round-robin", authMode: "oauth" });
    expect(applyKeyPoolStrategy("genspark", oauth)).toBe(oauth);
  });
});

describe("apiKeyPoolStrategy routing (integration)", () => {
  function config(strategy?: OcxProviderConfig["apiKeyPoolStrategy"]): OcxConfig {
    return {
      port: 0,
      hostname: "127.0.0.1",
      defaultProvider: "genspark",
      providers: {
        genspark: provider(strategy ? { apiKeyPoolStrategy: strategy } : {}),
      },
    } as OcxConfig;
  }

  test("routeModel applies round-robin per route call without persisting a cursor", () => {
    const cfg = config("round-robin");
    const k1 = routeModel(cfg, "genspark/gpt-test").provider.apiKey;
    const k2 = routeModel(cfg, "genspark/gpt-test").provider.apiKey;
    const k3 = routeModel(cfg, "genspark/gpt-test").provider.apiKey;
    expect([k1, k2, k3]).toEqual([KEY_A, KEY_B, KEY_C]);
    // The persisted active key is untouched — the cursor lives in memory only.
    expect(cfg.providers.genspark!.apiKey).toBe(KEY_A);
  });

  test("default strategy: repeated routes stay on the active key (Request 1..N -> key1)", () => {
    const cfg = config();
    for (let i = 0; i < 3; i++) {
      expect(routeModel(cfg, "genspark/gpt-test").provider.apiKey).toBe(KEY_A);
    }
  });

  test("round-robin requests hit the upstream with alternating keys AND reactive failover still applies", async () => {
    const seenAuth: string[] = [];
    globalThis.fetch = (async (input, init) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url === GENSPARK_CHAT) {
        const auth = new Headers(init?.headers).get("authorization") ?? "";
        seenAuth.push(auth);
        // Key B is broken: every request it makes 401s; A and C succeed.
        if (auth === `Bearer ${KEY_B}`) {
          return new Response(JSON.stringify({ error: { message: "Invalid API key" } }), {
            status: 401, headers: { "content-type": "application/json" },
          });
        }
        return new Response(JSON.stringify({
          id: "c", object: "chat.completion",
          choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }), { headers: { "content-type": "application/json" } });
      }
      return originalFetch(input, init);
    }) as typeof fetch;

    saveConfig(config("round-robin"));
    const server = startServer(0);
    try {
      const post = () => originalFetch(new URL("/v1/responses", server.url), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "genspark/gpt-test", input: "hi", stream: false }),
      });
      // Request 1 -> key A (ok). Request 2 -> key B (401) -> reactive failover replays.
      const res1 = await post();
      const res2 = await post();
      expect(res1.status).toBe(200);
      expect(res2.status).toBe(200);
      expect(seenAuth[0]).toBe(`Bearer ${KEY_A}`);
      expect(seenAuth[1]).toBe(`Bearer ${KEY_B}`); // round-robin picked B
      // The 401 replay went to a different, healthy key.
      expect(seenAuth[2]).not.toBe(`Bearer ${KEY_B}`);
      expect(seenAuth).toHaveLength(3);
      // Request 3: B is now cooling (invalid-auth) — round-robin must skip it.
      const res3 = await post();
      expect(res3.status).toBe(200);
      expect(seenAuth[3]).not.toBe(`Bearer ${KEY_B}`);
    } finally {
      await server.stop(true);
    }
  });

  test("config with a valid strategy round-trips through save/load; invalid is rejected at the schema", async () => {
    const { loadConfig } = await import("../src/config");
    saveConfig(config("quota-aware"));
    expect(loadConfig().providers.genspark?.apiKeyPoolStrategy).toBe("quota-aware");
    // Corrupt it by hand: the loader must not silently accept a misspelled strategy.
    const path = join(testDir, "config.json");
    const raw = JSON.parse(readFileSync(path, "utf-8")) as { providers: { genspark: Record<string, unknown> } };
    raw.providers.genspark.apiKeyPoolStrategy = "round-robbin";
    await Bun.write(path, JSON.stringify(raw, null, 2));
    const reloaded = loadConfig();
    // Invalid config falls back to defaults (backed up) — never a silently-wrong strategy.
    expect(reloaded.providers.genspark?.apiKeyPoolStrategy).not.toBe("round-robbin");
  });
});
