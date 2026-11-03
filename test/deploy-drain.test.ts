import { test } from "node:test";
import assert from "node:assert/strict";
import { createWorker } from "../src/runs/worker.ts";
import { createDrainController } from "../src/runs/drain.ts";
import { createPostgresInstanceRegistry } from "../src/runs/instance-registry.ts";
import { createSlackRuntimeReconciler } from "../src/surfaces/slack-runtime.ts";
import { createPostgresMapFactory } from "../src/persistence/durable-map.ts";
import type { Orchestrator, OrchestratorInput } from "../src/core/orchestrator.ts";
import type { Principal } from "../src/types.ts";
import { createPostgresRunStore } from "../src/runs/postgres-run-store.ts";
import { testDatabaseUrl } from "./support/test-database.ts";
import { settle } from "./support/settle.ts";

const actor: Principal = { id: "internal:U1", type: "internal" };
const turn = (threadRef: string): OrchestratorInput => ({
  actor,
  conversation: { kind: "dm", threadRef, audience: [actor] },
  origin: { kind: "direct" },
  text: "x",
});

function bootBuild(url: string, deploymentKey: string, startedAt: number) {
  const { runs } = createPostgresRunStore(url);
  const handled: string[] = [];
  const cron = { running: false, starts: 0 };
  const slack = { connected: false, connects: 0 };
  const slackRuntime = createSlackRuntimeReconciler({
    startPaused: true,
    intervalMs: 20,
    load: async () => ({ version: "environment", config: deploymentKey }),
    startPlugin: async () => {
      slack.connected = true;
      slack.connects++;
      return {
        stop: async () => {
          slack.connected = false;
        },
      };
    },
  });
  const drain = createDrainController({
    registry: createPostgresInstanceRegistry(createPostgresMapFactory(url).pool, {
      instanceId: `${deploymentKey}-1`,
      deploymentKey,
      startedAt,
      livenessMs: 400,
    }),
    sweepMs: 25,
  });
  const orchestrator = {
    handleTurn: async (input: OrchestratorInput) => {
      handled.push(input.conversation.threadRef);
      return { status: "ok", reply: "done" };
    },
  } as unknown as Orchestrator;
  const worker = createWorker({ runs, orchestrator, leaseTtlMs: 10_000, pollMs: 5, canClaim: () => drain.canClaim() });
  worker.start();
  drain.start({
    resume() {
      cron.running = true;
      cron.starts++;
      slackRuntime.start();
    },
    async standDown() {
      cron.running = false;
      await slackRuntime.stop();
    },
  });
  return {
    runs,
    handled,
    cron,
    slack,
    async stop() {
      await drain.stop();
      await worker.stop();
      await slackRuntime.stop();
    },
  };
}

test("starting a newer build stands the older one down; stopping it brings the older one back", async () => {
  const url = testDatabaseUrl();
  const a = bootBuild(url, "deployment-a", 1_000);
  try {
    await settle(async () => a.cron.running && a.slack.connected);
    assert.ok(a.cron.running && a.slack.connected, "the only build runs cron and holds the Slack connection");
    const first = (await a.runs.enqueue({ sessionId: "s1", request: turn("t1"), maxAttempts: 3 })).run;
    assert.equal((await a.runs.waitFor(first.id, 5_000)).status, "done");
    assert.deepEqual(a.handled, ["t1"]);

    const b = bootBuild(url, "deployment-b", 2_000);
    try {
      await settle(async () => !a.cron.running && !a.slack.connected && b.cron.running && b.slack.connected);
      assert.equal(a.cron.running, false, "the older build stopped its cron scheduler");
      assert.equal(a.slack.connected, false, "the older build closed its Slack connection");
      assert.ok(b.cron.running && b.slack.connected, "the newer build runs cron and holds Slack");
      const second = (await b.runs.enqueue({ sessionId: "s2", request: turn("t2"), maxAttempts: 3 })).run;
      assert.equal((await b.runs.waitFor(second.id, 5_000)).status, "done");
      assert.deepEqual(a.handled, ["t1"], "the older build claimed no new turn");
      assert.deepEqual(b.handled, ["t2"]);
    } finally {
      await b.stop();
    }

    await settle(async () => a.cron.running && a.slack.connected);
    assert.ok(a.cron.running && a.slack.connected, "the older build resumed cron and Slack once the newer one died");
    assert.equal(a.cron.starts, 2);
    assert.equal(a.slack.connects, 2);
    const third = (await a.runs.enqueue({ sessionId: "s3", request: turn("t3"), maxAttempts: 3 })).run;
    assert.equal((await a.runs.waitFor(third.id, 5_000)).status, "done");
    assert.deepEqual(a.handled, ["t1", "t3"]);
  } finally {
    await a.stop();
  }
});
