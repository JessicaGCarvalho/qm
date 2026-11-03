import { randomUUID } from "node:crypto";
import { test } from "node:test";
import assert from "node:assert/strict";
import { createAgent } from "../src/agent/agent-run.ts";
import type { AgentTurnInput, SteeredApproval } from "../src/agent/agent.ts";
import type { NewEntry } from "../src/sessions/session-store.ts";
import type { ToolContext } from "../src/tools/primitives.ts";
import type { ScopeId, SessionEntry, TurnRequest } from "../src/types.ts";
import { createPostgresRunSignalStore } from "../src/runs/postgres-run-signal-store.ts";
import { testDatabaseUrl } from "./support/test-database.ts";

type Sink = Array<{ seq: number; type: string; payload: unknown }>;
type ProviderMessage = { role: string; content: Array<Record<string, unknown>> | string };

const scope = "personal:tester" as ScopeId;

function sse(events: Array<Record<string, unknown>>): Response {
  const body = events.map((event) => `event: ${event.type as string}\ndata: ${JSON.stringify(event)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

function gatedSse(events: Array<Record<string, unknown>>, gate: Promise<void>): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      for (const event of events.slice(0, -1))
        controller.enqueue(encoder.encode(`event: ${event.type as string}\ndata: ${JSON.stringify(event)}\n\n`));
      await gate;
      const last = events.at(-1)!;
      controller.enqueue(encoder.encode(`event: ${last.type as string}\ndata: ${JSON.stringify(last)}\n\n`));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
}

function textReplyEvents(text: string): Array<Record<string, unknown>> {
  return [
    {
      type: "message_start",
      message: {
        id: "msg_1",
        type: "message",
        role: "assistant",
        content: [],
        model: "claude-test",
        stop_reason: null,
        usage: { input_tokens: 5, output_tokens: 0 },
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 3 } },
    { type: "message_stop" },
  ];
}

function turnInput(sessionId: string, sink: Sink, overrides: Partial<AgentTurnInput>): AgentTurnInput {
  let seq = 1;
  return {
    session: { id: sessionId } as AgentTurnInput["session"],
    input: "carry on",
    systemPrompt: "BASE",
    history: [],
    tools: {} as ToolContext,
    scopeLabel: scope,
    orgScopeId: "org:test" as ScopeId,
    emit: async (entry: NewEntry) => {
      const appended = { ...entry, sessionId, parentSeq: null, seq: seq++, createdAt: Date.now() };
      sink.push({ seq: appended.seq, type: entry.type, payload: entry.payload });
      return appended as SessionEntry;
    },
    recordModelCall: () => {},
    ...overrides,
  };
}

function toolResults(messages: ProviderMessage[]): Array<Record<string, unknown>> {
  return messages.flatMap((m) =>
    Array.isArray(m.content) ? m.content.filter((block) => block.type === "tool_result") : [],
  );
}

test("an approved call runs once, exactly as stored, before the model sees the approval turn", async () => {
  const agent = createAgent({ apiKey: "sk-test" });
  const sink: Sink = [];
  const queries: string[] = [];
  const requests: ProviderMessage[][] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    requests.push((JSON.parse(String(init?.body ?? "{}")) as { messages?: ProviderMessage[] }).messages ?? []);
    return sse(textReplyEvents("the reminder is scheduled"));
  }) as typeof globalThis.fetch;
  try {
    const result = await agent.runTurn(
      turnInput("approved-pre", sink, {
        input: "(approval note)",
        tools: {
          async history(q: string) {
            queries.push(q);
            return ["user#1: reminder goes out at noon Taipei time"];
          },
        } as unknown as ToolContext,
        approvedToolCall: {
          callId: "approved_req1_ab12",
          requestId: "req1",
          tool: "history",
          input: { query: "noon" },
        },
      }),
    );
    assert.equal(result.reply, "the reminder is scheduled");
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.deepEqual(queries, ["noon"], "the stored arguments ran once; the model did not re-propose them");
  assert.deepEqual(
    sink.map((e) => e.type),
    ["tool_call", "tool_result", "user", "assistant"],
  );
  for (const entry of sink.slice(0, 2)) {
    assert.equal((entry.payload as { callId?: string }).callId, "approved_req1_ab12");
    assert.equal((entry.payload as { approvedRequestId?: string }).approvedRequestId, "req1");
  }
  assert.equal(requests.length, 1);
  const flattened = JSON.stringify(requests[0]);
  const resultAt = flattened.indexOf("noon Taipei time");
  assert.ok(resultAt > 0, "the approved call's result reaches the model");
  assert.ok(flattened.indexOf("(approval note)") > resultAt, "the result precedes the approval turn's note");
  assert.equal(toolResults(requests[0]!).length, 1);
});

test("an approved call whose tool is missing records an error result instead of failing the turn", async () => {
  const agent = createAgent({ apiKey: "sk-test" });
  const sink: Sink = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => sse(textReplyEvents("noted"))) as typeof globalThis.fetch;
  try {
    await agent.runTurn(
      turnInput("approved-missing", sink, {
        readOnly: true,
        approvedToolCall: { callId: "approved_req9_cd34", requestId: "req9", tool: "no_such_tool", input: {} },
      }),
    );
  } finally {
    globalThis.fetch = realFetch;
  }
  const result = sink.find((e) => e.type === "tool_result");
  assert.ok(result);
  assert.equal((result.payload as { isError?: boolean }).isError, true);
  assert.match(String((result.payload as { result?: string }).result), /not available on this turn/);
});

function approvalRequest(requestId: string, approved: boolean): TurnRequest {
  return {
    surface: "slack",
    actor: { externalId: "U1", displayName: "Gabrielle" },
    conversation: { kind: "dm", threadRef: "dm:U1:t1" },
    text: "make the reminder",
    approval: { requestId, approved, ...(approved ? { scope: "once" as const } : {}) },
  };
}

for (const approved of [true, false]) {
  test(`a ${approved ? "approval" : "denial"} clicked during a live run reaches that run's model`, async () => {
    const signals = createPostgresRunSignalStore(testDatabaseUrl());
    const agent = createAgent({ apiKey: "sk-test", signals });
    const runId = `run-live-approval-${approved}-${randomUUID()}`;
    const sink: Sink = [];
    const queries: string[] = [];
    const requests: ProviderMessage[][] = [];
    let release = () => {};
    const released = new Promise<void>((resolve) => {
      release = resolve;
    });
    const decided: TurnRequest[] = [];
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      requests.push((JSON.parse(String(init?.body ?? "{}")) as { messages?: ProviderMessage[] }).messages ?? []);
      if (requests.length === 1) {
        setTimeout(() => {
          void signals
            .send(runId, {
              kind: "steer",
              text: "[approval decision]",
              ts: "approval:req2",
              dedupeKey: `approval:${runId}`,
              request: approvalRequest("req2", approved),
            })
            .then(
              (ok) => console.error("DBG sent", ok),
              (e) => console.error("DBG send err", e),
            );
          setTimeout(release, 400);
        }, 30);
        return gatedSse(textReplyEvents("looking at the sheet"), released);
      }
      return sse(textReplyEvents("all set"));
    }) as typeof globalThis.fetch;
    try {
      const result = await agent.runTurn(
        turnInput(`live-${approved}`, sink, {
          runId,
          tools: {
            async history(q: string) {
              queries.push(q);
              return ["user#4: reminder created as push 18767"];
            },
          } as unknown as ToolContext,
          resolveSteeredApproval: async (request): Promise<SteeredApproval> => {
            decided.push(request);
            if (!request.approval?.approved) return { kind: "denied", summary: "history", approver: "Gabrielle" };
            return {
              kind: "run",
              summary: "history",
              approver: "Gabrielle",
              call: { callId: "approved_req2_ef56", requestId: "req2", tool: "history", input: { query: "push" } },
            };
          },
        }),
      );
      assert.equal(result.reply, "all set");
    } finally {
      globalThis.fetch = realFetch;
    }
    assert.equal(decided.length, 1, "the decision is resolved once");
    assert.deepEqual(queries, approved ? ["push"] : [], approved ? "the approved call ran once" : "nothing ran");
    assert.deepEqual(await signals.takePending(runId), [], "the decision signal is acknowledged");
    assert.equal(requests.length, 2);
    const followUp = JSON.stringify(requests[1]);
    if (approved) {
      assert.match(followUp, /Gabrielle approved the pending call history, and QM ran it exactly as approved/);
      assert.match(followUp, /push 18767/);
      const recorded = sink.filter((e) => (e.payload as { approvedRequestId?: string }).approvedRequestId === "req2");
      assert.deepEqual(
        recorded.map((e) => e.type),
        ["tool_call", "tool_result"],
      );
    } else {
      assert.match(followUp, /Gabrielle denied the pending call history, so it did not run/);
      assert.ok(!sink.some((e) => e.type === "tool_call"));
    }
  });
}
