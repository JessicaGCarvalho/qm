import { TURN_STOPPED_KIND, type ReplayImageRef } from "./replay.ts";
import { randomBytes } from "node:crypto";
import {
  parseSecurityScreenVerdict,
  SECURITY_SCREEN_STEP,
  SECURITY_SCREEN_SYSTEM_PROMPT,
} from "../security/security-posture.ts";
import type { ScopeId, SessionEntry } from "../types.ts";
import { createAgentTools, type AgentToolsOptions, type ToolContextRef } from "./agent-tools.ts";
import { goalWorkFromEntries } from "./goal.ts";
import type {
  ApprovedToolCall,
  TurnLlmRequestRecord,
  ModelUtilities,
  AgentTurnInput,
  AgentTurnResult,
} from "./agent.ts";
import { sanitizeTitle, TITLE_GENERATION_PROMPT, titleUserPrompt } from "./agent-run.ts";
import { errMessage, swallowAs } from "../util/errors.ts";
import { INTERRUPTED_TOOL_RESULT } from "./context-compaction.ts";

export interface ToolPlumbing {
  reachExec?: boolean;
  controlTools?: boolean;
  sandboxResources?: boolean;
  execTimeoutMs?: number;
  execTimeoutCeilingMs?: number;
  backgroundJobTtlMs?: number;
  backgroundJobTtlMaxMs?: number;
  sandboxCapabilityTtlMs?: number;
}

export type BridgedTool = {
  name: string;
  description: string;
  parameters: unknown;
  execute(
    callId: string,
    args: unknown,
  ): Promise<{
    content?: Array<{ type?: string; text?: string; data?: string; mimeType?: string }>;
    terminate?: boolean;
  }>;
};

export interface SteerIntake {
  text: string;
  prompt?: string;
  ts?: string;
  attachments?: AgentTurnInput["attachments"];
  images?: AgentTurnInput["images"];
  author?: string;
  acknowledge?: () => Promise<void>;
}

export function imageRefs(images: AgentTurnInput["images"]): { images?: ReplayImageRef[] } {
  if (!images?.length) return {};
  return {
    images: images.map((image) => ({
      mimeType: image.mimeType,
      ...(image.artifactId ? { artifactId: image.artifactId } : {}),
    })),
  };
}

export async function recordSteerIntake(turn: AgentTurnInput, steer: SteerIntake): Promise<void> {
  await turn.emit({
    type: "user",
    payload: {
      text: steer.text,
      ...(steer.prompt !== undefined && steer.prompt !== steer.text ? { prompt: steer.prompt } : {}),
      ...(steer.ts ? { ts: steer.ts } : {}),
      steered: true,
      ...(steer.attachments?.length ? { attachments: steer.attachments } : {}),
      ...imageRefs(steer.images),
    },
    scopeLabel: turn.scopeLabel,
    ...(steer.author ? { author: steer.author } : {}),
  });
  await steer.acknowledge?.().catch(swallowAs("steer acknowledge", undefined));
}

const STOPPED_FALLBACK_REASON = "unexpectedly";

export async function recordStoppedTurn(
  turn: AgentTurnInput,
  partial: string,
  reason: string = STOPPED_FALLBACK_REASON,
  replay: Record<string, unknown> = {},
): Promise<SessionEntry | null> {
  if (turn.shutdown?.aborted) return null;
  const answer = partial.trim()
    ? await turn.emit({
        type: "assistant",
        payload: { text: partial, stopped: true, ...replay },
        scopeLabel: turn.scopeLabel,
      })
    : null;
  const note = await turn.emit({
    type: "system",
    payload: { kind: TURN_STOPPED_KIND, reason },
    scopeLabel: turn.scopeLabel,
  });
  return note ?? answer;
}

export function turnToolContext(turn: AgentTurnInput): ToolContextRef {
  return {
    current: turn.tools,
    runtimeRunId: turn.runId,
    runtimeActorId: turn.runtimeActorId,
    pendingApprovals: [],
    pausedOnApproval: false,
    silentRequested: false,
    pollFire: Boolean(turn.pollFire),
    goal: turn.goal ?? null,
    ...(turn.goalMeter ? { goalMeter: turn.goalMeter } : {}),
    ...(turn.continueTurn ? { goalRound: 1 } : {}),
    saveGoal: turn.saveGoal,
    ...(turn.goalInbox ? { goalInbox: turn.goalInbox } : {}),
    emit: turn.emit,
    scopeLabel: turn.scopeLabel,
    orgScopeId: turn.orgScopeId,
    screenToolResult: turn.screenToolResult,
    governGoal: turn.governGoal,
    goalRecentWork: () => goalWorkFromEntries(turn.history.slice(-40)),
    toolApprovalGate: turn.toolApprovalGate,
    shutdown: turn.shutdown,
  };
}

export function turnToolOptions(opts: ToolPlumbing, turn?: AgentTurnInput): AgentToolsOptions {
  return {
    reachExec: opts.reachExec,
    ...(turn?.mcpTools?.length ? { mcpTools: turn.mcpTools } : {}),
    controlTools: opts.controlTools,
    sandboxResources: opts.sandboxResources,
    execTimeoutMs: opts.execTimeoutMs,
    execTimeoutCeilingMs: opts.execTimeoutCeilingMs,
    backgroundJobTtlMs: opts.backgroundJobTtlMs,
    backgroundJobTtlMaxMs: opts.backgroundJobTtlMaxMs,
    sandboxCapabilityTtlMs: opts.sandboxCapabilityTtlMs,
    ...(turn
      ? {
          readOnly: turn.readOnly,
          sessionTools: Boolean(turn.tools.sessionSyscalls),
          surfaceTools: turn.surfaceTools,
          delegateWork: turn.delegateWork,
          surfaceName: turn.surfaceName,
          ...(turn.clientTools?.length ? { clientTools: turn.clientTools } : {}),
        }
      : { surfaceTools: true, surfaceName: "slack" }),
  };
}

export function bridgedTools(ref: ToolContextRef, options: AgentToolsOptions): BridgedTool[] {
  return createAgentTools(ref, options) as unknown as BridgedTool[];
}

export interface ResumedToolResult {
  history: SessionEntry[];
  message: {
    role: "toolResult";
    toolCallId: string;
    toolName: string;
    content: NonNullable<Awaited<ReturnType<BridgedTool["execute"]>>["content"]>;
    isError: boolean;
    timestamp: number;
  };
}

export async function resumeInterruptedToolCall(
  turn: AgentTurnInput,
  ref: ToolContextRef,
  tools: readonly BridgedTool[],
): Promise<ResumedToolResult | null> {
  const call = turn.resumeToolCall;
  if (!call) return null;
  const tool = tools.find((candidate) => candidate.name === call.tool);
  const emit = ref.emit;
  const abortSignal = ref.abortSignal;
  let recorded: SessionEntry | undefined;
  ref.emit = async (entry) => {
    if (entry.type === "tool_call") return;
    const appended = await emit?.(entry);
    if (entry.type === "tool_result" && !recorded) recorded = appended as SessionEntry;
    return appended;
  };
  ref.abortSignal = turn.cancel;
  let content: ResumedToolResult["message"]["content"] = [{ type: "text", text: INTERRUPTED_TOOL_RESULT }];
  let isError = true;
  try {
    if (tool && !turn.cancel?.aborted) {
      content = (await tool.execute(call.callId, call.input)).content ?? [];
      isError = false;
    } else {
      console.error(
        `[agent] resume: ${tool ? "turn cancelled" : `tool ${call.tool} unavailable on this turn`}; recording call ${call.callId} as interrupted`,
      );
    }
    if (!recorded) {
      await ref.emit({
        type: "tool_result",
        payload: { tool: call.tool, callId: call.callId, isError, result: content[0]?.text ?? "", interrupted: true },
        scopeLabel: turn.scopeLabel,
      });
    }
  } catch (error) {
    const payload = recorded?.payload as { result?: unknown; isError?: unknown } | undefined;
    isError = payload ? payload.isError === true : true;
    content = [{ type: "text", text: payload ? String(payload.result ?? "") : `[error] ${errMessage(error)}` }];
    if (!recorded)
      await ref.emit({
        type: "tool_result",
        payload: { tool: call.tool, callId: call.callId, isError, result: content[0]!.text },
        scopeLabel: turn.scopeLabel,
      });
  } finally {
    ref.emit = emit;
    ref.abortSignal = abortSignal;
  }
  if (!recorded) return null;
  if (!isError)
    console.log(`[agent] resume: re-ran retry-safe ${call.tool} call ${call.callId} as seq ${recorded.seq}`);
  return {
    history: [...turn.history, recorded],
    message: {
      role: "toolResult",
      toolCallId: call.callId,
      toolName: call.tool,
      content,
      isError,
      timestamp: recorded.createdAt,
    },
  };
}

export interface ApprovedToolCallOutcome {
  entries: SessionEntry[];
  text: string;
  isError: boolean;
}

const APPROVED_RESULT_NOTE_MAX_CHARS = 4_000;

function approvedCallRoute(
  call: ApprovedToolCall,
  tools: readonly BridgedTool[],
): { tool: BridgedTool; args: unknown } | null {
  const direct = tools.find((candidate) => candidate.name === call.tool);
  if (direct) return { tool: direct, args: call.input };
  const connector = tools.find((candidate) => candidate.name === "mcp_call");
  return connector ? { tool: connector, args: { tool: call.tool, args: call.input } } : null;
}

export async function runApprovedToolCall(
  call: ApprovedToolCall,
  ref: ToolContextRef,
  tools: readonly BridgedTool[],
  opts: { scopeLabel: ScopeId; cancel?: AbortSignal },
): Promise<ApprovedToolCallOutcome> {
  const emit = ref.emit;
  const entries: SessionEntry[] = [];
  let called = false;
  let resulted = false;
  const tag = { approvedRequestId: call.requestId };
  const recordCall = async (): Promise<void> => {
    if (called) return;
    called = true;
    const appended = await emit?.({
      type: "tool_call",
      payload: { tool: call.tool, callId: call.callId, ...tag },
      scopeLabel: opts.scopeLabel,
    });
    if (appended) entries.push(appended as SessionEntry);
  };
  ref.emit = async (entry) => {
    const payload = (entry.payload ?? {}) as Record<string, unknown>;
    if (payload.callId !== call.callId) return emit?.(entry);
    if (entry.type === "tool_call") {
      if (called) return undefined;
      called = true;
      const appended = await emit?.({ ...entry, payload: { ...payload, ...tag } });
      if (appended) entries.push(appended as SessionEntry);
      return appended;
    }
    if (entry.type === "tool_result") {
      await recordCall();
      resulted = true;
      const appended = await emit?.({ ...entry, payload: { ...payload, ...tag } });
      if (appended) entries.push(appended as SessionEntry);
      return appended;
    }
    return emit?.(entry);
  };
  let text = "";
  let isError = true;
  try {
    const route = approvedCallRoute(call, tools);
    if (!route) text = `[error] the approved tool ${call.tool} is not available on this turn, so it did not run`;
    else if (opts.cancel?.aborted) text = "[error] the turn was cancelled before the approved call ran";
    else {
      const result = await route.tool.execute(call.callId, route.args);
      text = (result.content ?? []).flatMap((block) => (typeof block.text === "string" ? [block.text] : [])).join("\n");
      isError = false;
    }
  } catch (error) {
    text = `[error] ${errMessage(error)}`;
  } finally {
    ref.emit = emit;
  }
  const recordedResult = entries.find((e) => e.type === "tool_result");
  if (recordedResult) isError = (recordedResult.payload as { isError?: unknown } | null)?.isError === true || isError;
  if (!resulted) {
    await recordCall();
    const appended = await emit?.({
      type: "tool_result",
      payload: { tool: call.tool, callId: call.callId, isError, result: text, ...tag },
      scopeLabel: opts.scopeLabel,
    });
    if (appended) entries.push(appended as SessionEntry);
  }
  return { entries, text, isError };
}

export function approvedCallSteerNote(
  outcome: { summary: string; approver: string },
  ran: ApprovedToolCallOutcome,
): string {
  const clipped =
    ran.text.length > APPROVED_RESULT_NOTE_MAX_CHARS
      ? `${ran.text.slice(0, APPROVED_RESULT_NOTE_MAX_CHARS)}… [truncated; the full result is recorded in this conversation]`
      : ran.text;
  return (
    `[system] ${outcome.approver} approved the pending call ${outcome.summary}, and QM ran it exactly as approved` +
    `${ran.isError ? "; it returned an error" : ""}. Result:\n${clipped}\n` +
    "Do not call it again; continue with this result."
  );
}

export function deniedCallSteerNote(outcome: { summary: string; approver: string }): string {
  return `[system] ${outcome.approver} denied the pending call ${outcome.summary}, so it did not run. Do not retry it unless they ask again.`;
}

export type OneShotRunner = (
  systemPrompt: string,
  prompt: string,
  signal?: AbortSignal,
  observe?: Pick<AgentTurnInput, "recordModelCall" | "recordLlmRequest">,
  modelOverride?: string,
) => Promise<string | undefined>;

export function oneShotRunner(runPrompt: (turn: AgentTurnInput) => Promise<AgentTurnResult>): OneShotRunner {
  return async (systemPrompt, prompt, signal, observe, modelOverride) => {
    const session = { id: `oneshot-${randomBytes(8).toString("hex")}` } as AgentTurnInput["session"];
    const scope = { kind: "org", id: "oneshot" } as unknown as ScopeId;
    const emitted: SessionEntry[] = [];
    const result = await runPrompt({
      session,
      input: prompt,
      systemPrompt,
      history: [],
      tools: {} as AgentTurnInput["tools"],
      scopeLabel: scope,
      orgScopeId: scope,
      ...(signal ? { cancel: signal } : {}),
      ...(modelOverride ? { runtime: { modelId: modelOverride } } : {}),
      readOnly: true,
      emit: async (entry) => {
        const saved = {
          ...entry,
          sessionId: session.id,
          seq: emitted.length + 1,
          createdAt: Date.now(),
        } as SessionEntry;
        emitted.push(saved);
        return saved;
      },
      recordModelCall: observe?.recordModelCall ?? (() => {}),
      ...(observe?.recordLlmRequest
        ? {
            recordLlmRequest: (rec: TurnLlmRequestRecord, requestSignal?: AbortSignal) =>
              observe.recordLlmRequest!({ ...rec, turnSeq: null }, requestSignal),
          }
        : {}),
    });
    return result.reply || undefined;
  };
}

export function oneShotModelUtilities(
  single: OneShotRunner,
  judgeModelId?: string,
): Pick<ModelUtilities, "oneShot" | "judge" | "screenSecurity" | "generateTitle" | "summarizeApproval"> {
  return {
    oneShot: (system, prompt) => single(system, prompt),
    judge: (system, prompt, signal) => single(system, prompt, signal, undefined, judgeModelId),
    screenSecurity: async ({ payload, signal, recordModelCall, recordLlmRequest }) =>
      parseSecurityScreenVerdict(
        await single(SECURITY_SCREEN_SYSTEM_PROMPT, payload, signal, {
          recordModelCall,
          ...(recordLlmRequest
            ? {
                recordLlmRequest: (rec, requestSignal) =>
                  recordLlmRequest({ ...rec, step: SECURITY_SCREEN_STEP }, requestSignal),
              }
            : {}),
        }),
      ),
    generateTitle: async (transcript) =>
      sanitizeTitle(await single(TITLE_GENERATION_PROMPT, titleUserPrompt(transcript))),
    summarizeApproval: (command, reason, purpose) =>
      single(
        "Explain this command in one plain-English sentence for an approver.",
        [command, reason, purpose].filter(Boolean).join("\n"),
      ),
  };
}
