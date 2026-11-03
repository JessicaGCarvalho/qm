import { TURN_STOPPED_KIND, type ReplayImageRef } from "./replay.ts";
import { randomBytes } from "node:crypto";
import {
  parseSecurityScreenVerdict,
  SECURITY_SCREEN_STEP,
  SECURITY_SCREEN_SYSTEM_PROMPT,
} from "../security/security-posture.ts";
import type { ScopeId, SessionEntry } from "../types.ts";
import type { ResumableToolCall } from "../core/turn-resume.ts";
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

type ToolResultContent = NonNullable<Awaited<ReturnType<BridgedTool["execute"]>>["content"]>;

export interface ResumedToolResult {
  history: SessionEntry[];
  message: {
    role: "toolResult";
    toolCallId: string;
    toolName: string;
    content: ToolResultContent;
    isError: boolean;
    timestamp: number;
  };
}

export interface ApprovedToolCallOutcome {
  entries: SessionEntry[];
  text: string;
  isError: boolean;
}

function toolCallRoute(
  call: ResumableToolCall,
  tools: readonly BridgedTool[],
  viaConnector: boolean,
): { tool: BridgedTool; args: unknown } | null {
  const direct = tools.find((candidate) => candidate.name === call.tool);
  if (direct) return { tool: direct, args: call.input };
  if (!viaConnector) return null;
  const connector = tools.find((candidate) => candidate.name === "mcp_call");
  return connector ? { tool: connector, args: { tool: call.tool, args: call.input } } : null;
}

async function runToolCallOnce(
  call: ResumableToolCall,
  ref: ToolContextRef,
  tools: readonly BridgedTool[],
  opts: {
    scopeLabel: ScopeId;
    cancel?: AbortSignal;
    fresh?: Record<string, unknown>;
    notRun: (reason: "unavailable" | "cancelled") => { text: string; payload?: Record<string, unknown> };
  },
): Promise<ApprovedToolCallOutcome & { result?: SessionEntry; content: ToolResultContent }> {
  const emit = ref.emit;
  const abortSignal = ref.abortSignal;
  const tag = opts.fresh ?? {};
  const entries: SessionEntry[] = [];
  let result: SessionEntry | undefined;
  const append = async (entry: Parameters<NonNullable<typeof emit>>[0]) => {
    const appended = (await emit?.(entry)) as SessionEntry | undefined;
    if (appended) entries.push(appended);
    return appended;
  };
  const recordCall = async () => {
    if (opts.fresh && !entries.some((e) => e.type === "tool_call"))
      await append({
        type: "tool_call",
        payload: { tool: call.tool, callId: call.callId, ...tag },
        scopeLabel: opts.scopeLabel,
      });
  };
  const recordResult = async (payload: Record<string, unknown>) => {
    await recordCall();
    result = await append({
      type: "tool_result",
      payload: { tool: call.tool, callId: call.callId, ...payload, ...tag },
      scopeLabel: opts.scopeLabel,
    });
  };
  ref.emit = async (entry) => {
    const payload = (entry.payload ?? {}) as Record<string, unknown>;
    if (entry.type === "tool_call") {
      if (!opts.fresh) return undefined;
      if (payload.callId !== call.callId || entries.some((e) => e.type === "tool_call")) return emit?.(entry);
      return append({ ...entry, payload: { ...payload, ...tag } });
    }
    if (entry.type !== "tool_result" || result || (opts.fresh && payload.callId !== call.callId)) return emit?.(entry);
    await recordCall();
    result = await append({ ...entry, payload: { ...payload, ...tag } });
    return result;
  };
  ref.abortSignal = opts.cancel;
  let text: string;
  let content: ToolResultContent;
  let isError = true;
  try {
    const route = toolCallRoute(call, tools, !!opts.fresh);
    if (!route || opts.cancel?.aborted) {
      const reason = route ? "cancelled" : "unavailable";
      console.error(
        `[agent] ${opts.fresh ? "approved call" : "resume"}: ${call.tool} ${reason}; call ${call.callId} did not run`,
      );
      const notRun = opts.notRun(reason);
      text = notRun.text;
      content = [{ type: "text", text }];
      if (!result) await recordResult({ isError: true, result: text, ...notRun.payload });
    } else {
      content = (await route.tool.execute(call.callId, route.args)).content ?? [];
      text = content.flatMap((block) => (typeof block.text === "string" ? [block.text] : [])).join("\n");
      isError = false;
      if (!result) await recordResult({ isError, result: text });
    }
  } catch (error) {
    const recorded = result?.payload as { result?: unknown; isError?: unknown } | undefined;
    isError = recorded ? recorded.isError === true : true;
    text = recorded ? String(recorded.result ?? "") : `[error] ${errMessage(error)}`;
    content = [{ type: "text", text }];
    if (!result) await recordResult({ isError, result: text });
  } finally {
    ref.emit = emit;
    ref.abortSignal = abortSignal;
  }
  if (result && (result.payload as { isError?: unknown } | null)?.isError === true) isError = true;
  return { entries, text, content, isError, ...(result ? { result } : {}) };
}

export async function resumeInterruptedToolCall(
  turn: AgentTurnInput,
  ref: ToolContextRef,
  tools: readonly BridgedTool[],
): Promise<ResumedToolResult | null> {
  const call = turn.resumeToolCall;
  if (!call) return null;
  const ran = await runToolCallOnce(call, ref, tools, {
    scopeLabel: turn.scopeLabel,
    ...(turn.cancel ? { cancel: turn.cancel } : {}),
    notRun: () => ({ text: INTERRUPTED_TOOL_RESULT, payload: { interrupted: true } }),
  });
  if (!ran.result) return null;
  if (!ran.isError)
    console.log(`[agent] resume: re-ran retry-safe ${call.tool} call ${call.callId} as seq ${ran.result.seq}`);
  return {
    history: [...turn.history, ran.result],
    message: {
      role: "toolResult",
      toolCallId: call.callId,
      toolName: call.tool,
      content: ran.content,
      isError: ran.isError,
      timestamp: ran.result.createdAt,
    },
  };
}

export async function runApprovedToolCall(
  call: ApprovedToolCall,
  ref: ToolContextRef,
  tools: readonly BridgedTool[],
  opts: { scopeLabel: ScopeId; cancel?: AbortSignal },
): Promise<ApprovedToolCallOutcome> {
  const { entries, text, isError } = await runToolCallOnce(call, ref, tools, {
    ...opts,
    fresh: { approvedRequestId: call.requestId },
    notRun: (reason) => ({
      text:
        reason === "unavailable"
          ? `[error] the approved tool ${call.tool} is not available on this turn, so it did not run`
          : "[error] the turn was cancelled before the approved call ran",
    }),
  });
  return { entries, text, isError };
}

export function approvedCallSteerNote(
  outcome: { summary: string; approver: string },
  ran: ApprovedToolCallOutcome,
  withResult: boolean,
): string {
  const head =
    `[system] ${outcome.approver} approved the pending call ${outcome.summary}, and QM ran it exactly as approved` +
    (ran.isError ? "; it returned an error" : "");
  return withResult
    ? `${head}. Result:\n${ran.text}\nDo not call it again; continue with this result.`
    : `${head}. Its result is recorded in this conversation. Do not call it again.`;
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
