/**
 * PlannerExecutorAgent: Two-tier agent architecture for browser automation.
 *
 * MVP implementation with:
 * - Stepwise (ReAct-style) planning
 * - Compact context formatting for small models
 * - Snapshot escalation for reliable element capture
 * - Pre-step verification
 * - Retry/repair logic
 * - Token usage tracking
 * - Pre-action authorization hook (for sidecar policy integration)
 *
 * Reliability integrations:
 * - Vision fallback detection and vision-capable executor routing
 * - Modal/overlay dismissal and checkout continuation
 * - Intent heuristics
 * - Recovery navigation checkpoints
 *
 * Still deferred:
 * - Captcha handling
 */

import type { LLMProvider, LLMResponse } from '../../llm-provider';
import type { PlannerExecutorConfig } from './config';
import { mergeConfig, type DeepPartial } from './config';
import type {
  ActionRecord,
  RepairFailureCategory,
  RepairHistoryEntry,
  StepOutcome,
  RunOutcome,
  TokenUsageSummary,
  TokenUsageTotals,
  SnapshotContext,
  ParsedAction,
  Snapshot,
  SnapshotElement,
  SerializedAgentState,
  SerializedRecoveryState,
} from './plan-models';
import { StepStatus, ReplanPatchSchema } from './plan-models';
import {
  buildStepwisePlannerPrompt,
  buildExecutorPrompt,
  buildVisionExecutorPrompt,
  type StepwisePlannerResponse,
} from './prompts';
import { buildRepairPlannerPrompt } from './replan-prompts';
import {
  parseAction,
  extractJson,
  normalizePlan,
  normalizeReplanPatch,
  formatContext,
  selectContextElements,
} from './plan-utils';
import { evaluatePredicates } from './predicates';
import { detectSnapshotFailure } from './vision-fallback';
import { RecoveryState, verifyRecoveryCheckpoint } from './recovery';
import {
  DEFAULT_CHECKOUT_CONFIG,
  detectAuthBoundary,
  isCheckoutElement,
  isSearchLikeTypeAndSubmit,
  isUrlChangeRelevantToIntent,
} from './boundary-detection';
import { ComposableHeuristics } from './composable-heuristics';
import { normalizeTaskCategory, TaskCategory } from './task-category';
import type { ResolvedAgentProfile } from './profile-types';
import { HeuristicHint } from './heuristic-hint';
import {
  detectModalAppearance,
  detectModalDismissed,
  findDismissalTarget,
  shouldAutoContinueCheckoutFlow,
} from './modal-dismissal';
import { detectPruningCategory } from './category-pruner';
import { pruneWithRecovery, fullSnapshotContainsIntent } from './pruning-recovery';
import type { Tracer } from '../../tracing/tracer';
import {
  isTextExtractionTask,
  isExtractionTask,
  buildExtractionPrompt,
} from './extraction-keywords';

// ---------------------------------------------------------------------------
// Token Usage Collector
// ---------------------------------------------------------------------------

/**
 * Collects token usage statistics by role (planner/executor) and model.
 */
class TokenUsageCollector {
  private byRole: Map<string, TokenUsageTotals> = new Map();
  private byModel: Map<string, TokenUsageTotals> = new Map();

  record(role: string, resp: LLMResponse): void {
    // By role
    const roleTotals = this.byRole.get(role) || {
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    roleTotals.calls += 1;
    const pt = typeof resp.promptTokens === 'number' ? Math.max(0, resp.promptTokens) : 0;
    const ct = typeof resp.completionTokens === 'number' ? Math.max(0, resp.completionTokens) : 0;
    const tt = typeof resp.totalTokens === 'number' ? Math.max(0, resp.totalTokens) : pt + ct;
    roleTotals.promptTokens += pt;
    roleTotals.completionTokens += ct;
    roleTotals.totalTokens += tt;
    this.byRole.set(role, roleTotals);

    // By model
    const modelName = (resp.modelName || '').trim() || 'unknown';
    const modelTotals = this.byModel.get(modelName) || {
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    modelTotals.calls += 1;
    modelTotals.promptTokens += pt;
    modelTotals.completionTokens += ct;
    modelTotals.totalTokens += tt;
    this.byModel.set(modelName, modelTotals);
  }

  reset(): void {
    this.byRole.clear();
    this.byModel.clear();
  }

  summary(): TokenUsageSummary {
    // Sum totals
    const total: TokenUsageTotals = {
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    for (const t of this.byRole.values()) {
      total.calls += t.calls;
      total.promptTokens += t.promptTokens;
      total.completionTokens += t.completionTokens;
      total.totalTokens += t.totalTokens;
    }

    // Convert maps to records
    const byRole: Record<string, TokenUsageTotals> = {};
    for (const [k, v] of this.byRole) {
      byRole[k] = { ...v };
    }
    const byModel: Record<string, TokenUsageTotals> = {};
    for (const [k, v] of this.byModel) {
      byModel[k] = { ...v };
    }

    return { total, byRole, byModel };
  }
}

// ---------------------------------------------------------------------------
// Pre-Action Authorizer Interface
// ---------------------------------------------------------------------------

/**
 * Authorization result from pre-action check.
 */
export interface AuthorizationResult {
  /** Whether the action is allowed */
  allowed: boolean;
  /** Reason for denial (if not allowed) */
  reason?: string;
  /** Alternative action to take (if any) */
  alternative?: string;
}

/**
 * Hook for pre-action authorization (e.g., sidecar policy evaluation).
 *
 * Called before each action is executed. If the action is denied,
 * the agent will skip the action and record the denial.
 *
 * @example
 * ```typescript
 * const authorizer: PreActionAuthorizer = async (action, context) => {
 *   const response = await fetch('http://localhost:3500/v1/authorize', {
 *     method: 'POST',
 *     body: JSON.stringify({
 *       principal: 'agent:browser-automation',
 *       action: `browser.${action.type.toLowerCase()}`,
 *       resource: context.url,
 *     }),
 *   });
 *   const result = await response.json();
 *   return { allowed: result.decision === 'ALLOW', reason: result.reason };
 * };
 * ```
 */
export type PreActionAuthorizer = (
  action: { type: string; elementId?: number; value?: string },
  context: { url: string; stepGoal: string; taskGoal: string }
) => Promise<AuthorizationResult>;

// ---------------------------------------------------------------------------
// Intent Heuristics Interface
// ---------------------------------------------------------------------------

/**
 * Interface for intent-based element selection.
 *
 * Allows bypassing LLM for known patterns (e.g., "add to cart" buttons).
 * Can be implemented by domain-specific heuristics.
 */
export interface IntentHeuristics {
  /**
   * Find element matching the given intent.
   *
   * @param intent - Intent string (e.g., "add_to_cart", "search")
   * @param elements - Available elements from snapshot
   * @param url - Current page URL
   * @param goal - Step goal description
   * @returns Element ID if found, null otherwise
   */
  findElementForIntent(
    intent: string,
    elements: SnapshotElement[],
    url: string,
    goal: string
  ): number | null;

  /**
   * Get priority order for intent matching.
   * Higher priority intents are tried first.
   */
  priorityOrder(): string[];
}

/**
 * Common text patterns for intent matching.
 * Used as fallback when no custom heuristics are provided.
 */
const COMMON_INTENT_PATTERNS: Record<string, string[]> = {
  add_to_cart: ['add to cart', 'add to bag', 'add to basket', 'buy now', 'add item'],
  checkout: ['checkout', 'proceed to checkout', 'go to checkout', 'check out'],
  search: ['search', 'find', 'go', 'submit'],
  login: ['log in', 'login', 'sign in', 'signin'],
  submit: ['submit', 'send', 'continue', 'next', 'confirm'],
  close: ['close', 'dismiss', 'x', 'cancel', 'no thanks'],
};

/**
 * Action verbs to strip from descriptive intents.
 * E.g., "click the Add to Cart button" → "add to cart button"
 */
const ACTION_VERBS = [
  'click',
  'tap',
  'press',
  'select',
  'choose',
  'pick',
  'find',
  'locate',
  'look for',
  'search for',
  'type',
  'enter',
  'input',
  'fill',
  'scroll to',
  'navigate to',
  'go to',
  'open',
  'close',
  'dismiss',
  'accept',
  'the',
  'a',
  'an',
  'on',
  'button',
  'link',
  'field',
  'input',
  'element',
];

/**
 * Extract meaningful keywords from a descriptive intent.
 * Handles cases like:
 * - "click the Add to Cart button" → "add to cart"
 * - 'click "Add to Cart"' → "add to cart"
 * - "Add to Cart" → "add to cart"
 */
function extractIntentKeywords(intent: string): string[] {
  let normalized = intent.toLowerCase().trim();

  // Extract quoted text first (e.g., 'click "Add to Cart"')
  const quotedMatch = normalized.match(/["']([^"']+)["']/);
  if (quotedMatch) {
    return [quotedMatch[1].trim()];
  }

  // Strip action verbs from the beginning
  for (const verb of ACTION_VERBS) {
    const pattern = new RegExp(`^${verb}\\s+`, 'i');
    normalized = normalized.replace(pattern, '');
  }

  // Strip trailing words like "button", "link", "element"
  normalized = normalized.replace(/\s+(button|link|element|field|input)$/i, '');

  // Clean up extra spaces
  normalized = normalized.replace(/\s+/g, ' ').trim();

  // If the result is too short, return the original
  if (normalized.length < 2) {
    return [intent.toLowerCase().replace(/[_-]/g, ' ')];
  }

  return [normalized];
}

/**
 * Simple intent heuristics using text pattern matching.
 * Used as default when no custom heuristics are provided.
 *
 * This is GENERALIZABLE - it works for any intent by:
 * 1. Checking common patterns (add_to_cart, checkout, etc.)
 * 2. Extracting keywords from descriptive intents (e.g., "click the X button")
 * 3. Falling back to direct text matching
 */
class SimpleIntentHeuristics implements IntentHeuristics {
  findElementForIntent(
    intent: string,
    elements: SnapshotElement[],
    _url: string,
    _goal: string
  ): number | null {
    const normalizedIntent = intent.toLowerCase().replace(/[_-]/g, ' ');

    // Build patterns list - start with common patterns, then add extracted keywords
    const patterns: string[] = [];

    // Check if this is a known intent pattern
    if (COMMON_INTENT_PATTERNS[intent.toLowerCase()]) {
      patterns.push(...COMMON_INTENT_PATTERNS[intent.toLowerCase()]);
    } else if (COMMON_INTENT_PATTERNS[normalizedIntent]) {
      patterns.push(...COMMON_INTENT_PATTERNS[normalizedIntent]);
    }

    // Extract keywords from descriptive intent (generalizable)
    const keywords = extractIntentKeywords(intent);
    patterns.push(...keywords);

    // Also add the raw normalized intent as fallback
    if (!patterns.includes(normalizedIntent)) {
      patterns.push(normalizedIntent);
    }

    // Look for elements matching patterns (prefer clickable elements)
    for (const pattern of patterns) {
      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        const ariaLabel = (element.ariaLabel || '').toLowerCase();
        const name = (element.name || '').toLowerCase();

        if (text.includes(pattern) || ariaLabel.includes(pattern) || name.includes(pattern)) {
          // Prefer clickable buttons/links
          if (element.clickable || element.role === 'button' || element.role === 'link') {
            return element.id;
          }
        }
      }
    }

    // Second pass: less strict matching (any element)
    for (const pattern of patterns) {
      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        if (text.includes(pattern)) {
          return element.id;
        }
      }
    }

    // Third pass: word-by-word matching for multi-word intents
    // This handles cases where the element text is slightly different
    const intentWords = normalizedIntent.split(/\s+/).filter(w => w.length > 2);
    if (intentWords.length >= 2) {
      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        const matchCount = intentWords.filter(word => text.includes(word)).length;
        // If more than half the words match, consider it a match
        if (matchCount >= Math.ceil(intentWords.length / 2)) {
          if (element.clickable || element.role === 'button' || element.role === 'link') {
            return element.id;
          }
        }
      }
    }

    return null;
  }

  priorityOrder(): string[] {
    return ['add_to_cart', 'checkout', 'search', 'submit', 'close', 'login'];
  }
}

// ---------------------------------------------------------------------------
// AgentRuntime Interface (minimal for MVP)
// ---------------------------------------------------------------------------

/**
 * Minimal runtime interface for browser control.
 * This will be replaced with the full AgentRuntime integration.
 */
export interface AgentRuntime {
  /** Take a snapshot of the current page */
  snapshot(options?: {
    limit?: number;
    screenshot?: boolean;
    goal?: string;
  }): Promise<Snapshot | null>;

  /** Navigate to a URL */
  goto(url: string): Promise<void>;

  /** Click an element by ID */
  click(elementId: number): Promise<void>;

  /** Click at viewport coordinates (vision fallback) */
  clickCoordinate?(x: number, y: number): Promise<void>;

  /** Type text into an element */
  type(elementId: number, text: string): Promise<void>;

  /** Type text at current position without targeting an element (vision fallback) */
  typeCoordinate?(text: string): Promise<void>;

  /** Press a key */
  pressKey(key: string): Promise<void>;

  /** Scroll the page */
  scroll(direction: 'up' | 'down'): Promise<void>;

  /** Get current URL */
  getCurrentUrl(): Promise<string>;

  /** Get viewport height */
  getViewportHeight(): Promise<number>;

  /** Scroll by delta (returns true if scroll was effective) */
  scrollBy(dy: number): Promise<boolean>;

  /** Read page content as markdown (for EXTRACT actions) */
  readMarkdown?(options?: { maxChars?: number }): Promise<string | null>;
}

const LOGIN_INTENT_RE = /\b(log\s*in|sign\s*in|authenticate|auth\s*flow|credentials)\b/i;

function taskHasLoginIntent(task: string): boolean {
  return LOGIN_INTENT_RE.test(task);
}

// ---------------------------------------------------------------------------
// PlannerExecutorAgent Options
// ---------------------------------------------------------------------------

/**
 * Options for creating a PlannerExecutorAgent.
 */
export interface PlannerExecutorAgentOptions {
  /** LLM for generating plans (recommend 7B+ model) */
  planner: LLMProvider;
  /** LLM for executing steps (3B-7B model) */
  executor: LLMProvider;
  /** Optional tracer for Studio/debug parity events */
  tracer?: Tracer;
  /** Agent configuration (merged with defaults) */
  config?: DeepPartial<PlannerExecutorConfig>;
  /** Pre-action authorization hook */
  preActionAuthorizer?: PreActionAuthorizer;
  /** Custom intent heuristics for element selection */
  intentHeuristics?: IntentHeuristics;
  /** Enable verbose logging */
  verbose?: boolean;
  /** Resolved profile providing data-driven pruning, hints, and scoring overrides */
  resolvedProfile?: ResolvedAgentProfile;
  /** Callback invoked after each step with structured outcome (for learning extraction) */
  onStepOutcome?: (outcome: StepOutcome, snapshotElements?: SnapshotElement[]) => void;
  /** Callback invoked at the start of each step with progress info (step number, token usage) */
  onProgress?: (progress: {
    step: number;
    maxSteps: number;
    totalTokens: number;
    action?: string;
  }) => void;
  /**
   * Pause gate: called at each step boundary.
   * If it returns a Promise, the agent awaits it before proceeding (paused).
   * Return a resolved promise to continue immediately (not paused).
   * This enables pause/resume without AbortSignal.
   */
  pauseGate?: () => Promise<void>;
  /**
   * Checkpoint callback: called after each step with the serialized agent state.
   * Used for persisting task state to survive sidepanel context loss.
   */
  onStepCheckpoint?: (state: SerializedAgentState) => void;
}

// ---------------------------------------------------------------------------
// PlannerExecutorAgent
// ---------------------------------------------------------------------------

/**
 * Two-tier agent architecture with Planner and Executor models.
 *
 * The Planner (typically 7B+ parameters) generates JSON execution plans
 * with predicates. The Executor (3B-7B parameters) executes each step
 * using a snapshot-first approach.
 *
 * @example
 * ```typescript
 * import { PlannerExecutorAgent, OllamaProvider } from '@predicatesystems/runtime';
 *
 * const planner = new OllamaProvider({ model: 'qwen3:8b' });
 * const executor = new OllamaProvider({ model: 'qwen3:4b' });
 *
 * const agent = new PlannerExecutorAgent({
 *   planner,
 *   executor,
 *   config: { stepwise: { maxSteps: 20 }, verbose: true },
 * });
 *
 * const result = await agent.runStepwise(runtime, {
 *   task: 'Search for laptops and add first result to cart',
 *   startUrl: 'https://amazon.com',
 * });
 * ```
 */
export class PlannerExecutorAgent {
  readonly planner: LLMProvider;
  readonly executor: LLMProvider;
  readonly config: PlannerExecutorConfig;

  private tracer?: Tracer;
  private preActionAuthorizer?: PreActionAuthorizer;
  private baseIntentHeuristics: IntentHeuristics;
  private composableHeuristics: ComposableHeuristics;
  private currentTaskCategory: TaskCategory | null = null;
  private tokenCollector = new TokenUsageCollector();
  private recoveryState: RecoveryState | null = null;
  private resolvedProfile?: ResolvedAgentProfile;
  private onStepOutcome?: (outcome: StepOutcome, snapshotElements?: SnapshotElement[]) => void;
  private onProgress?: (progress: {
    step: number;
    maxSteps: number;
    totalTokens: number;
    action?: string;
  }) => void;
  private pauseGate?: () => Promise<void>;
  private onStepCheckpoint?: (state: SerializedAgentState) => void;

  // Run state
  private runId: string | null = null;
  private actionHistory: ActionRecord[] = [];
  private currentStepIndex = 0;
  private currentStep: { action: string; intent?: string } | null = null;

  constructor(options: PlannerExecutorAgentOptions) {
    this.planner = options.planner;
    this.executor = options.executor;
    this.tracer = options.tracer;
    this.config = mergeConfig(options.config || {});
    if (options.verbose !== undefined) {
      this.config = { ...this.config, verbose: options.verbose };
    }
    this.preActionAuthorizer = options.preActionAuthorizer;
    this.baseIntentHeuristics = options.intentHeuristics || new SimpleIntentHeuristics();
    this.composableHeuristics = new ComposableHeuristics({
      staticHeuristics: this.baseIntentHeuristics,
    });
    this.resolvedProfile = options.resolvedProfile;
    this.onStepOutcome = options.onStepOutcome;
    this.onProgress = options.onProgress;
    this.pauseGate = options.pauseGate;
    this.onStepCheckpoint = options.onStepCheckpoint;
  }

  // ---------------------------------------------------------------------------
  // Token Stats
  // ---------------------------------------------------------------------------

  /**
   * Get token usage statistics for the agent session.
   */
  getTokenStats(): TokenUsageSummary {
    return this.tokenCollector.summary();
  }

  /**
   * Reset token usage statistics.
   */
  resetTokenStats(): void {
    this.tokenCollector.reset();
  }

  exportRunState(loopLocals: {
    stepOutcomes: StepOutcome[];
    currentUrl: string;
    success: boolean;
    error: string | undefined;
    fallbackUsed: boolean;
    replansUsed: number;
    queuedRepairSteps: StepwisePlannerResponse[];
    repairHistory: RepairHistoryEntry[];
  }): SerializedAgentState {
    return {
      runId: this.runId,
      actionHistory: [...this.actionHistory],
      currentStepIndex: this.currentStepIndex,
      currentStep: this.currentStep ? { ...this.currentStep } : null,
      currentTaskCategory: this.currentTaskCategory,
      loopState: {
        stepOutcomes: [...loopLocals.stepOutcomes],
        currentUrl: loopLocals.currentUrl,
        success: loopLocals.success,
        error: loopLocals.error,
        fallbackUsed: loopLocals.fallbackUsed,
        replansUsed: loopLocals.replansUsed,
        queuedRepairSteps: loopLocals.queuedRepairSteps.map(s => ({ ...s })),
        repairHistory: [...loopLocals.repairHistory],
      },
      recoveryState: this.recoveryState ? this.serializeRecoveryState(this.recoveryState) : null,
      tokenUsage: this.tokenCollector.summary(),
    };
  }

  importRunState(state: SerializedAgentState): void {
    this.runId = state.runId;
    this.actionHistory = state.actionHistory.map(r => ({ ...r }));
    this.currentStepIndex = state.currentStepIndex;
    this.currentStep = state.currentStep ? { ...state.currentStep } : null;
    this.currentTaskCategory = state.currentTaskCategory as TaskCategory | null;
    if (state.recoveryState) {
      this.recoveryState = this.deserializeRecoveryState(state.recoveryState);
    }
  }

  private serializeRecoveryState(rs: RecoveryState): SerializedRecoveryState {
    const json = rs.toJSON();
    return {
      checkpoints: json.checkpoints,
      recoveryAttemptsUsed: json.recoveryAttemptsUsed,
      maxRecoveryAttempts: rs.maxRecoveryAttempts,
      maxCheckpoints: rs.maxCheckpoints,
    };
  }

  private deserializeRecoveryState(data: SerializedRecoveryState): RecoveryState {
    return RecoveryState.fromJSON(data);
  }

  private recordTokenUsage(role: string, resp: LLMResponse): void {
    try {
      this.tokenCollector.record(role, resp);
    } catch {
      // Don't fail on token tracking errors
    }
  }

  private getTraceStepId(stepIndex: number = this.currentStepIndex): string | undefined {
    if (!this.runId || stepIndex <= 0) {
      return undefined;
    }

    return `${this.runId}:step:${stepIndex}`;
  }

  private emitPlannerAction(
    stepIndex: number,
    plannerAction: StepwisePlannerResponse,
    source: 'planner' | 'repair'
  ): void {
    if (!this.tracer) {
      return;
    }

    const stepId = plannerAction.action === 'DONE' ? undefined : this.getTraceStepId(stepIndex);
    this.tracer.emit(
      'planner_action',
      {
        step_index: stepIndex,
        goal: plannerAction.goal || plannerAction.intent || plannerAction.action,
        action: plannerAction.action,
        details: {
          source,
          intent: plannerAction.intent,
          target: plannerAction.target,
          input: plannerAction.input,
          required: plannerAction.required ?? true,
          verify_count: plannerAction.verify?.length || 0,
          heuristic_hint_count: plannerAction.heuristicHints?.length || 0,
          optional_substep_count: plannerAction.optionalSubsteps?.length || 0,
          reasoning: plannerAction.reasoning,
        },
      },
      stepId
    );
  }

  private emitStepEnd(
    stepId: string,
    stepIndex: number,
    plannerAction: StepwisePlannerResponse,
    outcome: StepOutcome
  ): void {
    if (!this.tracer) {
      return;
    }

    const actionMatch = outcome.actionTaken?.match(/^([A-Z_]+)/);
    const executedAction = actionMatch?.[1] || plannerAction.action;
    const elementMatch = outcome.actionTaken?.match(/^[A-Z_]+\((\d+)/);
    const elementId = elementMatch ? Number(elementMatch[1]) : undefined;
    const execSuccess = outcome.status !== StepStatus.FAILED;

    this.tracer.emit(
      'step_end',
      {
        v: 1,
        step_id: stepId,
        step_index: stepIndex,
        goal: plannerAction.goal || plannerAction.intent || plannerAction.action,
        attempt: 0,
        pre: outcome.urlBefore
          ? {
              url: outcome.urlBefore,
              snapshot_digest: this.makeSnapshotDigest(outcome.urlBefore),
            }
          : undefined,
        llm: {
          response_text:
            outcome.llmResponseText || plannerAction.reasoning || outcome.actionTaken || '',
          model: this.executor.modelName,
        },
        exec: {
          success: execSuccess,
          action: executedAction,
          duration_ms: outcome.durationMs,
          element_id: elementId,
          error: outcome.error,
        },
        post: outcome.urlAfter
          ? {
              url: outcome.urlAfter,
              snapshot_digest: this.makeSnapshotDigest(outcome.urlAfter),
            }
          : undefined,
        verify: {
          passed: outcome.verificationPassed,
          signals: outcome.error ? { error: outcome.error } : {},
        },
      },
      stepId
    );
  }

  private normalizeVisionTraceReason(reason: string | null): string | null {
    if (reason === 'require_vision') {
      return 'page_requires_vision';
    }

    return reason;
  }

  // ---------------------------------------------------------------------------
  // Stepwise Run (ReAct-style)
  // ---------------------------------------------------------------------------

  /**
   * Run task using stepwise (ReAct-style) planning.
   *
   * Plans one step at a time based on current page state, adapting to
   * page changes as they happen. More reliable with small models.
   *
   * @param runtime - Browser runtime for page control
   * @param options - Task options
   * @returns Run outcome
   */
  async runStepwise(
    runtime: AgentRuntime,
    options: {
      task: string;
      startUrl?: string;
      category?: TaskCategory | string | null;
    }
  ): Promise<RunOutcome> {
    const { task, startUrl } = options;
    const startTime = Date.now();

    // Initialize run state
    this.runId = crypto.randomUUID();
    this.actionHistory = [];
    this.currentStepIndex = 0;
    this.tokenCollector.reset();
    this.currentTaskCategory =
      normalizeTaskCategory(options.category) ?? this.resolvedProfile?.categoryHint ?? null;
    this.composableHeuristics = new ComposableHeuristics({
      staticHeuristics: this.baseIntentHeuristics,
      taskCategory: this.currentTaskCategory,
    });
    // Seed heuristics with profile hints (if any)
    if (this.resolvedProfile?.heuristicHints && this.resolvedProfile.heuristicHints.length > 0) {
      this.composableHeuristics.setStepHints(
        this.resolvedProfile.heuristicHints.map(h => new HeuristicHint(h))
      );
    } else {
      this.composableHeuristics.clearStepHints();
    }
    this.recoveryState = this.config.recovery.enabled
      ? new RecoveryState(this.config.recovery)
      : null;

    const stepOutcomes: StepOutcome[] = [];
    let tracedStepCount = 0;
    let currentUrl = '';
    let success = false;
    let error: string | undefined;
    let fallbackUsed = false;
    let replansUsed = 0;
    const queuedRepairSteps: StepwisePlannerResponse[] = [];
    const repairHistory: RepairHistoryEntry[] = [];

    this.tracer?.emitRunStart('PlannerExecutorAgent', this.planner.modelName, {
      task,
      startUrl,
      category: this.currentTaskCategory || undefined,
    });

    try {
      // Navigate to start URL if provided
      if (startUrl) {
        if (this.config.verbose) {
          console.log(`[NAVIGATE] ${startUrl}`);
        }
        await runtime.goto(startUrl);
        currentUrl = startUrl;
      } else {
        currentUrl = await runtime.getCurrentUrl();
      }

      // Stepwise loop
      const maxSteps = this.config.stepwise.maxSteps;

      for (let stepNum = 1; stepNum <= maxSteps; stepNum++) {
        this.currentStepIndex = stepNum;
        this.composableHeuristics.clearStepHints();
        const stepStart = Date.now();

        if (this.config.verbose) {
          console.log(`\n${'='.repeat(60)}`);
          console.log(`[STEP ${stepNum}/${maxSteps}]`);
          console.log(`${'='.repeat(60)}`);
        }

        // Emit progress callback
        try {
          const tokenSummary = this.tokenCollector.summary();
          this.onProgress?.({
            step: stepNum,
            maxSteps,
            totalTokens: tokenSummary.total.totalTokens,
          });
        } catch {
          // Don't fail the run on progress callback errors
        }

        // Pause gate: await if paused, continue immediately if not
        if (this.pauseGate) {
          await this.pauseGate();
        }

        // Take snapshot with escalation
        const ctx = await this.snapshotWithEscalation(runtime, task);
        currentUrl = ctx.snapshot?.url || currentUrl;
        fallbackUsed = fallbackUsed || ctx.requiresVision;

        if (this.config.authBoundary.enabled) {
          const authResult = detectAuthBoundary(currentUrl, this.config.authBoundary);
          if (
            authResult.isAuthBoundary &&
            this.config.authBoundary.stopOnAuth &&
            !taskHasLoginIntent(task)
          ) {
            success = true;
            error = this.config.authBoundary.authSuccessMessage;
            break;
          }
        }

        if (this.config.verbose) {
          const elementCount = ctx.snapshot?.elements?.length || 0;
          console.log(`[SNAPSHOT] ${elementCount} elements, limit=${ctx.limitUsed}`);
          // Debug: show searchbox/textbox elements specifically
          const allElements = ctx.snapshot?.elements || [];
          const searchElements = allElements.filter(el =>
            ['textbox', 'searchbox', 'combobox', 'input'].includes((el.role || '').toLowerCase())
          );
          console.log(
            `  [SEARCH ELEMENTS] Found ${searchElements.length} textbox/searchbox/combobox/input elements:`
          );
          for (const el of searchElements.slice(0, 3)) {
            console.log(
              `    [EL ${el.id}] role=${el.role}, text=${(el.text || '').slice(0, 60).replace(/\n/g, ' ')}, clickable=${el.clickable}`
            );
          }
          if (searchElements.length === 0) {
            // Show first 5 elements for debugging
            console.log(`  [FIRST 5 ELEMENTS]:`);
            for (const el of allElements.slice(0, 5)) {
              console.log(
                `    [EL ${el.id}] role=${el.role}, text=${(el.text || '').slice(0, 40).replace(/\n/g, ' ')}`
              );
            }
          }
        }

        let plannerAction: StepwisePlannerResponse | null = null;
        let plannerActionSource: 'planner' | 'repair' = 'planner';
        if (queuedRepairSteps.length > 0) {
          plannerAction = queuedRepairSteps.shift()!;
          plannerActionSource = 'repair';
        } else {
          // Get planner's next action
          const [systemPrompt, userPrompt] = buildStepwisePlannerPrompt(
            task,
            currentUrl,
            ctx.compactRepresentation,
            this.actionHistory.slice(-this.config.stepwise.actionHistoryLimit)
          );

          if (this.config.verbose) {
            // Show the compact representation being sent to the LLM
            const contextLines = ctx.compactRepresentation.split('\n');
            console.log(`[PLANNER PROMPT] Sending ${contextLines.length} element lines to LLM:`);
            // Show header and first few elements
            for (const line of contextLines.slice(0, 6)) {
              console.log(`  ${line}`);
            }
            if (contextLines.length > 6) {
              console.log(`  ... (${contextLines.length - 6} more elements)`);
            }
          }

          let plannerResp: LLMResponse | null = null;
          try {
            plannerResp = await this.callPlannerWithRetry(systemPrompt, userPrompt, {
              temperature: this.config.plannerTemperature,
              max_tokens: this.config.plannerMaxTokens,
            });
            this.recordTokenUsage('planner', plannerResp);
          } catch (plannerError) {
            const plannerErrorMessage =
              plannerError instanceof Error ? plannerError.message : String(plannerError);
            const bootstrapAction = this.buildPlannerTimeoutBootstrapAction(
              stepNum,
              task,
              ctx,
              plannerErrorMessage
            );
            if (bootstrapAction) {
              plannerAction = bootstrapAction;
              plannerActionSource = 'repair';
              if (this.config.verbose) {
                console.log(
                  `[PLANNER FALLBACK] Executing deterministic bootstrap action after planner timeout: ${JSON.stringify(bootstrapAction)}`
                );
              }
              // Skip normal planner response parsing and continue with synthesized action.
            } else {
              // Log planner call failure
              if (this.config.verbose) {
                console.log(`[PLANNER ERROR] LLM call failed: ${plannerError}`);
              }
              stepOutcomes.push({
                stepId: stepNum,
                goal: 'Call planner LLM',
                status: StepStatus.FAILED,
                verificationPassed: false,
                usedVision: false,
                durationMs: Date.now() - stepStart,
                error: `Planner LLM call failed: ${plannerErrorMessage}`,
              });
              if (this.shouldAbortOnPlannerFailure(plannerErrorMessage)) {
                error = `Planner unavailable: ${plannerErrorMessage}`;
                break;
              }
              continue;
            }
          }

          if (!plannerActionSource || plannerActionSource === 'planner') {
            if (!plannerResp) {
              stepOutcomes.push({
                stepId: stepNum,
                goal: 'Call planner LLM',
                status: StepStatus.FAILED,
                verificationPassed: false,
                usedVision: false,
                durationMs: Date.now() - stepStart,
                error: 'Planner response missing after planner action path',
              });
              continue;
            }
            // Log planner call failure
            if (this.config.verbose) {
              // Show raw response for debugging (truncated if very long)
              const rawLen = plannerResp.content.length;
              const hasThink = plannerResp.content.includes('<think>');
              const displayContent =
                rawLen > 300
                  ? plannerResp.content.slice(0, 300) + `... (${rawLen} chars)`
                  : plannerResp.content;
              console.log(`[PLANNER]${hasThink ? ' (has <think>)' : ''} ${displayContent}`);
            }

            // Check for empty response
            if (!plannerResp.content || plannerResp.content.trim().length === 0) {
              if (this.config.verbose) {
                console.log(`[PLANNER ERROR] Empty response from LLM`);
              }
              stepOutcomes.push({
                stepId: stepNum,
                goal: 'Parse planner response',
                status: StepStatus.FAILED,
                verificationPassed: false,
                usedVision: false,
                durationMs: Date.now() - stepStart,
                error: 'Planner returned empty response',
              });
              continue;
            }

            // Parse planner response
            try {
              plannerAction = this.normalizePlannerAction(extractJson(plannerResp.content));
            } catch (e) {
              // Try to recover from malformed JSON
              const parsed = parseAction(plannerResp.content);
              if (parsed.action !== 'UNKNOWN') {
                plannerAction = {
                  action: parsed.action as StepwisePlannerResponse['action'],
                  input: parsed.args[1] as string | undefined,
                };
              } else {
                if (this.config.verbose) {
                  console.log(`[PLANNER ERROR] Raw response: ${plannerResp.content.slice(0, 200)}`);
                }
                stepOutcomes.push({
                  stepId: stepNum,
                  goal: 'Parse planner response',
                  status: StepStatus.FAILED,
                  verificationPassed: false,
                  usedVision: false,
                  durationMs: Date.now() - stepStart,
                  error: `Failed to parse planner response: ${e}`,
                });
                continue;
              }
            }
          }
        }

        if (!plannerAction) {
          stepOutcomes.push({
            stepId: stepNum,
            goal: 'Parse planner response',
            status: StepStatus.FAILED,
            verificationPassed: false,
            usedVision: false,
            durationMs: Date.now() - stepStart,
            error: 'Planner action missing after planner/repair resolution',
          });
          continue;
        }

        plannerAction = this.promoteVisibleResultClick(task, ctx, plannerAction);
        this.composableHeuristics.setStepHints(plannerAction.heuristicHints || []);
        this.emitPlannerAction(stepNum, plannerAction, plannerActionSource);

        // Handle DONE action
        if (plannerAction.action === 'DONE') {
          if (this.config.verbose) {
            console.log(`[DONE] Task completed`);
          }
          stepOutcomes.push({
            stepId: stepNum,
            goal: 'Task completed',
            status: StepStatus.SUCCESS,
            actionTaken: 'DONE',
            verificationPassed: true,
            usedVision: false,
            durationMs: Date.now() - stepStart,
          });
          success = true;
          break;
        }

        // Execute the action
        const stepTraceId = this.getTraceStepId(stepNum)!;
        tracedStepCount += 1;
        this.tracer?.emitStepStart(
          stepTraceId,
          stepNum,
          plannerAction.goal || plannerAction.intent || plannerAction.action,
          0,
          currentUrl
        );
        const outcome = await this.executeStepwiseAction(
          runtime,
          plannerAction,
          stepNum,
          task,
          ctx,
          stepStart
        );
        let finalOutcome = outcome;
        stepOutcomes.push(finalOutcome);

        // Emit structured step outcome for learning extraction
        try {
          this.onStepOutcome?.(finalOutcome, ctx.snapshot?.elements);
        } catch {
          // Don't fail the run on callback errors
        }

        // Update current URL
        let urlAfter = await runtime.getCurrentUrl();
        currentUrl = urlAfter;
        fallbackUsed = fallbackUsed || finalOutcome.usedVision;
        let shouldContinue = false;
        let actionHistoryRecorded = false;

        if (finalOutcome.status === StepStatus.FAILED) {
          if (this.config.authBoundary.enabled) {
            const authResult = detectAuthBoundary(currentUrl, this.config.authBoundary);
            if (
              authResult.isAuthBoundary &&
              this.config.authBoundary.stopOnAuth &&
              !taskHasLoginIntent(task)
            ) {
              finalOutcome = {
                ...finalOutcome,
                status: StepStatus.SUCCESS,
                verificationPassed: true,
                error: this.config.authBoundary.authSuccessMessage,
                urlAfter: currentUrl,
              };
              stepOutcomes[stepOutcomes.length - 1] = finalOutcome;
              success = true;
              error = this.config.authBoundary.authSuccessMessage;
            }
          }

          if (!success) {
            const optionalSubstepOutcomes = await this.executeOptionalSubsteps(
              runtime,
              plannerAction,
              stepNum,
              task
            );
            if (optionalSubstepOutcomes.length > 0) {
              stepOutcomes.push(...optionalSubstepOutcomes);
              fallbackUsed =
                fallbackUsed ||
                optionalSubstepOutcomes.some(substepOutcome => substepOutcome.usedVision);
              currentUrl = await runtime.getCurrentUrl();
              urlAfter = currentUrl;

              const substepsRecovered =
                (plannerAction.verify?.length || 0) > 0
                  ? await this.verifyStepOutcome(runtime, plannerAction)
                  : optionalSubstepOutcomes.some(
                      substepOutcome =>
                        substepOutcome.status === StepStatus.SUCCESS ||
                        substepOutcome.status === StepStatus.SKIPPED ||
                        substepOutcome.status === StepStatus.VISION_FALLBACK
                    );

              if (substepsRecovered) {
                finalOutcome = {
                  ...finalOutcome,
                  status: StepStatus.SUCCESS,
                  verificationPassed: true,
                  error: undefined,
                  urlAfter: currentUrl,
                };
                stepOutcomes[stepOutcomes.length - optionalSubstepOutcomes.length - 1] =
                  finalOutcome;
                shouldContinue = true;
              }
            }

            if (!shouldContinue) {
              if (plannerAction.required === false) {
                shouldContinue = true;
              } else {
                const shouldAttemptRecovery = plannerAction.action !== 'STUCK';
                if (shouldAttemptRecovery && (await this.attemptRecovery(runtime))) {
                  currentUrl = await runtime.getCurrentUrl();
                  urlAfter = currentUrl;
                  shouldContinue = true;
                } else if (replansUsed < this.config.retry.maxReplans) {
                  try {
                    this.actionHistory.push({
                      stepNum,
                      action: plannerAction.action,
                      target: this.summarizePlannerActionTarget(plannerAction),
                      intent: plannerAction.intent || null,
                      result: 'failed',
                      urlAfter,
                    });
                    actionHistoryRecorded = true;
                    const repairSteps = await this.requestRepairSteps(
                      task,
                      currentUrl,
                      plannerAction,
                      finalOutcome,
                      repairHistory
                    );
                    replansUsed += 1;
                    repairHistory.push({
                      attempt: replansUsed,
                      failureCategory: this.classifyStepFailure(
                        plannerAction,
                        finalOutcome,
                        currentUrl
                      ),
                      failedAction: `${plannerAction.action}(${this.summarizePlannerActionTarget(plannerAction) || ''})`,
                      reason: finalOutcome.error || 'step failed',
                    });
                    queuedRepairSteps.push(...repairSteps);
                    shouldContinue = true;
                  } catch (repairError) {
                    error = `Replan failed: ${repairError instanceof Error ? repairError.message : String(repairError)}`;
                  }
                } else {
                  error = `Step ${stepNum} failed after reaching max replans (${this.config.retry.maxReplans})`;
                }
              }
            }
          }
        }

        // Record action history after any auth-boundary or optional-substep recovery.
        if (!actionHistoryRecorded) {
          // For EXTRACT actions, include the extracted data so the planner
          // knows what was already extracted and can avoid repeating
          const extractedText =
            plannerAction.action === 'EXTRACT' && finalOutcome.extractedData
              ? typeof finalOutcome.extractedData === 'object' &&
                finalOutcome.extractedData !== null &&
                'text' in (finalOutcome.extractedData as Record<string, unknown>)
                ? ((finalOutcome.extractedData as Record<string, unknown>).text as string)
                : JSON.stringify(finalOutcome.extractedData)
              : undefined;

          this.actionHistory.push({
            stepNum,
            action: plannerAction.action,
            target: this.summarizePlannerActionTarget(plannerAction),
            intent: plannerAction.intent || null,
            result: this.actionHistoryResult(finalOutcome.status),
            urlAfter,
            extractedData: extractedText || undefined,
          });
        }

        if (
          finalOutcome.status === StepStatus.SUCCESS ||
          finalOutcome.status === StepStatus.SKIPPED ||
          finalOutcome.status === StepStatus.VISION_FALLBACK
        ) {
          if (
            !success &&
            finalOutcome.status === StepStatus.SUCCESS &&
            (await this.isCartAdditionTerminal(runtime, task, plannerAction))
          ) {
            success = true;
          }

          if (
            !success &&
            finalOutcome.status === StepStatus.SUCCESS &&
            this.isFinalFormSubmissionAction(task, plannerAction)
          ) {
            success = true;
          }

          // Auto-complete extraction tasks: if the action was a successful EXTRACT
          // with actual data, and the overall task is an extraction task, mark as done.
          // This prevents infinite EXTRACT loops on extraction-focused tasks.
          // Guard: auto-complete if either (a) the task is purely extraction with no
          // navigation/search keywords, or (b) at least one non-EXTRACT action has
          // been performed already. This avoids premature completion on hybrid tasks
          // like "Search for X and extract Y" while still auto-completing "Extract the
          // title of the first post" when already on the right page.
          const taskHasInteraction =
            /\b(search|navigate|go to|click|add to|fill|submit|login|sign)\b/i.test(task);
          const hasNonExtractAction = this.actionHistory.some(rec => rec.action !== 'EXTRACT');
          if (
            !success &&
            plannerAction.action === 'EXTRACT' &&
            finalOutcome.status === StepStatus.SUCCESS &&
            finalOutcome.extractedData &&
            isTextExtractionTask(task) &&
            (!taskHasInteraction || hasNonExtractAction)
          ) {
            if (this.config.verbose) {
              console.log(
                `[EXTRACT] Extraction task completed successfully, transitioning to DONE`
              );
            }
            success = true;
          }

          if (this.recoveryState && this.config.recovery.trackSuccessfulUrls && urlAfter) {
            this.recoveryState.recordCheckpoint({
              url: urlAfter,
              stepIndex: stepNum - 1,
              snapshotDigest: this.makeSnapshotDigest(urlAfter),
              predicatesPassed:
                plannerAction.verify?.map(
                  pred =>
                    `${pred.predicate}(${(pred.args || []).map(arg => String(arg)).join(',')})`
                ) || [],
              verificationPredicates: plannerAction.verify || [],
            });
          }
        }

        this.emitStepEnd(stepTraceId, stepNum, plannerAction, finalOutcome);

        try {
          if (this.onStepCheckpoint) {
            this.onStepCheckpoint(
              this.exportRunState({
                stepOutcomes,
                currentUrl,
                success,
                error,
                fallbackUsed,
                replansUsed,
                queuedRepairSteps,
                repairHistory,
              })
            );
          }
        } catch {
          // Don't fail the run on checkpoint callback errors
        }

        if (error) {
          break;
        }

        if (success) {
          break;
        }

        if (shouldContinue) {
          continue;
        }

        // Check for repeated failures
        const recentFailures = stepOutcomes.slice(-3).filter(o => o.status === StepStatus.FAILED);
        if (recentFailures.length >= 3) {
          error = 'Too many consecutive failures';
          break;
        }

        const repeatedCompletedFormSkip = this.detectRepeatedCompletedFormSkip(stepOutcomes);
        if (repeatedCompletedFormSkip) {
          error = repeatedCompletedFormSkip;
          break;
        }

        // Check for URL/action cycles (e.g., click → external site → navigate back → repeat)
        const cycleDetected = this.detectActionCycle(this.actionHistory);
        if (cycleDetected) {
          error = cycleDetected;
          break;
        }
      }

      // If we ran out of steps without DONE, mark as failed
      if (!success && !error) {
        error = `Exceeded maximum steps (${maxSteps})`;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }

    const runStatus = success ? 'success' : 'failure';
    if (this.tracer) {
      this.tracer.setFinalStatus(runStatus);
      this.tracer.emitRunEnd(tracedStepCount, runStatus);
    }

    return {
      runId: this.runId,
      task,
      success,
      stepsCompleted: stepOutcomes.filter(o => o.status === StepStatus.SUCCESS).length,
      stepsTotal: stepOutcomes.length,
      replansUsed,
      stepOutcomes,
      totalDurationMs: Date.now() - startTime,
      error,
      tokenUsage: this.tokenCollector.summary(),
      fallbackUsed,
    };
  }

  async resumeRunStepwise(
    runtime: AgentRuntime,
    serializedState: SerializedAgentState,
    task: string
  ): Promise<RunOutcome> {
    const startTime = Date.now();

    this.importRunState(serializedState);

    const ls = serializedState.loopState;
    const stepOutcomes: StepOutcome[] = [...ls.stepOutcomes];
    let currentUrl = ls.currentUrl;
    let success = ls.success;
    let error = ls.error;
    let fallbackUsed = ls.fallbackUsed;
    let replansUsed = ls.replansUsed;
    const queuedRepairSteps: StepwisePlannerResponse[] =
      ls.queuedRepairSteps as unknown as StepwisePlannerResponse[];
    const repairHistory: RepairHistoryEntry[] = [...ls.repairHistory];
    let tracedStepCount = stepOutcomes.length;

    this.composableHeuristics = new ComposableHeuristics({
      staticHeuristics: this.baseIntentHeuristics,
      taskCategory: this.currentTaskCategory,
    });

    const maxSteps = this.config.stepwise.maxSteps;
    const startStep = this.currentStepIndex + 1;

    if (this.config.verbose) {
      console.log(`[RESUME] Resuming from step ${startStep}/${maxSteps}, url=${currentUrl}`);
    }

    try {
      const currentBrowserUrl = await runtime.getCurrentUrl();
      if (currentUrl && currentBrowserUrl !== currentUrl) {
        if (this.config.verbose) {
          console.log(
            `[RESUME] URL drift detected: expected=${currentUrl}, actual=${currentBrowserUrl}. Navigating back.`
          );
        }
        await runtime.goto(currentUrl);
      }
      currentUrl = currentBrowserUrl;

      for (let stepNum = startStep; stepNum <= maxSteps; stepNum++) {
        this.currentStepIndex = stepNum;
        this.composableHeuristics.clearStepHints();
        const stepStart = Date.now();

        if (this.config.verbose) {
          console.log(`\n${'='.repeat(60)}`);
          console.log(`[STEP ${stepNum}/${maxSteps}] (resumed)`);
          console.log(`${'='.repeat(60)}`);
        }

        try {
          const tokenSummary = this.tokenCollector.summary();
          this.onProgress?.({
            step: stepNum,
            maxSteps,
            totalTokens: tokenSummary.total.totalTokens,
          });
        } catch {
          // Progress callback errors are non-blocking
        }

        if (this.pauseGate) {
          await this.pauseGate();
        }

        const ctx = await this.snapshotWithEscalation(runtime, task);
        currentUrl = ctx.snapshot?.url || currentUrl;
        fallbackUsed = fallbackUsed || ctx.requiresVision;

        if (this.config.authBoundary.enabled) {
          const authResult = detectAuthBoundary(currentUrl, this.config.authBoundary);
          if (
            authResult.isAuthBoundary &&
            this.config.authBoundary.stopOnAuth &&
            !taskHasLoginIntent(task)
          ) {
            success = true;
            error = this.config.authBoundary.authSuccessMessage;
            break;
          }
        }

        let plannerAction: StepwisePlannerResponse | null = null;
        let plannerActionSource: 'planner' | 'repair' = 'planner';
        if (queuedRepairSteps.length > 0) {
          plannerAction = queuedRepairSteps.shift()!;
          plannerActionSource = 'repair';
        } else {
          const [systemPrompt, userPrompt] = buildStepwisePlannerPrompt(
            task,
            currentUrl,
            ctx.compactRepresentation,
            this.actionHistory.slice(-this.config.stepwise.actionHistoryLimit)
          );

          let plannerResp: LLMResponse | null = null;
          try {
            plannerResp = await this.callPlannerWithRetry(systemPrompt, userPrompt, {
              temperature: this.config.plannerTemperature,
              max_tokens: this.config.plannerMaxTokens,
            });
            this.recordTokenUsage('planner', plannerResp);
          } catch (plannerError) {
            const plannerErrorMessage =
              plannerError instanceof Error ? plannerError.message : String(plannerError);
            const bootstrapAction = this.buildPlannerTimeoutBootstrapAction(
              stepNum,
              task,
              ctx,
              plannerErrorMessage
            );
            if (bootstrapAction) {
              plannerAction = bootstrapAction;
              plannerActionSource = 'repair';
            } else {
              stepOutcomes.push({
                stepId: stepNum,
                goal: 'Call planner LLM',
                status: StepStatus.FAILED,
                verificationPassed: false,
                usedVision: false,
                durationMs: Date.now() - stepStart,
                error: `Planner LLM call failed: ${plannerErrorMessage}`,
              });
              if (this.shouldAbortOnPlannerFailure(plannerErrorMessage)) {
                error = `Planner unavailable: ${plannerErrorMessage}`;
                break;
              }
              continue;
            }
          }

          if (!plannerActionSource || plannerActionSource === 'planner') {
            if (!plannerResp) {
              stepOutcomes.push({
                stepId: stepNum,
                goal: 'Call planner LLM',
                status: StepStatus.FAILED,
                verificationPassed: false,
                usedVision: false,
                durationMs: Date.now() - stepStart,
                error: 'Planner response missing after planner action path',
              });
              continue;
            }

            if (!plannerResp.content || plannerResp.content.trim().length === 0) {
              stepOutcomes.push({
                stepId: stepNum,
                goal: 'Parse planner response',
                status: StepStatus.FAILED,
                verificationPassed: false,
                usedVision: false,
                durationMs: Date.now() - stepStart,
                error: 'Planner returned empty response',
              });
              continue;
            }

            try {
              plannerAction = this.normalizePlannerAction(extractJson(plannerResp.content));
            } catch (e) {
              const parsed = parseAction(plannerResp.content);
              if (parsed.action !== 'UNKNOWN') {
                plannerAction = {
                  action: parsed.action as StepwisePlannerResponse['action'],
                  input: parsed.args[1] as string | undefined,
                };
              } else {
                stepOutcomes.push({
                  stepId: stepNum,
                  goal: 'Parse planner response',
                  status: StepStatus.FAILED,
                  verificationPassed: false,
                  usedVision: false,
                  durationMs: Date.now() - stepStart,
                  error: `Failed to parse planner response: ${e}`,
                });
                continue;
              }
            }
          }
        }

        if (!plannerAction) {
          stepOutcomes.push({
            stepId: stepNum,
            goal: 'Parse planner response',
            status: StepStatus.FAILED,
            verificationPassed: false,
            usedVision: false,
            durationMs: Date.now() - stepStart,
            error: 'Planner action missing after planner/repair resolution',
          });
          continue;
        }

        plannerAction = this.promoteVisibleResultClick(task, ctx, plannerAction);
        this.composableHeuristics.setStepHints(plannerAction.heuristicHints || []);
        this.emitPlannerAction(stepNum, plannerAction, plannerActionSource);

        if (plannerAction.action === 'DONE') {
          stepOutcomes.push({
            stepId: stepNum,
            goal: 'Task completed',
            status: StepStatus.SUCCESS,
            actionTaken: 'DONE',
            verificationPassed: true,
            usedVision: false,
            durationMs: Date.now() - stepStart,
          });
          success = true;
          break;
        }

        const stepTraceId = this.getTraceStepId(stepNum)!;
        tracedStepCount += 1;
        this.tracer?.emitStepStart(
          stepTraceId,
          stepNum,
          plannerAction.goal || plannerAction.intent || plannerAction.action,
          0,
          currentUrl
        );
        const outcome = await this.executeStepwiseAction(
          runtime,
          plannerAction,
          stepNum,
          task,
          ctx,
          stepStart
        );
        let finalOutcome = outcome;
        stepOutcomes.push(finalOutcome);

        try {
          this.onStepOutcome?.(finalOutcome, ctx.snapshot?.elements);
        } catch {
          // Step outcome callback errors are non-blocking
        }

        let urlAfter = await runtime.getCurrentUrl();
        currentUrl = urlAfter;
        fallbackUsed = fallbackUsed || finalOutcome.usedVision;
        let shouldContinue = false;
        let actionHistoryRecorded = false;

        if (finalOutcome.status === StepStatus.FAILED) {
          if (this.config.authBoundary.enabled) {
            const authResult = detectAuthBoundary(currentUrl, this.config.authBoundary);
            if (
              authResult.isAuthBoundary &&
              this.config.authBoundary.stopOnAuth &&
              !taskHasLoginIntent(task)
            ) {
              finalOutcome = {
                ...finalOutcome,
                status: StepStatus.SUCCESS,
                verificationPassed: true,
                error: this.config.authBoundary.authSuccessMessage,
                urlAfter: currentUrl,
              };
              stepOutcomes[stepOutcomes.length - 1] = finalOutcome;
              success = true;
              error = this.config.authBoundary.authSuccessMessage;
            }
          }

          if (!success) {
            const optionalSubstepOutcomes = await this.executeOptionalSubsteps(
              runtime,
              plannerAction,
              stepNum,
              task
            );
            if (optionalSubstepOutcomes.length > 0) {
              stepOutcomes.push(...optionalSubstepOutcomes);
              fallbackUsed = fallbackUsed || optionalSubstepOutcomes.some(o => o.usedVision);
              currentUrl = await runtime.getCurrentUrl();
              urlAfter = currentUrl;

              const substepsRecovered =
                (plannerAction.verify?.length || 0) > 0
                  ? await this.verifyStepOutcome(runtime, plannerAction)
                  : optionalSubstepOutcomes.some(
                      o =>
                        o.status === StepStatus.SUCCESS ||
                        o.status === StepStatus.SKIPPED ||
                        o.status === StepStatus.VISION_FALLBACK
                    );

              if (substepsRecovered) {
                finalOutcome = {
                  ...finalOutcome,
                  status: StepStatus.SUCCESS,
                  verificationPassed: true,
                  error: undefined,
                  urlAfter: currentUrl,
                };
                stepOutcomes[stepOutcomes.length - optionalSubstepOutcomes.length - 1] =
                  finalOutcome;
                shouldContinue = true;
              }
            }

            if (!shouldContinue) {
              if (plannerAction.required === false) {
                shouldContinue = true;
              } else {
                const shouldAttemptRecovery = plannerAction.action !== 'STUCK';
                if (shouldAttemptRecovery && (await this.attemptRecovery(runtime))) {
                  currentUrl = await runtime.getCurrentUrl();
                  urlAfter = currentUrl;
                  shouldContinue = true;
                } else if (replansUsed < this.config.retry.maxReplans) {
                  try {
                    this.actionHistory.push({
                      stepNum,
                      action: plannerAction.action,
                      target: this.summarizePlannerActionTarget(plannerAction),
                      intent: plannerAction.intent || null,
                      result: 'failed',
                      urlAfter,
                    });
                    actionHistoryRecorded = true;
                    const repairSteps = await this.requestRepairSteps(
                      task,
                      currentUrl,
                      plannerAction,
                      finalOutcome,
                      repairHistory
                    );
                    replansUsed += 1;
                    repairHistory.push({
                      attempt: replansUsed,
                      failureCategory: this.classifyStepFailure(
                        plannerAction,
                        finalOutcome,
                        currentUrl
                      ),
                      failedAction: `${plannerAction.action}(${this.summarizePlannerActionTarget(plannerAction) || ''})`,
                      reason: finalOutcome.error || 'step failed',
                    });
                    queuedRepairSteps.push(...repairSteps);
                    shouldContinue = true;
                  } catch (repairError) {
                    error = `Replan failed: ${repairError instanceof Error ? repairError.message : String(repairError)}`;
                  }
                } else {
                  error = `Step ${stepNum} failed after reaching max replans (${this.config.retry.maxReplans})`;
                }
              }
            }
          }
        }

        if (!actionHistoryRecorded) {
          const extractedText =
            plannerAction.action === 'EXTRACT' && finalOutcome.extractedData
              ? typeof finalOutcome.extractedData === 'object' &&
                finalOutcome.extractedData !== null &&
                'text' in (finalOutcome.extractedData as Record<string, unknown>)
                ? ((finalOutcome.extractedData as Record<string, unknown>).text as string)
                : JSON.stringify(finalOutcome.extractedData)
              : undefined;

          this.actionHistory.push({
            stepNum,
            action: plannerAction.action,
            target: this.summarizePlannerActionTarget(plannerAction),
            intent: plannerAction.intent || null,
            result: this.actionHistoryResult(finalOutcome.status),
            urlAfter,
            extractedData: extractedText || undefined,
          });
        }

        if (
          finalOutcome.status === StepStatus.SUCCESS ||
          finalOutcome.status === StepStatus.SKIPPED ||
          finalOutcome.status === StepStatus.VISION_FALLBACK
        ) {
          if (
            !success &&
            finalOutcome.status === StepStatus.SUCCESS &&
            (await this.isCartAdditionTerminal(runtime, task, plannerAction))
          ) {
            success = true;
          }

          if (
            !success &&
            finalOutcome.status === StepStatus.SUCCESS &&
            this.isFinalFormSubmissionAction(task, plannerAction)
          ) {
            success = true;
          }

          const taskHasInteraction =
            /\b(search|navigate|go to|click|add to|fill|submit|login|sign)\b/i.test(task);
          const hasNonExtractAction = this.actionHistory.some(rec => rec.action !== 'EXTRACT');
          if (
            !success &&
            plannerAction.action === 'EXTRACT' &&
            finalOutcome.status === StepStatus.SUCCESS &&
            finalOutcome.extractedData &&
            isTextExtractionTask(task) &&
            (!taskHasInteraction || hasNonExtractAction)
          ) {
            success = true;
          }

          if (this.recoveryState && this.config.recovery.trackSuccessfulUrls && urlAfter) {
            this.recoveryState.recordCheckpoint({
              url: urlAfter,
              stepIndex: stepNum - 1,
              snapshotDigest: this.makeSnapshotDigest(urlAfter),
              predicatesPassed:
                plannerAction.verify?.map(
                  pred =>
                    `${pred.predicate}(${(pred.args || []).map(arg => String(arg)).join(',')})`
                ) || [],
              verificationPredicates: plannerAction.verify || [],
            });
          }
        }

        this.emitStepEnd(stepTraceId, stepNum, plannerAction, finalOutcome);

        try {
          if (this.onStepCheckpoint) {
            this.onStepCheckpoint(
              this.exportRunState({
                stepOutcomes,
                currentUrl,
                success,
                error,
                fallbackUsed,
                replansUsed,
                queuedRepairSteps,
                repairHistory,
              })
            );
          }
        } catch {
          // Checkpoint callback errors are non-blocking
        }

        if (error) {
          break;
        }

        if (success) {
          break;
        }

        if (shouldContinue) {
          continue;
        }

        const recentFailures = stepOutcomes.slice(-3).filter(o => o.status === StepStatus.FAILED);
        if (recentFailures.length >= 3) {
          error = 'Too many consecutive failures';
          break;
        }

        const repeatedCompletedFormSkip = this.detectRepeatedCompletedFormSkip(stepOutcomes);
        if (repeatedCompletedFormSkip) {
          error = repeatedCompletedFormSkip;
          break;
        }

        const cycleDetected = this.detectActionCycle(this.actionHistory);
        if (cycleDetected) {
          error = cycleDetected;
          break;
        }
      }

      if (!success && !error) {
        error = `Exceeded maximum steps (${maxSteps})`;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }

    const runStatus = success ? 'success' : 'failure';
    if (this.tracer) {
      this.tracer.setFinalStatus(runStatus);
      this.tracer.emitRunEnd(tracedStepCount, runStatus);
    }

    return {
      runId: this.runId ?? crypto.randomUUID(),
      task,
      success,
      stepsCompleted: stepOutcomes.filter(o => o.status === StepStatus.SUCCESS).length,
      stepsTotal: stepOutcomes.length,
      replansUsed,
      stepOutcomes,
      totalDurationMs: Date.now() - startTime,
      error,
      tokenUsage: this.tokenCollector.summary(),
      fallbackUsed,
    };
  }

  /**
   * Call the planner LLM with retry and exponential backoff for 429 rate-limit errors.
   *
   * Without this, a single 429 burns an entire step. With retry, we wait for the
   * rate limit to reset (using the `retry-after` hint when available) and try again.
   */
  private async callPlannerWithRetry(
    systemPrompt: string,
    userPrompt: string,
    options: { temperature: number; max_tokens: number }
  ): Promise<LLMResponse> {
    const { llmRetryBackoffMs, llmMaxRetries } = this.config.retry;

    for (let attempt = 0; attempt <= llmMaxRetries; attempt++) {
      try {
        return await this.planner.generate(systemPrompt, userPrompt, options);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        const is429 =
          msg.includes('429') ||
          msg.toLowerCase().includes('rate_limit') ||
          msg.toLowerCase().includes('rate limit');

        if (!is429 || attempt >= llmMaxRetries) {
          throw err;
        }

        // Try to extract retry-after from error message (Groq includes "try again in X.XXs")
        const retryMatch = msg.match(/try again in ([\d.]+)\s*s/i);
        let delayMs = llmRetryBackoffMs * Math.pow(2, attempt);
        if (retryMatch) {
          delayMs = Math.max(delayMs, Math.ceil(parseFloat(retryMatch[1]) * 1000) + 500);
        }

        if (this.config.verbose) {
          console.log(
            `[PLANNER RETRY] 429 rate limit hit (attempt ${attempt + 1}/${llmMaxRetries}), waiting ${Math.round(delayMs / 1000)}s...`
          );
        }

        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }

    // Should never reach here, but TypeScript needs it
    throw new Error('Planner retry exhausted');
  }

  private shouldAbortOnPlannerFailure(message: string): boolean {
    const normalized = message.toLowerCase();
    return (
      normalized.includes('timed out') ||
      normalized.includes('browser agent run cancelled') ||
      normalized.includes('signal is aborted')
    );
  }

  private buildPlannerTimeoutBootstrapAction(
    stepNum: number,
    task: string,
    ctx: SnapshotContext,
    plannerErrorMessage: string
  ): StepwisePlannerResponse | null {
    if (stepNum !== 1 || !plannerErrorMessage.toLowerCase().includes('timed out')) {
      return null;
    }
    const elements = ctx.snapshot?.elements || [];
    const hasSearchInput = elements.some(element =>
      ['searchbox', 'textbox', 'combobox', 'input', 'textarea'].includes(
        (element.role || '').toLowerCase()
      )
    );
    if (!hasSearchInput) {
      return null;
    }
    const query = this.extractSearchQueryFromTask(task);
    if (!query) {
      return null;
    }
    const verificationToken = this.extractVerificationTokenFromQuery(query);
    return {
      action: 'TYPE_AND_SUBMIT',
      intent: 'searchbox',
      input: query,
      verify: verificationToken ? [{ predicate: 'url_contains', args: [verificationToken] }] : [],
      required: true,
      reasoning: 'Deterministic fallback after planner timeout on step 1',
    };
  }

  private extractSearchQueryFromTask(task: string): string | null {
    const directPattern = /search\s+for\s+(.+?)(?:\s+on\s+|,|then|$)/i;
    const directMatch = task.match(directPattern);
    const raw = directMatch?.[1] ?? task.match(/search\s+(.+?)(?:\s+on\s+|,|then|$)/i)?.[1] ?? null;
    if (!raw) {
      return null;
    }
    const cleaned = raw.replace(/^["']|["']$/g, '').trim();
    return cleaned.length > 0 ? cleaned : null;
  }

  private extractVerificationTokenFromQuery(query: string): string | null {
    const normalized = query
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(Boolean);
    const token = normalized.find(part => part.length >= 3);
    return token || null;
  }

  // ---------------------------------------------------------------------------
  // Execute Stepwise Action
  // ---------------------------------------------------------------------------

  private async executeStepwiseAction(
    runtime: AgentRuntime,
    plannerAction: StepwisePlannerResponse,
    stepNum: number,
    task: string,
    ctx: SnapshotContext,
    stepStart: number
  ): Promise<StepOutcome> {
    const currentUrl = ctx.snapshot?.url || '';
    const stepGoal = plannerAction.intent || plannerAction.action;

    if (this.isPreviouslyCompletedFormDataEntry(plannerAction, currentUrl)) {
      return {
        stepId: stepNum,
        goal: stepGoal,
        status: StepStatus.SKIPPED,
        actionTaken: 'SKIPPED(previously_completed_form_step)',
        verificationPassed: true,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        urlBefore: currentUrl,
        urlAfter: currentUrl,
      };
    }

    if (
      this.config.preStepVerification &&
      (plannerAction.verify?.length || 0) > 0 &&
      plannerAction.action !== 'TYPE_AND_SUBMIT' &&
      plannerAction.action !== 'TYPE' &&
      !this.isFormDataEntryPlannerAction(plannerAction) &&
      !this.isForwardNavigationPlannerAction(plannerAction)
    ) {
      const alreadySatisfied = await this.checkPreStepVerification(runtime, plannerAction);
      if (alreadySatisfied) {
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: StepStatus.SKIPPED,
          actionTaken: 'SKIPPED(pre_verification_passed)',
          verificationPassed: true,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter: currentUrl,
        };
      }
    }

    if (plannerAction.action === 'NAVIGATE') {
      if (!plannerAction.target) {
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: StepStatus.FAILED,
          verificationPassed: false,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          error: 'NAVIGATE action is missing target URL',
        };
      }

      if (this.isCopiedPlaceholderNavigation(plannerAction.target, currentUrl, task)) {
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: StepStatus.SKIPPED,
          actionTaken: 'SKIPPED(placeholder_navigation)',
          verificationPassed: true,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter: currentUrl,
        };
      }

      try {
        await runtime.goto(plannerAction.target);
        const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
        const urlAfter = await runtime.getCurrentUrl();
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: `NAVIGATE(${plannerAction.target})`,
          verificationPassed,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter,
        };
      } catch (e) {
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: StepStatus.FAILED,
          verificationPassed: false,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          error: e instanceof Error ? e.message : String(e),
        };
      }
    }

    // Handle SCROLL action
    if (plannerAction.action === 'SCROLL') {
      const direction = plannerAction.direction || 'down';
      try {
        await runtime.scroll(direction);
        return {
          stepId: stepNum,
          goal: `Scroll ${direction}`,
          status: StepStatus.SUCCESS,
          actionTaken: `SCROLL(${direction})`,
          verificationPassed: true,
          usedVision: false,
          durationMs: Date.now() - stepStart,
        };
      } catch (e) {
        return {
          stepId: stepNum,
          goal: `Scroll ${direction}`,
          status: StepStatus.FAILED,
          verificationPassed: false,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          error: e instanceof Error ? e.message : String(e),
        };
      }
    }

    if (plannerAction.action === 'WAIT') {
      const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
      return {
        stepId: stepNum,
        goal: stepGoal,
        status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
        actionTaken: 'WAIT',
        verificationPassed,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        urlBefore: currentUrl,
        urlAfter: await runtime.getCurrentUrl(),
      };
    }

    if (plannerAction.action === 'STUCK') {
      return {
        stepId: stepNum,
        goal: stepGoal,
        status: StepStatus.FAILED,
        actionTaken: 'STUCK',
        verificationPassed: false,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        error: plannerAction.reasoning || 'Planner reported no safe next action',
      };
    }

    if (plannerAction.action === 'FILL_FORM') {
      const fields = plannerAction.fields || [];
      const submitText = plannerAction.submitText || '';
      const elements = ctx.snapshot?.elements || [];
      const actions: string[] = [];
      let filledCount = 0;

      const inputRoles = ['textbox', 'searchbox', 'combobox', 'input', 'password'];

      for (const field of fields) {
        const label = (field.label || '').toLowerCase();
        const value = field.value || '';
        let matched: SnapshotElement | null = null;

        for (const el of elements) {
          const role = (el.role || '').toLowerCase();
          if (!inputRoles.some(r => role.includes(r))) continue;

          const elText = (el.text || '').toLowerCase();
          const elName = (el.name || '').toLowerCase();
          const elAria = (el.ariaLabel || '').toLowerCase();
          const elNearby = (el.nearbyText || '').toLowerCase();
          const allText = `${elText} ${elName} ${elAria} ${elNearby}`;

          if (allText.includes(label) || label.includes(elText) || label.includes(elName)) {
            matched = el;
            break;
          }
        }

        if (matched) {
          await runtime.type(matched.id, value);
          actions.push(`TYPE(${matched.id}, "${value}")`);
          filledCount++;
          if (this.config.verbose) {
            console.log(
              `[FILL_FORM] Typed "${value}" into element ${matched.id} (${matched.text || matched.role})`
            );
          }
        } else if (this.config.verbose) {
          console.log(`[FILL_FORM] No match for field label "${label}"`);
        }
      }

      if (filledCount === 0) {
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: StepStatus.FAILED,
          actionTaken: 'FILL_FORM(0 fields matched)',
          verificationPassed: false,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter: await runtime.getCurrentUrl(),
          error: 'No form fields matched the provided labels',
        };
      }

      let clickedSubmit = false;
      if (submitText) {
        const submitEl = this.findSubmitButtonByText(elements, submitText);
        if (submitEl !== null) {
          await runtime.click(submitEl);
          actions.push(`CLICK(${submitEl})`);
          clickedSubmit = true;
          if (this.config.verbose) {
            console.log(`[FILL_FORM] Clicked submit button element ${submitEl}`);
          }
        }
      }

      await new Promise(r => setTimeout(r, clickedSubmit ? 1000 : 300));
      const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
      const urlAfter = await runtime.getCurrentUrl();

      return {
        stepId: stepNum,
        goal: stepGoal,
        status: verificationPassed || clickedSubmit ? StepStatus.SUCCESS : StepStatus.FAILED,
        actionTaken: `FILL_FORM(${actions.join(' -> ')})`,
        verificationPassed,
        usedVision: false,
        durationMs: Date.now() - stepStart,
        urlBefore: currentUrl,
        urlAfter,
      };
    }

    // Handle EXTRACT action — read page content and extract data via LLM
    if (plannerAction.action === 'EXTRACT') {
      const extractQuery =
        plannerAction.goal ||
        plannerAction.intent ||
        plannerAction.target ||
        task ||
        'Extract relevant data from the current page';

      const stripThinkTags = (text: string): string =>
        text
          .replace(/<think[\s\S]*?<\/think>/gi, '')
          .replace(/<think[\s\S]*$/gi, '')
          .trim();

      if (this.config.verbose) {
        console.log(`[ACTION] EXTRACT - query: "${extractQuery}"`);
      }

      try {
        const useMarkdown = runtime.readMarkdown != null;

        if (useMarkdown && runtime.readMarkdown) {
          // Text-based extraction: read page as markdown, then use executor LLM
          const pageContent = await runtime.readMarkdown({ maxChars: 16000 });

          if (!pageContent) {
            return {
              stepId: stepNum,
              goal: extractQuery,
              status: StepStatus.FAILED,
              actionTaken: 'EXTRACT',
              verificationPassed: false,
              usedVision: false,
              durationMs: Date.now() - stepStart,
              error: 'Failed to read page content as markdown',
            };
          }

          if (this.config.verbose) {
            console.log(`  [ACTION] EXTRACT - got markdown (${pageContent.length} chars):`);
            console.log(pageContent.slice(0, 2000));
            if (pageContent.length > 2000)
              console.log(`  ... [truncated, ${pageContent.length - 2000} more chars]`);
          }

          // Build extraction prompt and call executor LLM
          const [extSystem, extUser] = buildExtractionPrompt(pageContent, extractQuery);
          const extractResp = await this.executor.generate(extSystem, extUser, {
            temperature: 0.0,
            max_tokens: 1024,
          });
          this.recordTokenUsage('extract', extractResp);

          const extractedText = stripThinkTags((extractResp.content || '').trim());
          if (extractedText && extractedText !== 'NOT_FOUND') {
            if (this.config.verbose) {
              console.log(`  [ACTION] EXTRACT ok: ${extractedText.slice(0, 160)}`);
            }
            return {
              stepId: stepNum,
              goal: extractQuery,
              status: StepStatus.SUCCESS,
              actionTaken: 'EXTRACT',
              verificationPassed: true,
              usedVision: false,
              durationMs: Date.now() - stepStart,
              urlBefore: currentUrl,
              urlAfter: currentUrl,
              extractedData: { text: extractedText, query: extractQuery },
            };
          } else {
            return {
              stepId: stepNum,
              goal: extractQuery,
              status: StepStatus.FAILED,
              actionTaken: 'EXTRACT',
              verificationPassed: false,
              usedVision: false,
              durationMs: Date.now() - stepStart,
              error: `Could not find requested data: ${extractQuery}`,
              pageContentPreview: pageContent.slice(0, 500),
            };
          }
        } else {
          // Fallback: use compact snapshot context for extraction
          const pageContent = ctx.compactRepresentation;
          if (!pageContent || pageContent.trim().length === 0) {
            return {
              stepId: stepNum,
              goal: extractQuery,
              status: StepStatus.FAILED,
              actionTaken: 'EXTRACT',
              verificationPassed: false,
              usedVision: false,
              durationMs: Date.now() - stepStart,
              error: 'No page content available for extraction',
            };
          }

          const [extSystem, extUser] = buildExtractionPrompt(pageContent, extractQuery);
          const extractResp = await this.executor.generate(extSystem, extUser, {
            temperature: 0.0,
            max_tokens: 1024,
          });
          this.recordTokenUsage('extract', extractResp);

          const extractedText = stripThinkTags((extractResp.content || '').trim());
          if (extractedText && extractedText !== 'NOT_FOUND') {
            if (this.config.verbose) {
              console.log(`  [ACTION] EXTRACT ok (snapshot): ${extractedText.slice(0, 160)}`);
            }
            return {
              stepId: stepNum,
              goal: extractQuery,
              status: StepStatus.SUCCESS,
              actionTaken: 'EXTRACT',
              verificationPassed: true,
              usedVision: false,
              durationMs: Date.now() - stepStart,
              urlBefore: currentUrl,
              urlAfter: currentUrl,
              extractedData: { text: extractedText, query: extractQuery },
            };
          } else {
            return {
              stepId: stepNum,
              goal: extractQuery,
              status: StepStatus.FAILED,
              actionTaken: 'EXTRACT',
              verificationPassed: false,
              usedVision: false,
              durationMs: Date.now() - stepStart,
              error: `Could not extract requested data: ${extractQuery}`,
            };
          }
        }
      } catch (e) {
        return {
          stepId: stepNum,
          goal: extractQuery,
          status: StepStatus.FAILED,
          actionTaken: 'EXTRACT',
          verificationPassed: false,
          usedVision: false,
          durationMs: Date.now() - stepStart,
          error: e instanceof Error ? e.message : String(e),
        };
      }
    }

    // For CLICK and TYPE_AND_SUBMIT, we need to find the element
    const isTypeAction = plannerAction.action === 'TYPE_AND_SUBMIT';

    // Actions that need to find a target element
    const elementTargetingActions = ['CLICK', 'TYPE_AND_SUBMIT', 'SCROLL_TO'];
    const needsElementLookup = elementTargetingActions.includes(plannerAction.action);

    // Check if target element exists in current snapshot
    // If not, try scroll-after-escalation to find it
    // This is GENERALIZABLE - works for any action that needs to find an element
    let activeCtx = ctx;
    if (needsElementLookup && this.config.snapshot.scrollAfterEscalation && plannerAction.intent) {
      const elements = ctx.snapshot?.elements || [];
      const url = ctx.snapshot?.url || '';
      const foundElement = this.tryIntentHeuristics(plannerAction.intent, elements, url, task);

      if (foundElement === null) {
        // Element not found in current viewport - try scroll-after-escalation
        if (this.config.verbose) {
          console.log(
            `[SCROLL-TO-FIND] Target "${plannerAction.intent}" not in viewport, scrolling to find...`
          );
        }

        const newCtx = await this.snapshotWithEscalation(runtime, task, {
          action: plannerAction.action,
          intent: plannerAction.intent,
        });

        if (newCtx.snapshot) {
          activeCtx = newCtx;
          if (this.config.verbose) {
            console.log(
              `[SCROLL-TO-FIND] Updated context with ${newCtx.snapshot.elements.length} elements`
            );
          }
        }
      }
    }

    let { parsed, shouldUseVision, executorResp } = await this.resolveExecutorAction(
      plannerAction,
      activeCtx,
      task
    );

    if (
      parsed.action === 'NONE' &&
      plannerAction.intent &&
      activeCtx.snapshot &&
      activeCtx.pruningCategory !== null &&
      activeCtx.prunedNodeCount < activeCtx.snapshot.elements.length &&
      fullSnapshotContainsIntent(activeCtx.snapshot, plannerAction.intent)
    ) {
      const relaxedCtx = await this.snapshotWithEscalation(runtime, task, {
        action: plannerAction.action,
        intent: plannerAction.intent,
        relaxPruning: true,
      });
      activeCtx = relaxedCtx;
      ({ parsed, shouldUseVision, executorResp } = await this.resolveExecutorAction(
        plannerAction,
        activeCtx,
        task
      ));
    }

    if (
      parsed.action === 'NONE' &&
      !shouldUseVision &&
      plannerAction.intent &&
      activeCtx.snapshot &&
      fullSnapshotContainsIntent(activeCtx.snapshot, plannerAction.intent) &&
      this.executor.supportsVision() &&
      Boolean(activeCtx.screenshotBase64)
    ) {
      ({ parsed, shouldUseVision, executorResp } = await this.resolveExecutorAction(
        plannerAction,
        activeCtx,
        task,
        true
      ));
    }

    // Debug: Show parsed result
    if (this.config.verbose) {
      console.log(`[EXECUTOR PARSED] ${parsed.action}, args: ${JSON.stringify(parsed.args)}`);
    }

    if (parsed.action === 'NONE') {
      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        llmResponseText: executorResp?.content,
        verificationPassed: false,
        usedVision: shouldUseVision,
        durationMs: Date.now() - stepStart,
        error: 'Executor could not find suitable element',
      };
    }

    if (parsed.action === 'UNKNOWN') {
      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        llmResponseText: executorResp?.content,
        verificationPassed: false,
        usedVision: shouldUseVision,
        durationMs: Date.now() - stepStart,
        error: `Failed to parse executor response: ${executorResp?.content ?? 'no executor response'}`,
      };
    }

    // Pre-action authorization
    if (this.preActionAuthorizer) {
      const actionContext = {
        type: parsed.action,
        elementId: parsed.args[0] as number | undefined,
        value: isTypeAction
          ? ((plannerAction.input || parsed.args[1]) as string | undefined)
          : undefined,
      };
      const authResult = await this.preActionAuthorizer(actionContext, {
        url: currentUrl,
        stepGoal: plannerAction.intent || plannerAction.action,
        taskGoal: task,
      });

      if (!authResult.allowed) {
        return {
          stepId: stepNum,
          goal: plannerAction.intent || plannerAction.action,
          status: StepStatus.FAILED,
          llmResponseText: executorResp?.content,
          verificationPassed: false,
          usedVision: shouldUseVision,
          durationMs: Date.now() - stepStart,
          error: `Action denied by policy: ${authResult.reason || 'unauthorized'}`,
        };
      }
    }

    // Execute the action
    try {
      const elementId = parsed.args[0] as number;

      // Vision-fallback coordinate actions
      if (parsed.action === 'CLICK_XY') {
        const [vx, vy] = parsed.args as [number, number];
        if (this.config.verbose) {
          console.log(`[VISION] CLICK_XY(${vx}, ${vy})`);
        }
        if (runtime.clickCoordinate) {
          await runtime.clickCoordinate(vx, vy);
        } else {
          return {
            stepId: stepNum,
            goal: stepGoal,
            status: StepStatus.FAILED,
            llmResponseText: executorResp?.content,
            verificationPassed: false,
            usedVision: true,
            durationMs: Date.now() - stepStart,
            error: 'CLICK_XY not supported by runtime (no clickCoordinate method)',
          };
        }
        const isVisionTypeAction =
          plannerAction.action === 'TYPE_AND_SUBMIT' || plannerAction.action === 'TYPE';
        if (isVisionTypeAction && plannerAction.input && runtime.typeCoordinate) {
          await new Promise(r => setTimeout(r, 300));
          await runtime.typeCoordinate(plannerAction.input);
          if (this.config.verbose) {
            console.log(`[VISION] AUTO-TYPE after CLICK_XY: "${plannerAction.input}"`);
          }
        }
        await new Promise(r => setTimeout(r, 500));
        let urlAfter = await runtime.getCurrentUrl();
        let verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
        let finalActionTaken =
          isVisionTypeAction && plannerAction.input
            ? `CLICK_XY(${vx}, ${vy}) + TYPE_AT("${plannerAction.input}")`
            : `CLICK_XY(${vx}, ${vy})`;
        let postNextPreview: string | undefined;
        if (
          !verificationPassed &&
          plannerAction.action === 'TYPE' &&
          (plannerAction.verify?.length || 0) > 0
        ) {
          const nextButtonId = this.findSubmitButton(activeCtx.snapshot?.elements || [], 0, false);
          if (nextButtonId !== null) {
            try {
              await runtime.click(nextButtonId);
              finalActionTaken = `${finalActionTaken} -> CLICK(${nextButtonId})`;
              verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
              urlAfter = await runtime.getCurrentUrl();
              if (!verificationPassed) {
                const postNextSnapshot = await runtime.snapshot({
                  limit: this.config.snapshot.limitBase,
                  screenshot: false,
                  goal: plannerAction.intent || plannerAction.action,
                });
                if (postNextSnapshot) {
                  verificationPassed = this.inferSameUrlWizardProgressAfterNext(
                    plannerAction,
                    activeCtx.snapshot,
                    postNextSnapshot
                  );
                  postNextPreview = formatContext(
                    postNextSnapshot.elements || [],
                    this.config.snapshot.limitBase
                  ).slice(0, 500);
                }
              }
            } catch (e) {
              if (this.config.verbose) {
                console.log(`[VISION-TYPE-FORM-NEXT] Next click failed: ${e}`);
              }
            }
          }
        }
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: finalActionTaken,
          llmResponseText: executorResp?.content,
          verificationPassed,
          usedVision: true,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter,
          pageContentPreview: verificationPassed ? undefined : postNextPreview,
        };
      }

      if (parsed.action === 'CLICK_RECT') {
        const [rx, ry, rw, rh] = parsed.args as [number, number, number, number];
        const cx = rx + rw / 2;
        const cy = ry + rh / 2;
        if (this.config.verbose) {
          console.log(`[VISION] CLICK_RECT(${rx}, ${ry}, ${rw}, ${rh}) → center (${cx}, ${cy})`);
        }
        if (runtime.clickCoordinate) {
          await runtime.clickCoordinate(cx, cy);
        } else {
          return {
            stepId: stepNum,
            goal: stepGoal,
            status: StepStatus.FAILED,
            llmResponseText: executorResp?.content,
            verificationPassed: false,
            usedVision: true,
            durationMs: Date.now() - stepStart,
            error: 'CLICK_RECT not supported by runtime (no clickCoordinate method)',
          };
        }
        await new Promise(r => setTimeout(r, 500));
        const urlAfter = await runtime.getCurrentUrl();
        const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: `CLICK_RECT(${rx}, ${ry}, ${rw}, ${rh})`,
          llmResponseText: executorResp?.content,
          verificationPassed,
          usedVision: true,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter,
        };
      }

      if (parsed.action === 'TYPE_AT') {
        const [textToType] = parsed.args as [string];
        if (this.config.verbose) {
          console.log(`[VISION] TYPE_AT("${textToType}")`);
        }
        if (runtime.typeCoordinate) {
          await runtime.typeCoordinate(textToType);
        } else {
          return {
            stepId: stepNum,
            goal: stepGoal,
            status: StepStatus.FAILED,
            llmResponseText: executorResp?.content,
            verificationPassed: false,
            usedVision: true,
            durationMs: Date.now() - stepStart,
            error: 'TYPE_AT not supported by runtime (no typeCoordinate method)',
          };
        }
        await new Promise(r => setTimeout(r, 300));
        const urlAfter = await runtime.getCurrentUrl();
        const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
        return {
          stepId: stepNum,
          goal: stepGoal,
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: `TYPE_AT("${textToType}")`,
          llmResponseText: executorResp?.content,
          verificationPassed,
          usedVision: true,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter,
        };
      }

      if (parsed.action === 'CLICK') {
        const targetElement =
          activeCtx.snapshot?.elements.find(element => element.id === elementId) || null;
        await runtime.click(elementId);
        let postClickEffectsError: string | undefined;
        let modalHandled = false;
        try {
          const postClickResult = await this.handlePostClickEffects(
            runtime,
            plannerAction,
            activeCtx
          );
          modalHandled = postClickResult.handled;
        } catch (postClickError) {
          postClickEffectsError =
            postClickError instanceof Error ? postClickError.message : String(postClickError);
          if (this.config.verbose) {
            console.log(`[CLICK] Post-click effects failed: ${postClickEffectsError}`);
          }
        }
        const urlAfter = await runtime.getCurrentUrl();
        const hasUrlVerification = (plannerAction.verify || []).some(
          predicate =>
            predicate.predicate === 'url_contains' ||
            predicate.predicate === 'url_equals' ||
            predicate.predicate === 'url_matches'
        );
        const relevantUrlChange = isUrlChangeRelevantToIntent(
          currentUrl,
          urlAfter,
          plannerAction,
          targetElement
        );
        const navigationSatisfied =
          relevantUrlChange &&
          (!hasUrlVerification ||
            this.clickedHrefMatchesNavigation(currentUrl, urlAfter, targetElement));
        const hasVerificationPredicates = (plannerAction.verify?.length || 0) > 0;
        const requiresNavigation = this.clickIntentRequiresNavigation(plannerAction, targetElement);
        const predicateSatisfied = hasVerificationPredicates
          ? await this.verifyStepOutcome(runtime, plannerAction)
          : false;
        let verificationPassed =
          navigationSatisfied ||
          predicateSatisfied ||
          (!requiresNavigation && !hasVerificationPredicates) ||
          modalHandled;

        if (!verificationPassed) {
          const postClickSnapshot = await runtime.snapshot({
            limit: this.config.snapshot.limitBase,
            screenshot: false,
            goal: plannerAction.intent || plannerAction.action,
          });
          if (postClickSnapshot) {
            verificationPassed = this.inferSameUrlWizardProgressAfterNavigationClick(
              plannerAction,
              targetElement,
              activeCtx.snapshot,
              postClickSnapshot
            );
            if (!verificationPassed) {
              verificationPassed = this.inferTerminalFormSubmissionAfterClick(
                plannerAction,
                targetElement,
                activeCtx.snapshot,
                postClickSnapshot
              );
            }
            if (!verificationPassed) {
              verificationPassed = this.inferFormChoiceSelectionAfterClick(
                plannerAction,
                targetElement,
                postClickSnapshot
              );
            }
          }
        }

        if (
          verificationPassed &&
          (this.currentTaskCategory === TaskCategory.EXTRACTION || isTextExtractionTask(task))
        ) {
          const navigatedAway = this.urlHostKey(currentUrl) !== this.urlHostKey(urlAfter);
          const clickWasOnListedData =
            plannerAction.intent &&
            /^(product|item|listing|result|article|post|link)/i.test(plannerAction.intent);
          if (navigatedAway && clickWasOnListedData) {
            verificationPassed = false;
            if (this.config.verbose) {
              console.log(
                `[CLICK] Extraction task: click on "${plannerAction.intent}" navigated away from source page, marking as failed`
              );
            }
          }
        }

        let finalActionTaken = `CLICK(${elementId})`;
        let finalUrlAfter = urlAfter;

        const clickedElementWasNavigationLink =
          targetElement &&
          ((targetElement.role || '').toLowerCase() === 'link' ||
            Boolean(targetElement.href && !targetElement.href.startsWith('#')));
        if (
          !verificationPassed &&
          requiresNavigation &&
          clickedElementWasNavigationLink &&
          activeCtx.snapshot?.elements
        ) {
          const fallbackElementId = this.findFallbackNavigationClickTarget(
            activeCtx.snapshot.elements,
            elementId,
            plannerAction
          );
          if (fallbackElementId !== null) {
            const fallbackElement =
              activeCtx.snapshot.elements.find(element => element.id === fallbackElementId) || null;
            try {
              await runtime.click(fallbackElementId);
              finalActionTaken = `CLICK(${elementId}) -> CLICK(${fallbackElementId})`;
              finalUrlAfter = await runtime.getCurrentUrl();
              const fallbackNavigationSatisfied = isUrlChangeRelevantToIntent(
                currentUrl,
                finalUrlAfter,
                plannerAction,
                fallbackElement
              );
              const fallbackPredicateSatisfied = hasVerificationPredicates
                ? await this.verifyStepOutcome(runtime, plannerAction)
                : false;
              verificationPassed = fallbackNavigationSatisfied || fallbackPredicateSatisfied;
            } catch (fallbackClickError) {
              if (this.config.verbose) {
                console.log(
                  `[CLICK-FALLBACK] Failed fallback click ${fallbackElementId}: ${fallbackClickError}`
                );
              }
            }
          }
        }
        return {
          stepId: stepNum,
          goal: plannerAction.intent || 'Click element',
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: finalActionTaken,
          llmResponseText: executorResp?.content,
          verificationPassed,
          usedVision: shouldUseVision,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter: finalUrlAfter,
          error: !verificationPassed ? postClickEffectsError : undefined,
        };
      } else if (parsed.action === 'TYPE') {
        const text = plannerAction.input || (parsed.args[1] as string) || '';
        await runtime.type(elementId, text);

        const elements = activeCtx.snapshot?.elements || [];
        const inputElement = elements.find(element => element.id === elementId) || null;
        const isSearchLike = isSearchLikeTypeAndSubmit(plannerAction, inputElement);
        const hasVerificationPredicates = (plannerAction.verify?.length || 0) > 0;
        let submissionSatisfied = false;
        let finalActionTaken = `TYPE(${elementId}, "${text}")`;

        // Submit with Enter key for TYPE_AND_SUBMIT, plus planner TYPE actions that clearly target search.
        if (
          plannerAction.action === 'TYPE_AND_SUBMIT' ||
          (plannerAction.action === 'TYPE' && isSearchLike)
        ) {
          const preUrl = await runtime.getCurrentUrl();
          const submitButtonId = this.findSubmitButton(elements, elementId, isSearchLike);
          const hasRetryBudget = this.config.retry.executorRepairAttempts > 0;

          let changedUrl: string | null = null;

          const checkSubmissionSatisfied = async (): Promise<boolean> => {
            if (
              changedUrl !== null &&
              isUrlChangeRelevantToIntent(preUrl, changedUrl, plannerAction, inputElement)
            ) {
              return true;
            }

            if ((plannerAction.verify?.length || 0) > 0) {
              return this.verifyStepOutcome(runtime, plannerAction);
            }

            return false;
          };

          const submitWithClick = async (): Promise<boolean> => {
            if (submitButtonId === null) {
              return false;
            }

            try {
              await runtime.click(submitButtonId);
              changedUrl = await this.waitForUrlChange(runtime, preUrl, 5000);
              return checkSubmissionSatisfied();
            } catch (e) {
              if (this.config.verbose) {
                console.log(`[TYPE_AND_SUBMIT] Explicit submit click failed: ${e}`);
              }
              return false;
            }
          };

          const submitWithEnter = async (): Promise<boolean> => {
            await runtime.pressKey('Enter');
            changedUrl = await this.waitForUrlChange(runtime, preUrl, 5000);
            return checkSubmissionSatisfied();
          };

          if (!isSearchLike && submitButtonId !== null) {
            submissionSatisfied = await submitWithClick();
          }

          if (!submissionSatisfied && (isSearchLike || submitButtonId === null || hasRetryBudget)) {
            submissionSatisfied = await submitWithEnter();
          }

          if (this.config.verbose) {
            if (submissionSatisfied && changedUrl) {
              console.log(
                `[TYPE_AND_SUBMIT] Relevant URL change detected: ${String(changedUrl).slice(0, 60)}...`
              );
            } else if (submissionSatisfied) {
              console.log(`[TYPE_AND_SUBMIT] Verification passed without a relevant URL change`);
            } else {
              console.log(
                `[TYPE_AND_SUBMIT] No relevant URL change detected, evaluating explicit retry...`
              );
            }
          }

          // Retry with button click if Enter didn't produce a relevant submission.
          if (!submissionSatisfied && isSearchLike && hasRetryBudget && submitButtonId !== null) {
            if (this.config.verbose) {
              console.log(
                `[TYPE_AND_SUBMIT-RETRY] Found submit button ${submitButtonId}, retrying with click`
              );
            }

            submissionSatisfied = await submitWithClick();

            if (this.config.verbose && changedUrl) {
              console.log(
                `[TYPE_AND_SUBMIT-RETRY] URL after click: ${String(changedUrl).slice(0, 60)}...`
              );
            }
          } else if (!submissionSatisfied && this.config.verbose && submitButtonId === null) {
            console.log(`[TYPE_AND_SUBMIT-RETRY] No submit button found for retry`);
          }

          // Wait for page to stabilize
          if (submissionSatisfied && changedUrl) {
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }

        let verificationPassed =
          submissionSatisfied || (await this.verifyStepOutcome(runtime, plannerAction));
        const urlAfter = await runtime.getCurrentUrl();

        if (
          !verificationPassed &&
          plannerAction.action === 'TYPE' &&
          !isSearchLike &&
          hasVerificationPredicates
        ) {
          const nextButtonId = this.findSubmitButton(elements, elementId, false);
          if (nextButtonId !== null) {
            try {
              await runtime.click(nextButtonId);
              finalActionTaken = `${finalActionTaken} -> CLICK(${nextButtonId})`;
              let advanced = await this.verifyStepOutcome(runtime, plannerAction);
              if (!advanced) {
                const postNextSnapshot = await runtime.snapshot({
                  limit: this.config.snapshot.limitBase,
                  screenshot: false,
                  goal: plannerAction.intent || plannerAction.action,
                });
                if (postNextSnapshot) {
                  advanced = this.inferSameUrlWizardProgressAfterNext(
                    plannerAction,
                    activeCtx.snapshot,
                    postNextSnapshot
                  );
                }
              }
              if (advanced) {
                verificationPassed = true;
              }
            } catch (e) {
              if (this.config.verbose) {
                console.log(`[TYPE-FORM-NEXT] Next click failed: ${e}`);
              }
            }
          }
        }

        return {
          stepId: stepNum,
          goal: plannerAction.intent || 'Type text',
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: finalActionTaken,
          llmResponseText: executorResp?.content,
          verificationPassed,
          usedVision: shouldUseVision,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter,
        };
      } else if (parsed.action === 'PRESS') {
        const key = parsed.args[0] as string;
        await runtime.pressKey(key);
        const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
        return {
          stepId: stepNum,
          goal: plannerAction.intent || `Press ${key}`,
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: `PRESS(${key})`,
          llmResponseText: executorResp?.content,
          verificationPassed,
          usedVision: shouldUseVision,
          durationMs: Date.now() - stepStart,
        };
      }

      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        llmResponseText: executorResp?.content,
        verificationPassed: false,
        usedVision: shouldUseVision,
        durationMs: Date.now() - stepStart,
        error: `Unknown action type: ${parsed.action}`,
      };
    } catch (e) {
      return {
        stepId: stepNum,
        goal: plannerAction.intent || plannerAction.action,
        status: StepStatus.FAILED,
        llmResponseText: executorResp?.content,
        verificationPassed: false,
        usedVision: shouldUseVision,
        durationMs: Date.now() - stepStart,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  }

  // ---------------------------------------------------------------------------
  // Snapshot with Escalation
  // ---------------------------------------------------------------------------

  /**
   * Take snapshot with progressive limit escalation and optional scroll-to-find.
   *
   * Starts with base limit and increases on element-not-found scenarios
   * to capture more of the page. After exhausting limit escalation, if
   * scrollAfterEscalation is enabled, scrolls down/up to find elements
   * that may be outside the current viewport.
   *
   * @param runtime - Browser runtime for snapshots and scrolling
   * @param goal - Goal string for context formatting
   * @param step - Optional step info for intent-based element detection during scroll
   */
  private async snapshotWithEscalation(
    runtime: AgentRuntime,
    goal: string,
    step?: { action: string; intent?: string; relaxPruning?: boolean }
  ): Promise<SnapshotContext> {
    const cfg = this.config.snapshot;
    const captureScreenshot = this.executor.supportsVision();
    let currentLimit = cfg.limitBase;
    const maxLimit = cfg.enabled ? cfg.limitMax : cfg.limitBase;
    let lastSnapshot: Awaited<ReturnType<AgentRuntime['snapshot']>> = null;
    let lastCompact = '';
    let screenshotBase64: string | null = null;
    let requiresVision = false;
    let visionReason: string | null = null;
    let pruningCategory: string | null = null;
    let prunedNodeCount = 0;
    let previousElementCount = -1;

    // Phase 1: Limit escalation loop
    while (currentLimit <= maxLimit) {
      try {
        const snap = await runtime.snapshot({
          limit: currentLimit,
          screenshot: captureScreenshot,
          goal,
        });

        if (snap === null) {
          if (!cfg.enabled) break;
          currentLimit = Math.min(currentLimit + cfg.limitStep, maxLimit + 1);
          continue;
        }

        lastSnapshot = snap;
        lastCompact = formatContext(snap.elements || [], currentLimit);
        screenshotBase64 = typeof snap.screenshot === 'string' ? snap.screenshot : null;
        let actionableContextCount = selectContextElements(
          snap.elements || [],
          currentLimit
        ).length;

        const elementCount = snap.elements?.length || 0;

        // Early exit: widening the limit didn't discover new elements
        if (elementCount > 0 && elementCount === previousElementCount) {
          if (this.config.verbose) {
            console.log(
              `[ESCALATION] Element count unchanged (${elementCount}), stopping escalation`
            );
          }
          break;
        }
        previousElementCount = elementCount;

        const visionResult = detectSnapshotFailure(snap);
        if (this.config.verbose) {
          console.log(
            `[VISION CHECK] elements=${elementCount}, shouldUseVision=${visionResult.shouldUseVision}, reason=${visionResult.reason}, screenshot=${screenshotBase64 ? 'yes' : 'no'}, executor.supportsVision=${this.executor.supportsVision()}`
          );
        }
        if (visionResult.shouldUseVision) {
          requiresVision = true;
          visionReason = visionResult.reason;
          break;
        }

        if (cfg.pruningEnabled) {
          const effectiveCategory = detectPruningCategory(
            this.currentTaskCategory,
            step?.intent || goal
          );
          if (effectiveCategory) {
            const pruned = pruneWithRecovery(snap, {
              goal: step?.intent || goal,
              category: effectiveCategory,
              relaxationLevel: step?.relaxPruning ? 1 : 0,
              minElementCount: cfg.pruningMinElements,
              maxRelaxation: cfg.pruningMaxRelaxation,
              profilePolicy: this.resolvedProfile?.pruningPolicy,
              learnedFingerprints: this.resolvedProfile?.learnedFingerprints,
            });
            pruningCategory = pruned.category;
            prunedNodeCount = pruned.actionableElementCount;
            lastCompact = pruned.promptBlock;
            actionableContextCount = pruned.actionableElementCount;
          }
        }

        // If escalation disabled, we're done after first successful snapshot
        if (!cfg.enabled) break;

        // Check element count - if sufficient, no need to escalate
        const hasEnoughContext = pruningCategory
          ? actionableContextCount >= cfg.pruningMinElements
          : elementCount >= 10;
        if (hasEnoughContext) break;

        // Escalate limit
        if (currentLimit < maxLimit) {
          currentLimit = Math.min(currentLimit + cfg.limitStep, maxLimit);
          if (this.config.verbose) {
            console.log(
              `[ESCALATION] Low element count (${elementCount}), increasing limit to ${currentLimit}`
            );
          }
        } else {
          break;
        }
      } catch (e) {
        if (this.config.verbose) {
          console.error(`[SNAPSHOT] Error: ${e}`);
        }
        if (!cfg.enabled) break;
        currentLimit = Math.min(currentLimit + cfg.limitStep, maxLimit + 1);
      }
    }

    // Phase 2: Scroll-after-escalation
    // Only trigger for CLICK actions with specific intents
    // Skip entirely if the page has fewer elements than limitBase (nothing below the fold)
    const pageElementCount = lastSnapshot?.elements?.length ?? 0;
    const shouldTryScroll =
      cfg.scrollAfterEscalation &&
      step !== undefined &&
      lastSnapshot !== null &&
      !requiresVision &&
      step.action === 'CLICK' &&
      step.intent &&
      pageElementCount >= cfg.limitBase;

    if (shouldTryScroll && lastSnapshot) {
      // Check if we can find the target element using intent heuristics
      const elements = lastSnapshot.elements || [];
      const url = lastSnapshot.url || '';
      let foundElement = this.tryIntentHeuristics(step.intent!, elements, url, goal);

      if (foundElement === null) {
        // Element not found in current viewport - try scrolling
        if (this.config.verbose) {
          console.log(
            `[SNAPSHOT-ESCALATION] Target element not found for intent "${step.intent}", trying scroll-after-escalation...`
          );
        }

        // Get viewport height and calculate scroll delta
        const viewportHeight = await runtime.getViewportHeight();
        const scrollDelta = viewportHeight * cfg.scrollViewportFraction;

        for (const direction of cfg.scrollDirections) {
          // Map direction to dy (pixels): down=positive, up=negative
          const scrollDy = direction === 'down' ? scrollDelta : -scrollDelta;

          for (let scrollNum = 0; scrollNum < cfg.scrollMaxAttempts; scrollNum++) {
            if (this.config.verbose) {
              console.log(
                `[SNAPSHOT-ESCALATION] Scrolling ${direction} (${scrollNum + 1}/${cfg.scrollMaxAttempts})...`
              );
            }

            // Scroll with verification
            const scrollEffective = await runtime.scrollBy(scrollDy);

            if (!scrollEffective) {
              if (this.config.verbose) {
                console.log(
                  `[SNAPSHOT-ESCALATION] Scroll ${direction} had no effect (reached boundary), skipping remaining attempts`
                );
              }
              break; // No point trying more scrolls in this direction
            }

            // Wait for stabilization after successful scroll
            if (cfg.scrollStabilizeMs > 0) {
              await new Promise(resolve => setTimeout(resolve, cfg.scrollStabilizeMs));
            }

            // Take new snapshot at max limit (we already escalated)
            try {
              const snap = await runtime.snapshot({
                limit: cfg.limitMax,
                screenshot: captureScreenshot,
                goal,
              });

              if (snap === null) continue;

              lastSnapshot = snap;
              lastCompact = formatContext(snap.elements || [], cfg.limitMax);
              screenshotBase64 = typeof snap.screenshot === 'string' ? snap.screenshot : null;
              let actionableContextCount = selectContextElements(
                snap.elements || [],
                cfg.limitMax
              ).length;

              if (cfg.pruningEnabled) {
                const effectiveCategory = detectPruningCategory(
                  this.currentTaskCategory,
                  step?.intent || goal
                );
                if (effectiveCategory) {
                  const pruned = pruneWithRecovery(snap, {
                    goal: step?.intent || goal,
                    category: effectiveCategory,
                    relaxationLevel: step?.relaxPruning ? 1 : 0,
                    minElementCount: cfg.pruningMinElements,
                    maxRelaxation: cfg.pruningMaxRelaxation,
                    profilePolicy: this.resolvedProfile?.pruningPolicy,
                    learnedFingerprints: this.resolvedProfile?.learnedFingerprints,
                  });
                  pruningCategory = pruned.category;
                  prunedNodeCount = pruned.actionableElementCount;
                  lastCompact = pruned.promptBlock;
                  actionableContextCount = pruned.actionableElementCount;
                }
              }

              // Check if target element is now visible
              const newElements = snap.elements || [];
              const newUrl = snap.url || '';
              foundElement = this.tryIntentHeuristics(step.intent!, newElements, newUrl, goal);

              if (foundElement !== null) {
                if (this.config.verbose) {
                  console.log(
                    `[SNAPSHOT-ESCALATION] Found target element ${foundElement} after scrolling ${direction}`
                  );
                }
                break; // Break out of scroll attempts loop
              }

              if (pruningCategory && actionableContextCount >= cfg.pruningMinElements) {
                break;
              }
            } catch {
              continue;
            }
          }

          // If found, break out of direction loop
          if (foundElement !== null) break;
        }

        if (foundElement === null && this.config.verbose) {
          console.log(`[SNAPSHOT-ESCALATION] Target element not found after scrolling`);
        }
      }
    }

    // Fallback for failed capture
    if (lastSnapshot === null) {
      lastSnapshot = { url: '', title: '', elements: [] };
      requiresVision = true;
      visionReason = 'snapshot_capture_failed';
    }

    if (this.tracer && visionReason) {
      this.tracer.emit(
        'vision_decision',
        {
          step_index: this.currentStepIndex,
          goal: step?.intent || goal,
          details: {
            use_vision: this.executor.supportsVision() && Boolean(screenshotBase64),
            reason: this.normalizeVisionTraceReason(visionReason),
            action: step?.action,
            intent: step?.intent,
            limit_used: currentLimit,
          },
        },
        this.getTraceStepId()
      );
    }

    if (this.config.verbose) {
      console.log(
        `[SNAPSHOT CONTEXT] requiresVision=${requiresVision}, visionReason=${visionReason}, screenshotBase64=${screenshotBase64 != null ? `present(${screenshotBase64.length}chars)` : 'null'}, executor.supportsVision=${this.executor.supportsVision()}, elements=${lastSnapshot?.elements?.length || 0}`
      );
    }

    return {
      snapshot: lastSnapshot,
      compactRepresentation: lastCompact,
      screenshotBase64,
      capturedAt: new Date(),
      limitUsed: currentLimit,
      snapshotSuccess: !requiresVision,
      requiresVision,
      visionReason,
      pruningCategory,
      prunedNodeCount,
    };
  }

  // ---------------------------------------------------------------------------
  // Intent Heuristics Helper
  // ---------------------------------------------------------------------------

  /**
   * Try to find an element using intent heuristics.
   *
   * @param intent - Intent string (e.g., "add_to_cart")
   * @param elements - Available elements from snapshot
   * @param url - Current page URL
   * @param goal - Step goal description
   * @returns Element ID if found, null otherwise
   */
  private tryIntentHeuristics(
    intent: string,
    elements: SnapshotElement[],
    url: string,
    goal: string
  ): number | null {
    try {
      return this.composableHeuristics.findElementForIntent(intent, elements, url, goal);
    } catch {
      return null;
    }
  }

  private resolveHeuristicAction(
    plannerAction: StepwisePlannerResponse,
    ctx: SnapshotContext,
    task: string
  ): ParsedAction | null {
    if (!plannerAction.intent || !ctx.snapshot) {
      return null;
    }

    const elementId = this.tryIntentHeuristics(
      plannerAction.intent,
      ctx.snapshot.elements || [],
      ctx.snapshot.url || '',
      task
    );
    if (elementId === null) {
      return null;
    }

    if (plannerAction.action === 'CLICK') {
      return { action: 'CLICK', args: [elementId] };
    }

    const matchedElement = ctx.snapshot.elements.find(element => element.id === elementId) || null;
    if (
      plannerAction.action === 'TYPE_AND_SUBMIT' &&
      plannerAction.input &&
      isSearchLikeTypeAndSubmit(plannerAction, matchedElement)
    ) {
      return { action: 'TYPE', args: [elementId, plannerAction.input] };
    }

    if (plannerAction.action === 'TYPE' && plannerAction.input) {
      return { action: 'TYPE', args: [elementId, plannerAction.input] };
    }

    return null;
  }

  private async resolveExecutorAction(
    plannerAction: StepwisePlannerResponse,
    ctx: SnapshotContext,
    task: string,
    forceVision: boolean = false
  ): Promise<{ parsed: ParsedAction; shouldUseVision: boolean; executorResp: LLMResponse | null }> {
    const lowElementHtmlSnapshot =
      ctx.requiresVision &&
      (ctx.visionReason === 'below_threshold' || ctx.visionReason === 'too_few_elements');
    const typingAction =
      plannerAction.action === 'TYPE' || plannerAction.action === 'TYPE_AND_SUBMIT';
    const hasTextElements = (ctx.snapshot?.elements?.length || 0) > 0;
    const allowHeuristicsDespiteVision =
      (typingAction && (lowElementHtmlSnapshot || hasTextElements)) ||
      Boolean(plannerAction.heuristicHints?.length);
    const heuristicAction =
      (!ctx.requiresVision || allowHeuristicsDespiteVision) && plannerAction.intent
        ? this.resolveHeuristicAction(plannerAction, ctx, task)
        : null;

    const [execSystem, execUser] = buildExecutorPrompt(
      plannerAction.intent || `${plannerAction.action} element`,
      plannerAction.intent,
      ctx.compactRepresentation,
      plannerAction.input,
      this.currentTaskCategory || undefined,
      plannerAction.action
    );

    if (this.config.verbose) {
      console.log(`[EXECUTOR PROMPT] system len=${execSystem.length}, user len=${execUser.length}`);
      console.log(`[EXECUTOR USER PROMPT (first 300)]:\n${execUser.slice(0, 300)}...`);
    }

    const shouldUseVision =
      (forceVision || ctx.requiresVision) &&
      this.executor.supportsVision() &&
      Boolean(ctx.screenshotBase64);
    let executorResp: LLMResponse | null = null;

    if (heuristicAction === null) {
      if (shouldUseVision) {
        const [visSystem, visUser] = buildVisionExecutorPrompt(
          plannerAction.intent || `${plannerAction.action} element`,
          plannerAction.intent,
          plannerAction.input,
          plannerAction.action
        );
        executorResp = await this.executor.generateWithImage(
          visSystem,
          visUser,
          ctx.screenshotBase64!,
          {
            temperature: this.config.executorTemperature,
            max_tokens: this.config.executorMaxTokens,
          }
        );
        this.recordTokenUsage('vision', executorResp);
      } else {
        executorResp = await this.executor.generate(execSystem, execUser, {
          temperature: this.config.executorTemperature,
          max_tokens: this.config.executorMaxTokens,
        });
        this.recordTokenUsage('executor', executorResp);
      }
    }

    if (executorResp && this.config.verbose) {
      const rawLen = executorResp.content.length;
      const hasThink = executorResp.content.includes('<think>');
      const displayContent =
        rawLen > 500
          ? executorResp.content.slice(0, 500) + `... (${rawLen} chars total)`
          : executorResp.content;
      console.log(
        `[EXECUTOR RAW]${hasThink ? ' (has <think>)' : ''} len=${rawLen}:\n${displayContent}`
      );
    }

    return {
      parsed: heuristicAction ?? parseAction(executorResp?.content ?? 'NONE'),
      shouldUseVision,
      executorResp,
    };
  }

  // ---------------------------------------------------------------------------
  // Verification / Recovery / Modal Helpers
  // ---------------------------------------------------------------------------

  private normalizePlannerAction(raw: Record<string, unknown>): StepwisePlannerResponse {
    const normalized = normalizePlan({
      steps: [{ id: 1, goal: 'stepwise_action', required: true, verify: [], ...raw }],
    });
    const step = Array.isArray(normalized.steps)
      ? (normalized.steps[0] as Record<string, unknown>)
      : raw;
    const normalizedAction =
      typeof step.action === 'string'
        ? step.action
        : typeof raw.action === 'string'
          ? raw.action
          : 'DONE';

    const isHumanReadable = (s: string) => !s.startsWith('http') && s !== 'stepwise_action';

    const normalizedGoal =
      typeof step.goal === 'string' && isHumanReadable(step.goal)
        ? step.goal
        : typeof step.intent === 'string'
          ? step.intent
          : typeof raw.goal === 'string' && isHumanReadable(raw.goal)
            ? raw.goal
            : typeof raw.intent === 'string'
              ? raw.intent
              : typeof step.target === 'string'
                ? step.target
                : undefined;

    return {
      id: typeof step.id === 'number' ? step.id : undefined,
      goal: normalizedGoal,
      action: normalizedAction.toUpperCase() as StepwisePlannerResponse['action'],
      target: typeof step.target === 'string' ? step.target : undefined,
      intent: typeof step.intent === 'string' ? step.intent : undefined,
      input: typeof step.input === 'string' ? step.input : undefined,
      direction: step.direction === 'up' || step.direction === 'down' ? step.direction : undefined,
      verify: Array.isArray(step.verify)
        ? (step.verify as Array<{ predicate: string; args: unknown[] }>)
        : [],
      required: typeof step.required === 'boolean' ? step.required : undefined,
      stopIfTrue: typeof step.stopIfTrue === 'boolean' ? step.stopIfTrue : undefined,
      optionalSubsteps: Array.isArray(step.optionalSubsteps)
        ? (step.optionalSubsteps as Array<Record<string, unknown>>)
        : [],
      heuristicHints: Array.isArray(step.heuristicHints)
        ? (step.heuristicHints as Array<Record<string, unknown>>)
        : [],
      reasoning: typeof step.reasoning === 'string' ? step.reasoning : undefined,
      fields: Array.isArray(step.fields)
        ? (step.fields as Array<{ label: string; value: string }>)
        : undefined,
      submitText: typeof step.submitText === 'string' ? step.submitText : undefined,
    };
  }

  private isCopiedPlaceholderNavigation(
    targetUrl: string,
    currentUrl: string,
    task: string
  ): boolean {
    if (!this.isExampleDotComUrl(targetUrl)) {
      return false;
    }

    if (this.isExampleDotComUrl(currentUrl) || /\bexample\.com\b/i.test(task)) {
      return false;
    }

    return true;
  }

  private isExampleDotComUrl(url: string): boolean {
    try {
      const hostname = new URL(url).hostname.toLowerCase();
      return hostname === 'example.com' || hostname.endsWith('.example.com');
    } catch {
      return false;
    }
  }

  private promoteVisibleResultClick(
    task: string,
    ctx: SnapshotContext,
    plannerAction: StepwisePlannerResponse
  ): StepwisePlannerResponse {
    if (plannerAction.action !== 'SCROLL' && plannerAction.action !== 'WAIT') {
      return plannerAction;
    }

    if (isExtractionTask(task)) {
      return plannerAction;
    }

    const candidate = this.findVisibleResultLink(task, ctx.snapshot);
    if (!candidate) {
      return plannerAction;
    }

    const label = this.elementLabel(candidate);
    const hrefVerify = this.hrefVerificationSignal(candidate.href || '', ctx.snapshot?.url || '');

    return {
      ...plannerAction,
      action: 'CLICK',
      goal: plannerAction.goal || 'Open visible result link',
      intent: 'visible result link',
      input: label || plannerAction.input,
      verify: hrefVerify ? [{ predicate: 'url_contains', args: [hrefVerify] }] : [],
      heuristicHints: [
        {
          intent_pattern: 'visible_result_link',
          text_patterns: label ? [label] : [],
          role_filter: ['link'],
          priority: 20,
        },
      ],
      reasoning:
        plannerAction.reasoning ||
        'Visible result link matched the task goal; clicking it is more direct than scrolling.',
    };
  }

  private findVisibleResultLink(
    task: string,
    snapshot: Snapshot | null | undefined
  ): SnapshotElement | null {
    const elements = snapshot?.elements || [];
    if (elements.length === 0 || !this.taskWantsResultNavigation(task)) {
      return null;
    }

    const candidates = elements
      .filter(element => this.isResultNavigationLink(element))
      .sort((left, right) => (right.importance || 0) - (left.importance || 0));

    return candidates[0] || null;
  }

  private taskWantsResultNavigation(task: string): boolean {
    const normalized = task.toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ');
    const actionCue = /\b(click|open|pick|choose|select|go to|visit)\b/.test(normalized);
    const targetCue = /\b(product|result|item|listing|detail page|details page)\b/.test(normalized);
    return actionCue && targetCue;
  }

  private isResultNavigationLink(element: SnapshotElement): boolean {
    const role = (element.role || '').toLowerCase();
    const href = (element.href || '').toLowerCase();
    if (role !== 'link' || !href) {
      return false;
    }

    if (/\/(?:dp|gp\/product|product|products|item|items|p)\//.test(href)) {
      return true;
    }

    const label = this.elementLabel(element);
    return Boolean(element.inDominantGroup && label.length >= 15);
  }

  private elementLabel(element: SnapshotElement): string {
    return (element.text || element.ariaLabel || element.name || '').trim();
  }

  private hrefVerificationSignal(href: string, baseUrl: string): string | null {
    if (!href.trim()) {
      return null;
    }

    try {
      const parsed = new URL(href, baseUrl || undefined);
      if (parsed.pathname && parsed.pathname !== '/') {
        return parsed.pathname;
      }
      return parsed.hostname || null;
    } catch {
      return href.startsWith('/') ? href : null;
    }
  }

  private summarizePlannerActionTarget(plannerAction: StepwisePlannerResponse): string | null {
    if (plannerAction.action === 'FILL_FORM') {
      const fields = plannerAction.fields || [];
      return fields.map(f => `${f.label}=${f.value}`).join(', ') || null;
    }
    if (plannerAction.action === 'TYPE' || plannerAction.action === 'TYPE_AND_SUBMIT') {
      return plannerAction.input || plannerAction.intent || plannerAction.target || null;
    }

    return plannerAction.intent || plannerAction.input || plannerAction.target || null;
  }

  private isPreviouslyCompletedFormDataEntry(
    plannerAction: StepwisePlannerResponse,
    currentUrl: string
  ): boolean {
    if (!plannerAction.intent) {
      return false;
    }

    const action = plannerAction.action;
    if (!['TYPE', 'TYPE_AND_SUBMIT', 'CLICK', 'FILL_FORM'].includes(action)) {
      return false;
    }

    const intentKey = this.normalizeCompletedFormIntent(plannerAction.intent);
    if (!intentKey || !this.isFormDataEntryIntent(intentKey)) {
      return false;
    }

    return this.actionHistory.some(record => {
      if (!['success', 'skipped', 'vision_fallback'].includes(record.result)) {
        return false;
      }
      const recordIntent = record.intent ? this.normalizeCompletedFormIntent(record.intent) : '';
      if (!this.isFormDataEntryIntent(recordIntent)) {
        return false;
      }
      if (!this.completedFormIntentsMatch(recordIntent, intentKey)) {
        return false;
      }
      if (record.urlAfter && currentUrl) {
        return (
          this.normalizeNavigationUrl(record.urlAfter) === this.normalizeNavigationUrl(currentUrl)
        );
      }
      return true;
    });
  }

  private isFormDataEntryPlannerAction(plannerAction: StepwisePlannerResponse): boolean {
    const intentKey = plannerAction.intent
      ? this.normalizeCompletedFormIntent(plannerAction.intent)
      : '';
    return Boolean(intentKey && this.isFormDataEntryIntent(intentKey));
  }

  private isForwardNavigationPlannerAction(plannerAction: StepwisePlannerResponse): boolean {
    if (plannerAction.action !== 'CLICK') {
      return false;
    }

    const actionText = `${plannerAction.intent || ''} ${plannerAction.input || ''} ${
      plannerAction.target || ''
    }`;
    return /\b(next|continue)\b/i.test(actionText);
  }

  private normalizeCompletedFormIntent(intent: string): string {
    return intent
      .toLowerCase()
      .replace(
        /\b(field|input|textbox|text box|radio button|checkbox|dropdown|select|button)\b/g,
        ''
      )
      .replace(/\s+/g, ' ')
      .trim();
  }

  private completedFormIntentsMatch(previousIntent: string, nextIntent: string): boolean {
    if (!previousIntent || !nextIntent) {
      return false;
    }
    if (previousIntent === nextIntent) {
      return true;
    }

    const previousCategories = this.completedFormIntentCategories(previousIntent);
    const nextCategories = this.completedFormIntentCategories(nextIntent);
    return [...previousCategories].some(category => nextCategories.has(category));
  }

  private completedFormIntentCategories(intent: string): Set<string> {
    const categories = new Set<string>();
    if (/\bemail\b/.test(intent)) {
      categories.add('email');
    }
    if (/\bdisplay\b/.test(intent) || /\bname\b/.test(intent)) {
      categories.add('display_name');
    }
    if (/\bplan\b/.test(intent) || /\bpro\b/.test(intent)) {
      categories.add('plan');
    }
    if (/\bterms?\b/.test(intent) || /\bagree\b/.test(intent) || /\bconsent\b/.test(intent)) {
      categories.add('terms');
    }
    return categories;
  }

  private isFormDataEntryIntent(intent: string): boolean {
    if (/\b(next|continue|submit|done|finish|back|previous)\b/.test(intent)) {
      return false;
    }

    return /\b(email|display|name|plan|pro|terms?|agree|consent|checkbox|radio|dropdown)\b/.test(
      intent
    );
  }

  private classifyStepFailure(
    plannerAction: StepwisePlannerResponse,
    outcome: StepOutcome,
    currentUrl: string
  ): RepairFailureCategory {
    const error = (outcome.error || '').toLowerCase();

    if (error.includes('failed to parse')) {
      return 'parse-failure';
    }

    if (
      error.includes('could not find suitable element') ||
      error.includes('no executor response') ||
      error.includes('planner returned empty response')
    ) {
      return 'element-not-found';
    }

    if (
      error.includes('auth') ||
      error.includes('credential') ||
      error.includes('login') ||
      error.includes('policy') ||
      error.includes('recover')
    ) {
      return 'auth-or-recovery';
    }

    if (
      this.config.authBoundary.enabled &&
      detectAuthBoundary(currentUrl, this.config.authBoundary).isAuthBoundary
    ) {
      return 'auth-or-recovery';
    }

    if (!outcome.verificationPassed || plannerAction.action === 'STUCK') {
      return 'verification-failed';
    }

    return 'element-not-found';
  }

  private async requestRepairSteps(
    task: string,
    currentUrl: string,
    failedStep: StepwisePlannerResponse,
    outcome: StepOutcome,
    repairHistory: RepairHistoryEntry[]
  ): Promise<StepwisePlannerResponse[]> {
    const failureCategory = this.classifyStepFailure(failedStep, outcome, currentUrl);
    this.tracer?.emit(
      'replan',
      {
        step_index: this.currentStepIndex,
        goal: failedStep.goal || failedStep.intent || failedStep.action,
        details: {
          phase: 'start',
          failure_category: failureCategory,
          failed_action: failedStep.action,
          reason: outcome.error || 'step failed',
        },
      },
      this.getTraceStepId()
    );
    try {
      const [repairSystem, repairUser] = buildRepairPlannerPrompt({
        task,
        currentUrl,
        failedStep,
        failureReason: outcome.error || 'verification_failed',
        failureCategory,
        actionHistory: this.actionHistory.slice(-this.config.stepwise.actionHistoryLimit),
        repairHistory,
      });

      const repairResp = await this.planner.generate(repairSystem, repairUser, {
        temperature: this.config.plannerTemperature,
        max_tokens: this.config.plannerMaxTokens,
      });
      this.recordTokenUsage('replan', repairResp);

      const normalizedPatch = normalizeReplanPatch(extractJson(repairResp.content));
      const patch = ReplanPatchSchema.parse(normalizedPatch);

      const replacementSteps = [...patch.replaceSteps]
        .sort((a, b) => a.id - b.id)
        .map(item => this.normalizePlannerAction(item.step as unknown as Record<string, unknown>));

      if (replacementSteps.length === 0) {
        throw new Error('Repair planner returned no replacement steps');
      }

      this.tracer?.emit(
        'replan',
        {
          step_index: this.currentStepIndex,
          goal: failedStep.goal || failedStep.intent || failedStep.action,
          success: true,
          details: {
            phase: 'result',
            replacement_step_count: replacementSteps.length,
          },
        },
        this.getTraceStepId()
      );

      return replacementSteps;
    } catch (repairError) {
      const reason = repairError instanceof Error ? repairError.message : String(repairError);
      this.tracer?.emit(
        'replan',
        {
          step_index: this.currentStepIndex,
          goal: failedStep.goal || failedStep.intent || failedStep.action,
          success: false,
          details: {
            phase: 'result',
            failure_category: failureCategory,
            error: reason,
          },
        },
        this.getTraceStepId()
      );
      throw repairError;
    }
  }

  private async executeOptionalSubsteps(
    runtime: AgentRuntime,
    plannerAction: StepwisePlannerResponse,
    stepNum: number,
    task: string
  ): Promise<StepOutcome[]> {
    if (!plannerAction.optionalSubsteps || plannerAction.optionalSubsteps.length === 0) {
      return [];
    }

    const substepOutcomes: StepOutcome[] = [];
    for (const [substepIndex, substepRaw] of plannerAction.optionalSubsteps.entries()) {
      const substep = this.normalizePlannerAction(substepRaw);
      const substepId =
        typeof substep.id === 'number' ? substep.id : stepNum * 100 + substepIndex + 1;
      const substepStart = Date.now();
      const substepCtx = await this.snapshotWithEscalation(runtime, substep.goal || task, {
        action: substep.action,
        intent: substep.intent,
      });
      const substepOutcome = await this.executeStepwiseAction(
        runtime,
        substep,
        substepId,
        task,
        substepCtx,
        substepStart
      );
      substepOutcomes.push(substepOutcome);
      this.actionHistory.push({
        stepNum: substepId,
        action: substep.action,
        target: this.summarizePlannerActionTarget(substep),
        intent: substep.intent || null,
        result: this.actionHistoryResult(substepOutcome.status),
        urlAfter: substepOutcome.urlAfter || (await runtime.getCurrentUrl()),
      });
    }

    return substepOutcomes;
  }

  private actionHistoryResult(status: StepStatus): string {
    if (
      status === StepStatus.SUCCESS ||
      status === StepStatus.SKIPPED ||
      status === StepStatus.VISION_FALLBACK
    ) {
      return status;
    }
    return 'failed';
  }

  private async checkPreStepVerification(
    runtime: AgentRuntime,
    plannerAction: StepwisePlannerResponse
  ): Promise<boolean> {
    if (!plannerAction.verify || plannerAction.verify.length === 0) {
      return false;
    }

    try {
      const snap = await runtime.snapshot({
        limit: 30,
        screenshot: false,
        goal: plannerAction.intent || plannerAction.action,
      });
      if (!snap) {
        return false;
      }
      return evaluatePredicates(plannerAction.verify, snap);
    } catch {
      return false;
    }
  }

  private async verifyStepOutcome(
    runtime: AgentRuntime,
    plannerAction: StepwisePlannerResponse
  ): Promise<boolean> {
    if (!plannerAction.verify || plannerAction.verify.length === 0) {
      return true;
    }

    const timeoutMs = Math.max(0, this.config.retry.verifyTimeoutMs);
    const pollMs = Math.max(1, this.config.retry.verifyPollMs);
    const start = Date.now();

    const allUrlPredicates = plannerAction.verify.every(
      p =>
        p.predicate === 'url_contains' ||
        p.predicate === 'url_equals' ||
        p.predicate === 'url_matches'
    );

    while (Date.now() - start <= timeoutMs) {
      try {
        if (allUrlPredicates) {
          const url = await runtime.getCurrentUrl();
          const pseudoSnap: Snapshot = { url, title: '', elements: [] };
          if (evaluatePredicates(plannerAction.verify, pseudoSnap)) {
            return true;
          }
        } else {
          const snap = await runtime.snapshot({
            limit: this.config.snapshot.limitBase,
            screenshot: false,
            goal: plannerAction.intent || plannerAction.action,
          });
          if (snap && evaluatePredicates(plannerAction.verify, snap)) {
            return true;
          }
        }
      } catch {
        // Keep polling until timeout.
      }

      await new Promise(resolve => setTimeout(resolve, pollMs));
    }

    return false;
  }

  private clickedHrefMatchesNavigation(
    previousUrl: string,
    nextUrl: string,
    element: SnapshotElement | null
  ): boolean {
    const href = element?.href?.trim();
    if (!href) {
      return false;
    }

    try {
      const expected = new URL(href, previousUrl);
      const actual = new URL(nextUrl, previousUrl);
      if (
        !['http:', 'https:'].includes(expected.protocol) ||
        !['http:', 'https:'].includes(actual.protocol)
      ) {
        return false;
      }

      expected.hash = '';
      actual.hash = '';
      expected.search = '';
      actual.search = '';

      return (
        this.normalizeNavigationUrl(expected.toString()) ===
        this.normalizeNavigationUrl(actual.toString())
      );
    } catch {
      return false;
    }
  }

  private clickIntentRequiresNavigation(
    plannerAction: StepwisePlannerResponse,
    targetElement: SnapshotElement | null
  ): boolean {
    if ((plannerAction.verify?.length || 0) > 0) {
      return true;
    }
    const intentText = `${plannerAction.intent || ''} ${plannerAction.input || ''}`.toLowerCase();
    if (/\b(product|result|detail|item|open|visit|page|link)\b/.test(intentText)) {
      return true;
    }
    const href = (targetElement?.href || '').trim().toLowerCase();
    if (!href) {
      return false;
    }
    return !href.startsWith('#') && !href.startsWith('javascript:');
  }

  private detectActionCycle(history: ActionRecord[]): string | null {
    if (history.length < 4) {
      return null;
    }

    const recent = history.slice(-6);

    // Don't flag EXTRACT-only loops here — the extraction auto-complete
    // or max-steps limit handles those. EXTRACT doesn't navigate so
    // repeating it isn't a "cycle" in the navigation sense.
    const nonExtractRecent = recent.filter(entry => entry.action !== 'EXTRACT');
    if (nonExtractRecent.length < 4) {
      return null;
    }

    const urlsAfter = nonExtractRecent
      .filter(entry => entry.urlAfter)
      .map(entry => entry.urlAfter!);

    // Same-page interaction is not a navigation cycle. Multi-step forms,
    // SPAs, and paginated lists all stay on the same URL while the agent
    // makes forward progress. Only flag cycles involving actual navigation
    // (different URLs visited repeatedly).
    const distinctUrls = new Set(urlsAfter);
    if (distinctUrls.size <= 1) {
      return null;
    }

    const hosts = urlsAfter.map(url => this.urlHostKey(url));

    if (hosts.length < 4) {
      return null;
    }

    const hostCounts = new Map<string, number>();
    for (const host of hosts) {
      hostCounts.set(host, (hostCounts.get(host) || 0) + 1);
    }

    for (const [host, count] of hostCounts) {
      if (count < 4) continue;

      const sameHostEntries = nonExtractRecent.filter(
        entry => entry.urlAfter && this.urlHostKey(entry.urlAfter) === host
      );
      const hasRepeatedAction = sameHostEntries.some(
        (entry, idx) => idx > 0 && entry.action === sameHostEntries[0].action
      );

      if (hasRepeatedAction) {
        return `Action cycle detected: visited ${host} ${count} times with repeated ${sameHostEntries[0].action} actions`;
      }
    }

    // Alternating host detection — only flag if the sequence actually alternates
    if (hostCounts.size === 2) {
      const counts = [...hostCounts.values()];
      if (counts.every(c => c >= 2) && hosts.length >= 4) {
        // Verify strict alternation: no two consecutive entries are the same host
        const strictlyAlternating = hosts.every((host, i) => i === 0 || host !== hosts[i - 1]);
        if (strictlyAlternating) {
          const [hostA, hostB] = [...hostCounts.keys()];
          return `Action cycle detected: alternating between ${hostA} and ${hostB} (${hosts.length} navigations)`;
        }
      }
    }

    return null;
  }

  private detectRepeatedCompletedFormSkip(stepOutcomes: StepOutcome[]): string | null {
    if (stepOutcomes.length < 3) {
      return null;
    }

    const recent = stepOutcomes.slice(-3);
    if (
      !recent.every(
        outcome =>
          outcome.status === StepStatus.SKIPPED &&
          outcome.actionTaken === 'SKIPPED(previously_completed_form_step)'
      )
    ) {
      return null;
    }

    const intentKeys = recent.map(outcome => this.normalizeCompletedFormIntent(outcome.goal || ''));
    if (intentKeys.some(intent => !intent || !this.isFormDataEntryIntent(intent))) {
      return null;
    }

    const firstIntent = intentKeys[0];
    const repeatedIntent = intentKeys.every(intent =>
      this.completedFormIntentsMatch(firstIntent, intent)
    );
    if (!repeatedIntent) {
      return null;
    }

    const urls = recent.map(outcome => outcome.urlAfter || outcome.urlBefore || '').filter(Boolean);
    const sameUrl =
      urls.length === recent.length &&
      new Set(urls.map(url => this.normalizeNavigationUrl(url))).size === 1;
    if (!sameUrl) {
      return null;
    }

    return `Planner repeated already completed form step "${recent[0].goal}" without moving to the next incomplete goal`;
  }

  private urlHostKey(url: string): string {
    try {
      const parsed = new URL(url);
      return parsed.hostname.replace(/^www\./, '');
    } catch {
      return url.split('/')[0] || url;
    }
  }

  private findFallbackNavigationClickTarget(
    elements: SnapshotElement[],
    attemptedElementId: number,
    plannerAction: StepwisePlannerResponse
  ): number | null {
    const terms = `${plannerAction.intent || ''} ${plannerAction.input || ''}`
      .toLowerCase()
      .split(/\s+/)
      .filter(term => term.length >= 3);

    const candidates = elements
      .filter(element => element.id !== attemptedElementId)
      .filter(element => {
        const role = (element.role || '').toLowerCase();
        return role === 'link' || Boolean(element.href);
      })
      .map(element => {
        const href = (element.href || '').toLowerCase();
        const text =
          `${element.text || ''} ${element.name || ''} ${element.ariaLabel || ''}`.toLowerCase();
        let score = 0;
        if (
          /\/(?:dp|gp\/product|product|products|item|items|sku|detail|details|listing|p)\b/.test(
            href
          )
        ) {
          score += 100;
        }
        if (/\/(?:s|search|results?)\b/.test(href)) {
          score += 20;
        }
        if (this.looksLikeNavChromeLink(href, text)) {
          score -= 80;
        }
        if (this.extractPathDepth(href) >= 2) {
          score += 10;
        }
        if (element.clickable) {
          score += 10;
        }
        if (terms.some(term => text.includes(term) || href.includes(term))) {
          score += 25;
        }
        if (text.length >= 12) {
          score += 5;
        }
        return { id: element.id, score };
      })
      .filter(candidate => candidate.score > 0)
      .sort((a, b) => b.score - a.score);

    return candidates.length > 0 ? candidates[0].id : null;
  }

  private looksLikeNavChromeLink(href: string, label: string): boolean {
    const chromeCues =
      /\b(home|logo|account|signin|sign in|login|help|customer service|returns?)\b/;
    return chromeCues.test(href) || chromeCues.test(label);
  }

  private extractPathDepth(href: string): number {
    if (!href) {
      return 0;
    }
    try {
      const parsed = new URL(href, 'https://placeholder.local');
      return parsed.pathname.split('/').filter(Boolean).length;
    } catch {
      return href.split('?')[0].split('/').filter(Boolean).length;
    }
  }

  private normalizeNavigationUrl(url: string): string {
    return url.trim().replace(/\/+$/, '').toLowerCase();
  }

  private async isCartAdditionTerminal(
    runtime: AgentRuntime,
    task: string,
    plannerAction: StepwisePlannerResponse
  ): Promise<boolean> {
    const taskText = task.toLowerCase();
    if (
      !/\badd(?:ed)?\b[\s\S]*\bcart\b|\bcart[_\s-]?addition\b/.test(taskText) ||
      /\bcheckout\b|\bcheck out\b|\bpayment\b|\bplace order\b|\bbuy now\b/.test(taskText)
    ) {
      return false;
    }

    const actionText = [
      plannerAction.intent,
      plannerAction.input,
      plannerAction.goal,
      plannerAction.action,
    ]
      .filter((value): value is string => typeof value === 'string')
      .join(' ')
      .toLowerCase()
      .replace(/[_-]+/g, ' ');

    if (!/\badd(?:ed)?\b[\s\S]*\bcart\b|\bcart contains\b/.test(actionText)) {
      return false;
    }

    try {
      const snap = await runtime.snapshot({
        limit: this.config.snapshot.limitBase,
        screenshot: false,
        goal: 'cart addition confirmation',
      });
      if (!snap) {
        return false;
      }

      return (snap.elements || []).some(element => {
        const label = [element.text, element.ariaLabel, element.name]
          .filter((value): value is string => typeof value === 'string')
          .join(' ')
          .toLowerCase();
        return (
          /\badded to (?:cart|bag|basket)\b/.test(label) ||
          /\bcart contains\s+[1-9]\d*\s+items?\b/.test(label) ||
          /\b[1-9]\d*\s+items?\s+in (?:your )?(?:cart|bag|basket)\b/.test(label)
        );
      });
    } catch {
      return false;
    }
  }

  private async attemptRecovery(runtime: AgentRuntime): Promise<boolean> {
    if (!this.recoveryState) {
      return false;
    }

    let checkpoint = this.recoveryState.consumeRecoveryAttempt();
    if (!checkpoint) {
      return false;
    }

    while (checkpoint) {
      const checkpointUrl = checkpoint.url;
      this.tracer?.emit(
        'recovery',
        {
          step_index: this.currentStepIndex,
          goal: 'recovery',
          details: {
            phase: 'attempt',
            checkpoint_url: checkpointUrl,
          },
        },
        this.getTraceStepId()
      );
      try {
        const currentUrl = await runtime.getCurrentUrl();
        const alreadyAtCheckpointUrl =
          this.normalizeNavigationUrl(currentUrl) === this.normalizeNavigationUrl(checkpoint.url);
        if (!alreadyAtCheckpointUrl) {
          await runtime.goto(checkpoint.url);
        }
        const verificationSnapshot = await runtime.snapshot({
          limit: this.config.snapshot.limitBase,
          screenshot: false,
          goal: 'recovery verification',
        });
        const recovered = verifyRecoveryCheckpoint(checkpoint, verificationSnapshot);
        if (!recovered && alreadyAtCheckpointUrl) {
          this.tracer?.emit(
            'recovery',
            {
              step_index: this.currentStepIndex,
              goal: 'recovery',
              success: false,
              details: {
                phase: 'skipped_same_url_navigation',
                checkpoint_url: checkpointUrl,
              },
            },
            this.getTraceStepId()
          );
          this.recoveryState.clearRecoveryTarget();
          return false;
        }
        this.tracer?.emit(
          'recovery',
          {
            step_index: this.currentStepIndex,
            goal: 'recovery',
            success: recovered,
            details: {
              phase: 'result',
              checkpoint_url: checkpointUrl,
            },
          },
          this.getTraceStepId()
        );
        if (recovered) {
          this.recoveryState.clearRecoveryTarget();
          return true;
        }
        this.recoveryState.popCheckpoint();
        checkpoint = this.recoveryState.getRecoveryTarget();
      } catch {
        this.tracer?.emit(
          'recovery',
          {
            step_index: this.currentStepIndex,
            goal: 'recovery',
            success: false,
            details: {
              phase: 'result',
              checkpoint_url: checkpointUrl,
            },
          },
          this.getTraceStepId()
        );
        this.recoveryState.popCheckpoint();
        checkpoint = this.recoveryState.getRecoveryTarget();
      }
    }

    this.recoveryState.clearRecoveryTarget();
    return false;
  }

  private async handlePostClickEffects(
    runtime: AgentRuntime,
    plannerAction: StepwisePlannerResponse,
    ctx: SnapshotContext
  ): Promise<{ handled: boolean }> {
    if (!this.config.modal.enabled || !ctx.snapshot) {
      return { handled: false };
    }

    const postSnap = await runtime.snapshot({
      limit: this.config.snapshot.limitMax,
      screenshot: false,
      goal: plannerAction.intent || plannerAction.action,
    });
    if (!postSnap) {
      return { handled: false };
    }

    const preElements = new Set((ctx.snapshot.elements || []).map(el => el.id));
    const postElements = new Set((postSnap.elements || []).map(el => el.id));
    if (!detectModalAppearance(preElements, postElements, this.config.modal.minNewElements)) {
      return { handled: false };
    }

    const modalElements = (postSnap.elements || []).filter(element => !preElements.has(element.id));
    const checkoutTarget = this.findCheckoutContinuationTarget(modalElements);
    if (checkoutTarget !== null) {
      if (!shouldAutoContinueCheckoutFlow(plannerAction.intent)) {
        // Drawer has checkout/cart controls but intent is unrelated — leave it alone.
        // Still report as handled since the modal appearance confirms the click worked.
        return { handled: true };
      }
      this.tracer?.emit(
        'modal_action',
        {
          step_index: this.currentStepIndex,
          goal: plannerAction.goal || plannerAction.intent || plannerAction.action,
          action: 'continue_checkout',
          element_id: checkoutTarget,
          details: {
            intent: plannerAction.intent,
            reason: 'checkout_control_detected',
          },
        },
        this.getTraceStepId()
      );
      await runtime.click(checkoutTarget);
      return { handled: true };
    }

    const dismissal = findDismissalTarget(modalElements, this.config.modal);
    if (!dismissal.found || dismissal.elementId === null) {
      return { handled: false };
    }

    this.tracer?.emit(
      'modal_action',
      {
        step_index: this.currentStepIndex,
        goal: plannerAction.goal || plannerAction.intent || plannerAction.action,
        action: 'dismiss',
        element_id: dismissal.elementId,
        details: {
          intent: plannerAction.intent,
          reason: 'dismissal_target_found',
        },
      },
      this.getTraceStepId()
    );
    await runtime.click(dismissal.elementId);

    const finalSnap = await runtime.snapshot({
      limit: this.config.snapshot.limitMax,
      screenshot: false,
      goal: plannerAction.intent || plannerAction.action,
    });
    if (!finalSnap) {
      return { handled: true };
    }

    detectModalDismissed(postElements, new Set((finalSnap.elements || []).map(el => el.id)));
    return { handled: true };
  }

  private findCheckoutContinuationTarget(elements: SnapshotElement[]): number | null {
    for (const element of elements) {
      const role = (element.role || '').toLowerCase();
      if (!['button', 'link'].includes(role)) {
        continue;
      }
      const labels = [element.text || '', element.ariaLabel || ''].filter(Boolean);
      if (labels.some(label => isCheckoutElement(label, DEFAULT_CHECKOUT_CONFIG))) {
        return element.id;
      }
    }
    return null;
  }

  private makeSnapshotDigest(url: string): string {
    let hash = 0;
    for (let i = 0; i < url.length; i++) {
      hash = (hash * 31 + url.charCodeAt(i)) >>> 0;
    }
    return hash.toString(16).padStart(8, '0');
  }

  // ---------------------------------------------------------------------------
  // Retry/Repair Helpers
  // ---------------------------------------------------------------------------

  /**
   * Wait for URL to change from the given URL.
   *
   * @param runtime - Browser runtime
   * @param originalUrl - URL to compare against
   * @param timeoutMs - Maximum wait time in milliseconds
   * @returns true if URL changed, false if timeout
   */
  private async waitForUrlChange(
    runtime: AgentRuntime,
    originalUrl: string,
    timeoutMs: number
  ): Promise<string | null> {
    const startTime = Date.now();
    const pollInterval = 500;

    while (Date.now() - startTime < timeoutMs) {
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      try {
        const currentUrl = await runtime.getCurrentUrl();
        if (currentUrl !== originalUrl) {
          return currentUrl;
        }
      } catch {
        // Ignore errors during URL check
      }
    }
    return null;
  }

  /**
   * Find a submit button near the input element.
   *
   * Looks for buttons/links with submit-related text that appear after
   * the input element in the DOM (higher element ID typically means
   * later in DOM order).
   *
   * @param elements - Snapshot elements
   * @param inputElementId - ID of the input element
   * @returns Submit button element ID if found, null otherwise
   */
  private findSubmitButton(
    elements: SnapshotElement[],
    inputElementId: number,
    searchLike: boolean
  ): number | null {
    // Submit-related patterns
    const submitPatterns = searchLike
      ? ['search', 'go', 'find', 'submit', 'apply', 'done', 'ok', 'send', 'enter']
      : ['next', 'continue', 'submit', 'save', 'send', 'sign in', 'log in', 'apply', 'ok', 'done'];

    // Icon patterns (exact match)
    const iconPatterns = ['>', '→', '🔍', '⌕'];

    // Look for submit buttons
    const candidates: Array<{ id: number; score: number }> = [];

    for (const element of elements) {
      // Only consider buttons and links
      const role = (element.role || '').toLowerCase();
      if (!['button', 'link'].includes(role)) continue;

      // Skip if not clickable
      if (element.clickable === false) continue;

      // Skip the input element itself
      if (element.id === inputElementId) continue;

      const text = (element.text || '').toLowerCase().trim();
      const ariaLabel = (element.ariaLabel || '').toLowerCase();

      // Check for icon patterns (exact match, high priority)
      for (const icon of iconPatterns) {
        if (text === icon || ariaLabel === icon) {
          const proximityBonus = 100 - Math.min(Math.abs(element.id - inputElementId), 100);
          candidates.push({ id: element.id, score: 200 + proximityBonus });
          break;
        }
      }

      // Check for submit patterns
      for (let i = 0; i < submitPatterns.length; i++) {
        const pattern = submitPatterns[i];
        if (text.includes(pattern) || ariaLabel.includes(pattern)) {
          // Score: pattern priority + proximity to input element
          // Lower distance from input = higher score
          const proximityBonus = 100 - Math.min(Math.abs(element.id - inputElementId), 100);
          candidates.push({ id: element.id, score: 100 - i + proximityBonus });
          break;
        }
      }
    }

    // Return best candidate (highest score)
    if (candidates.length === 0) return null;
    candidates.sort((a, b) => b.score - a.score);
    return candidates[0].id;
  }

  private findSubmitButtonByText(elements: SnapshotElement[], targetText: string): number | null {
    const lower = targetText.toLowerCase().trim();
    for (const el of elements) {
      const role = (el.role || '').toLowerCase();
      if (!['button', 'link'].includes(role)) continue;
      if (el.clickable === false) continue;
      const text = (el.text || '').toLowerCase().trim();
      const ariaLabel = (el.ariaLabel || '').toLowerCase().trim();
      if (
        text === lower ||
        text.includes(lower) ||
        lower.includes(text) ||
        ariaLabel.includes(lower)
      ) {
        return el.id;
      }
    }
    return null;
  }

  private inferSameUrlWizardProgressAfterNext(
    plannerAction: StepwisePlannerResponse,
    beforeSnapshot: Snapshot | null,
    afterSnapshot: Snapshot
  ): boolean {
    if (plannerAction.action !== 'TYPE' || !plannerAction.input) {
      return false;
    }

    const beforeElements = beforeSnapshot?.elements || [];
    const afterElements = afterSnapshot.elements || [];
    if (afterElements.length === 0) {
      return false;
    }

    const beforeSignature = this.formPaneSignature(beforeElements);
    const afterSignature = this.formPaneSignature(afterElements);
    if (!afterSignature || beforeSignature === afterSignature) {
      return false;
    }

    if (this.hasFormValidationError(afterElements)) {
      return false;
    }

    const hasBack = afterElements.some(element => this.elementText(element) === 'back');
    const hasForward = afterElements.some(element =>
      /^(next|continue|submit|done|finish)$/i.test(this.elementText(element))
    );
    if (!hasBack || !hasForward) {
      return false;
    }

    const typedValue = plannerAction.input.trim().toLowerCase();
    const textboxes = afterElements.filter(element =>
      ['textbox', 'searchbox', 'combobox'].includes((element.role || '').toLowerCase())
    );
    return (
      textboxes.length === 0 ||
      textboxes.some(element => {
        const text = this.elementText(element);
        return text.length > 0 && !text.includes(typedValue);
      })
    );
  }

  private inferSameUrlWizardProgressAfterNavigationClick(
    plannerAction: StepwisePlannerResponse,
    clickedElement: SnapshotElement | null,
    beforeSnapshot: Snapshot | null,
    afterSnapshot: Snapshot
  ): boolean {
    if (plannerAction.action !== 'CLICK') {
      return false;
    }

    const clickText = clickedElement ? this.elementText(clickedElement) : '';
    const plannerText = `${plannerAction.intent || ''} ${plannerAction.goal || ''}`.toLowerCase();
    const plannerRequestedForwardNavigation = /\b(next|continue)\b/.test(plannerText);
    const clickedForwardNavigation = this.isForwardNavigationControlText(clickText);
    if (!plannerRequestedForwardNavigation || !clickedForwardNavigation) {
      return false;
    }

    const beforeSignature = this.formPaneSignature(beforeSnapshot?.elements || []);
    const afterSignature = this.formPaneSignature(afterSnapshot.elements || []);
    if (!afterSignature || beforeSignature === afterSignature) {
      return false;
    }

    if (this.hasFormValidationError(afterSnapshot.elements || [])) {
      return false;
    }

    const hasBack = (afterSnapshot.elements || []).some(
      element => this.elementText(element) === 'back'
    );
    const hasForwardOrSubmit = (afterSnapshot.elements || []).some(element =>
      /^(next|continue|submit|done|finish)$/i.test(this.elementText(element))
    );

    return hasBack && hasForwardOrSubmit;
  }

  private inferTerminalFormSubmissionAfterClick(
    plannerAction: StepwisePlannerResponse,
    clickedElement: SnapshotElement | null,
    beforeSnapshot: Snapshot | null,
    afterSnapshot: Snapshot
  ): boolean {
    if (plannerAction.action !== 'CLICK') {
      return false;
    }

    const clickText = clickedElement ? this.elementText(clickedElement) : '';
    const plannerText = `${plannerAction.intent || ''} ${plannerAction.goal || ''}`.toLowerCase();
    const plannerRequestedSubmission = this.isTerminalSubmissionText(plannerText);
    const clickedSubmissionControl = this.isTerminalSubmissionControlText(clickText);
    if (!plannerRequestedSubmission || !clickedSubmissionControl) {
      return false;
    }

    const afterElements = afterSnapshot.elements || [];
    if (afterElements.length === 0 || this.hasFormValidationError(afterElements)) {
      return false;
    }

    const beforePageText = (beforeSnapshot?.elements || [])
      .map(element => this.elementText(element))
      .join(' ');
    const pageText = afterElements.map(element => this.elementText(element)).join(' ');
    const confirmationPattern =
      /\b(thanks?|thank you|submitted|complete|completed|success|received|confirmation)\b/i;
    if (confirmationPattern.test(pageText) && !confirmationPattern.test(beforePageText)) {
      return true;
    }

    const beforeSignature = this.formPaneSignature(beforeSnapshot?.elements || []);
    const afterSignature = this.formPaneSignature(afterElements);
    if (beforeSignature && beforeSignature === afterSignature) {
      return false;
    }

    if (confirmationPattern.test(pageText)) {
      return true;
    }

    const hasForwardOrTerminalControl = afterElements.some(element =>
      /^(next|continue|submit)$/i.test(this.elementText(element))
    );
    return beforeSignature.length > 0 && !hasForwardOrTerminalControl;
  }

  private isFinalFormSubmissionAction(
    task: string,
    plannerAction: StepwisePlannerResponse
  ): boolean {
    if (plannerAction.action !== 'CLICK') {
      return false;
    }

    const actionText = `${plannerAction.intent || ''} ${plannerAction.goal || ''} ${
      plannerAction.input || ''
    } ${plannerAction.target || ''}`;
    if (!this.isTerminalSubmissionText(actionText)) {
      return false;
    }

    return /\b(lastly|finally|final step|submit\s+(?:the\s+)?(?:multi-step\s+)?form|complete\s+(?:the\s+)?(?:form|onboarding|registration)|confirm registration|place order)\b/i.test(
      task
    );
  }

  private isForwardNavigationControlText(text: string): boolean {
    return /^(next|continue)$/i.test(text.trim());
  }

  private isTerminalSubmissionText(text: string): boolean {
    return /\b(submit|send|confirm|complete|completed|done|finish|register|sign\s*up|signup|create account|place order)\b/i.test(
      text
    );
  }

  private isTerminalSubmissionControlText(text: string): boolean {
    const normalized = text.trim();
    return (
      this.isTerminalSubmissionText(normalized) && !this.isForwardNavigationControlText(normalized)
    );
  }

  private inferFormChoiceSelectionAfterClick(
    plannerAction: StepwisePlannerResponse,
    clickedElement: SnapshotElement | null,
    afterSnapshot: Snapshot
  ): boolean {
    if (plannerAction.action !== 'CLICK' || !clickedElement) {
      return false;
    }

    const role = (clickedElement.role || '').toLowerCase();
    if (role !== 'radio' && role !== 'checkbox') {
      return false;
    }

    const afterElements = afterSnapshot.elements || [];
    if (afterElements.length === 0 || this.hasFormValidationError(afterElements)) {
      return false;
    }

    const clickedText = this.elementText(clickedElement);
    const plannerText = `${plannerAction.intent || ''} ${plannerAction.goal || ''}`.toLowerCase();
    return this.selectionControlMatchesIntent(clickedText, plannerText);
  }

  private selectionControlMatchesIntent(controlText: string, intentText: string): boolean {
    const controlWords = new Set(
      controlText
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter(word => word.length >= 3)
    );
    if (controlWords.size === 0) {
      return false;
    }

    const ignoredIntentWords = new Set([
      'button',
      'checkbox',
      'radio',
      'field',
      'input',
      'select',
      'choose',
      'click',
      'option',
    ]);
    const intentWords = intentText
      .split(/[^a-z0-9]+/)
      .filter(word => word.length >= 3 && !ignoredIntentWords.has(word));

    return intentWords.some(word => controlWords.has(word));
  }

  private formPaneSignature(elements: SnapshotElement[]): string {
    return elements
      .filter(element => {
        const role = (element.role || '').toLowerCase();
        return ['textbox', 'searchbox', 'combobox', 'checkbox', 'radio', 'button'].includes(role);
      })
      .map(element => `${(element.role || '').toLowerCase()}:${this.elementText(element)}`)
      .filter(Boolean)
      .join('|');
  }

  private hasFormValidationError(elements: SnapshotElement[]): boolean {
    return elements.some(element =>
      /\b(required|invalid|error|please enter|must enter|try again|failed)\b/i.test(
        this.elementText(element)
      )
    );
  }

  private elementText(element: SnapshotElement): string {
    return [element.text, element.name, element.ariaLabel, element.nearbyText]
      .filter((part): part is string => typeof part === 'string' && part.trim().length > 0)
      .join(' ')
      .trim()
      .toLowerCase();
  }
}
