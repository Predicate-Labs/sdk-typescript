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
} from './plan-models';
import { StepStatus, ReplanPatchSchema } from './plan-models';
import {
  buildStepwisePlannerPrompt,
  buildExecutorPrompt,
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
import { normalizeTaskCategory, type TaskCategory } from './task-category';
import { getCommonHint } from './common-hints';
import {
  detectModalAppearance,
  detectModalDismissed,
  findDismissalTarget,
  shouldAutoContinueCheckoutFlow,
} from './modal-dismissal';
import { detectPruningCategory } from './category-pruner';
import { pruneWithRecovery, fullSnapshotContainsIntent } from './pruning-recovery';
import type { Tracer } from '../../tracing/tracer';

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

  /** Type text into an element */
  type(elementId: number, text: string): Promise<void>;

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
    this.runId = `run-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    this.actionHistory = [];
    this.currentStepIndex = 0;
    this.tokenCollector.reset();
    this.currentTaskCategory = normalizeTaskCategory(options.category);
    this.composableHeuristics = new ComposableHeuristics({
      staticHeuristics: this.baseIntentHeuristics,
      taskCategory: this.currentTaskCategory,
    });
    this.composableHeuristics.clearStepHints();
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

        // Take snapshot with escalation
        const ctx = await this.snapshotWithEscalation(runtime, task);
        currentUrl = ctx.snapshot?.url || currentUrl;
        fallbackUsed = fallbackUsed || ctx.requiresVision;

        if (this.config.authBoundary.enabled) {
          const authResult = detectAuthBoundary(currentUrl, this.config.authBoundary);
          if (authResult.isAuthBoundary && this.config.authBoundary.stopOnAuth) {
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

        let plannerAction: StepwisePlannerResponse;
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

          let plannerResp: LLMResponse;
          try {
            plannerResp = await this.planner.generate(systemPrompt, userPrompt, {
              temperature: this.config.plannerTemperature,
              max_tokens: this.config.plannerMaxTokens,
            });
            this.recordTokenUsage('planner', plannerResp);
          } catch (plannerError) {
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
              error: `Planner LLM call failed: ${plannerError instanceof Error ? plannerError.message : String(plannerError)}`,
            });
            continue;
          }

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

        // Update current URL
        let urlAfter = await runtime.getCurrentUrl();
        currentUrl = urlAfter;
        fallbackUsed = fallbackUsed || finalOutcome.usedVision;
        let shouldContinue = false;
        let actionHistoryRecorded = false;

        if (finalOutcome.status === StepStatus.FAILED) {
          if (this.config.authBoundary.enabled) {
            const authResult = detectAuthBoundary(currentUrl, this.config.authBoundary);
            if (authResult.isAuthBoundary && this.config.authBoundary.stopOnAuth) {
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
          this.actionHistory.push({
            stepNum,
            action: plannerAction.action,
            target: this.summarizePlannerActionTarget(plannerAction),
            result: finalOutcome.status === StepStatus.SUCCESS ? 'success' : 'failed',
            urlAfter,
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

    if (this.config.preStepVerification && (plannerAction.verify?.length || 0) > 0) {
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

      if (parsed.action === 'CLICK') {
        await runtime.click(elementId);
        await this.handlePostClickEffects(runtime, plannerAction, activeCtx);
        const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
        const urlAfter = await runtime.getCurrentUrl();
        return {
          stepId: stepNum,
          goal: plannerAction.intent || 'Click element',
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: `CLICK(${elementId})`,
          llmResponseText: executorResp?.content,
          verificationPassed,
          usedVision: shouldUseVision,
          durationMs: Date.now() - stepStart,
          urlBefore: currentUrl,
          urlAfter,
        };
      } else if (parsed.action === 'TYPE') {
        const text = plannerAction.input || (parsed.args[1] as string) || '';
        await runtime.type(elementId, text);

        // Submit with Enter key for TYPE_AND_SUBMIT
        if (plannerAction.action === 'TYPE_AND_SUBMIT') {
          const preUrl = await runtime.getCurrentUrl();
          const elements = activeCtx.snapshot?.elements || [];
          const inputElement = elements.find(element => element.id === elementId) || null;
          const isSearchLike = isSearchLikeTypeAndSubmit(plannerAction, inputElement);
          const submitButtonId = this.findSubmitButton(elements, elementId, isSearchLike);
          const hasRetryBudget = this.config.retry.executorRepairAttempts > 0;

          let changedUrl: string | null = null;
          let submissionSatisfied = false;

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

        const verificationPassed = await this.verifyStepOutcome(runtime, plannerAction);
        const urlAfter = await runtime.getCurrentUrl();

        return {
          stepId: stepNum,
          goal: plannerAction.intent || 'Type text',
          status: verificationPassed ? StepStatus.SUCCESS : StepStatus.FAILED,
          actionTaken: `TYPE(${elementId}, "${text}")`,
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

        const visionResult = detectSnapshotFailure(snap);
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
        const elementCount = snap.elements?.length || 0;
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
    const shouldTryScroll =
      cfg.scrollAfterEscalation &&
      step !== undefined &&
      lastSnapshot !== null &&
      !requiresVision &&
      step.action === 'CLICK' &&
      step.intent;

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

    return null;
  }

  private async resolveExecutorAction(
    plannerAction: StepwisePlannerResponse,
    ctx: SnapshotContext,
    task: string,
    forceVision: boolean = false
  ): Promise<{ parsed: ParsedAction; shouldUseVision: boolean; executorResp: LLMResponse | null }> {
    const hasExplicitStepHints = (plannerAction.heuristicHints?.length || 0) > 0;
    const hasCommonHint = Boolean(plannerAction.intent && getCommonHint(plannerAction.intent));
    const allowHeuristicsDespiteVision =
      ctx.requiresVision &&
      ctx.visionReason === 'too_few_elements' &&
      (hasExplicitStepHints || hasCommonHint);
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
        executorResp = await this.executor.generateWithImage(
          execSystem,
          execUser,
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

    return {
      id: typeof step.id === 'number' ? step.id : undefined,
      goal: typeof step.goal === 'string' ? step.goal : undefined,
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
    };
  }

  private summarizePlannerActionTarget(plannerAction: StepwisePlannerResponse): string | null {
    if (plannerAction.action === 'TYPE' || plannerAction.action === 'TYPE_AND_SUBMIT') {
      return plannerAction.input || plannerAction.intent || plannerAction.target || null;
    }

    return plannerAction.intent || plannerAction.input || plannerAction.target || null;
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
        result: substepOutcome.status === StepStatus.SUCCESS ? 'success' : 'failed',
        urlAfter: substepOutcome.urlAfter || (await runtime.getCurrentUrl()),
      });
    }

    return substepOutcomes;
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

    while (Date.now() - start <= timeoutMs) {
      try {
        const snap = await runtime.snapshot({
          limit: this.config.snapshot.limitBase,
          screenshot: false,
          goal: plannerAction.intent || plannerAction.action,
        });
        if (snap && evaluatePredicates(plannerAction.verify, snap)) {
          return true;
        }
      } catch {
        // Keep polling until timeout.
      }

      await new Promise(resolve => setTimeout(resolve, pollMs));
    }

    return false;
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
        await runtime.goto(checkpoint.url);
        const verificationSnapshot = await runtime.snapshot({
          limit: this.config.snapshot.limitBase,
          screenshot: false,
          goal: 'recovery verification',
        });
        const recovered = verifyRecoveryCheckpoint(checkpoint, verificationSnapshot);
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
  ): Promise<void> {
    if (!this.config.modal.enabled || !ctx.snapshot) {
      return;
    }

    const postSnap = await runtime.snapshot({
      limit: this.config.snapshot.limitMax,
      screenshot: false,
      goal: plannerAction.intent || plannerAction.action,
    });
    if (!postSnap) {
      return;
    }

    const preElements = new Set((ctx.snapshot.elements || []).map(el => el.id));
    const postElements = new Set((postSnap.elements || []).map(el => el.id));
    if (!detectModalAppearance(preElements, postElements, this.config.modal.minNewElements)) {
      return;
    }

    const modalElements = (postSnap.elements || []).filter(element => !preElements.has(element.id));
    const checkoutTarget = this.findCheckoutContinuationTarget(modalElements);
    if (checkoutTarget !== null) {
      if (!shouldAutoContinueCheckoutFlow(plannerAction.intent)) {
        return;
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
      return;
    }

    const dismissal = findDismissalTarget(modalElements, this.config.modal);
    if (!dismissal.found || dismissal.elementId === null) {
      return;
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
      return;
    }

    detectModalDismissed(postElements, new Set((finalSnap.elements || []).map(el => el.id)));
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
      : ['submit', 'continue', 'save', 'send', 'sign in', 'log in', 'apply', 'ok', 'done'];

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
}
