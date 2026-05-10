/**
 * Browser-safe planner-executor entrypoint.
 *
 * This subpath intentionally excludes Playwright, Node filesystem tracing sinks,
 * and the package root's browser automation helpers so Chrome extensions can
 * bundle the planner-executor core behind their own AgentRuntime adapter.
 */

export { LLMProvider, type LLMResponse } from './llm-provider';

export {
  type SnapshotEscalationConfig,
  type RetryConfig,
  type StepwisePlanningConfig,
  type PlannerExecutorConfig,
  type DeepPartial,
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
} from './agents/planner-executor/config';

export {
  PredicateSpecSchema,
  PlanStepSchema,
  PlanSchema,
  ReplanPatchSchema,
  ActionType,
  StepStatus,
  type PredicateSpec,
  type PlanStep,
  type Plan,
  type ReplanPatch,
  type ActionRecord,
  type StepOutcome,
  type RunOutcome,
  type TokenUsageTotals,
  type TokenUsageSummary,
  type SnapshotContext,
  type ParsedAction,
  type Snapshot,
  type SnapshotElement,
  type SerializedAgentState,
  type SerializedLoopState,
  type SerializedRecoveryState,
} from './agents/planner-executor/plan-models';

export {
  buildStepwisePlannerPrompt,
  buildExecutorPrompt,
  type StepwisePlannerResponse,
} from './agents/planner-executor/prompts';

export {
  parseAction,
  extractJson,
  normalizePlan,
  validatePlanSmoothness,
  formatContext,
} from './agents/planner-executor/plan-utils';

export { TaskCategory, normalizeTaskCategory } from './agents/planner-executor/task-category';
export {
  AutomationTaskSchema,
  type AutomationTask,
} from './agents/planner-executor/automation-task';
export { HeuristicHint, type HeuristicHintInput } from './agents/planner-executor/heuristic-hint';
export { COMMON_HINTS, getCommonHint } from './agents/planner-executor/common-hints';
export {
  ComposableHeuristics,
  type ComposableHeuristicsOptions,
} from './agents/planner-executor/composable-heuristics';

export {
  type Predicate,
  urlContains,
  urlMatches,
  exists,
  notExists,
  elementCount,
  anyOf,
  allOf,
  buildPredicate,
  evaluatePredicates,
} from './agents/planner-executor/predicates';

export {
  PlannerExecutorAgent,
  type PlannerExecutorAgentOptions,
  type PreActionAuthorizer,
  type AuthorizationResult,
  type AgentRuntime,
  type IntentHeuristics,
} from './agents/planner-executor/planner-executor-agent';

// Profile and learning exports
export {
  type BrowserAgentProfile,
  type DataDrivenPruningPolicy,
  type ResolvedAgentProfile,
  type LearnedTargetFingerprint,
  type DomainProfile,
  EMPTY_RESOLVED_PROFILE,
} from './agents/planner-executor/profile-types';
export {
  BrowserAgentProfileSchema,
  BrowserAgentProfileArraySchema,
} from './agents/planner-executor/profile-schema';
export { ProfileRegistry } from './agents/planner-executor/profile-registry';
export { pruneWithPolicy } from './agents/planner-executor/data-driven-pruner';
export {
  computeTaskHash,
  extractDomain,
  createFingerprint,
  mergeFingerprint,
  recordFingerprintFailure,
} from './agents/planner-executor/fingerprint-normalizer';
export type { LearningStore } from './agents/planner-executor/learning-store';
export { InMemoryLearningStore } from './agents/planner-executor/learning-store';
export {
  detectPruningCategory,
  pruneSnapshotForTask,
} from './agents/planner-executor/category-pruner';
export {
  type PruningTaskCategory,
  type PruneSnapshotOptions,
} from './agents/planner-executor/pruning-types';
export {
  isSensitiveUrl,
  extractFingerprintFromOutcome,
  applyFingerprintFailure,
  applyFingerprintSuccess,
  isFingerprintStale,
  isFingerprintExpired,
  fingerprintToHint,
} from './agents/planner-executor/learning-extractor';
export type {
  LearningExtractionOptions,
  LearningExtractionResult,
} from './agents/planner-executor/learning-extractor';
