/**
 * PlannerExecutorAgent Module
 *
 * Two-tier LLM architecture for browser automation:
 * - Planner (7B+ model): Generates JSON execution plans
 * - Executor (3B-7B model): Executes steps with tight prompts
 *
 * Phase 1 (MVP) Features:
 * - Stepwise (ReAct-style) planning
 * - Snapshot limit escalation for reliable element capture
 * - Token usage tracking
 * - Pre-action authorization hook (for sidecar policy integration)
 *
 * Phase 2 (Reliability) Features:
 * - Scroll-after-escalation (viewport scrolling to find elements)
 * - Intent heuristics (text pattern matching for common intents)
 * - Pre-step verification (skip steps if predicates already pass)
 * - Retry/repair logic (submit method alternation)
 *
 * Phase 3 (Advanced) Features:
 * - Vision fallback detection (detectSnapshotFailure)
 * - Recovery navigation (RecoveryState, RecoveryCheckpoint)
 * - Boundary detection (auth pages, checkout pages)
 * - Modal/overlay dismissal (findDismissalTarget)
 */

// Configuration
export {
  SnapshotEscalationConfig,
  RetryConfig,
  StepwisePlanningConfig,
  PlannerExecutorConfig,
  DeepPartial,
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
} from './config';

// Factory
export {
  CreateAgentOptions,
  AgentProviders,
  detectProvider,
  createProvider,
  resolveConfig,
  createPlannerExecutorAgentProviders,
} from './agent-factory';

// Plan Models (Zod schemas and types)
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
} from './plan-models';

// Prompts
export {
  buildStepwisePlannerPrompt,
  buildExecutorPrompt,
  type StepwisePlannerResponse,
} from './prompts';

// Utilities
export {
  parseAction,
  extractJson,
  normalizePlan,
  validatePlanSmoothness,
  formatContext,
} from './plan-utils';

// Task Categories
export { TaskCategory, normalizeTaskCategory } from './task-category';
export { AutomationTaskSchema, type AutomationTask } from './automation-task';

// Planner-generated Heuristic Hints
export { HeuristicHint, type HeuristicHintInput } from './heuristic-hint';
export { COMMON_HINTS, getCommonHint } from './common-hints';
export { ComposableHeuristics, type ComposableHeuristicsOptions } from './composable-heuristics';

// Predicates
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
} from './predicates';

// Vision Fallback
export {
  type SnapshotDiagnostics,
  type VisionFallbackResult,
  detectSnapshotFailure,
  shouldUseVision,
} from './vision-fallback';

// Recovery Navigation
export {
  type RecoveryNavigationConfig,
  type RecoveryCheckpoint,
  RecoveryState,
  DEFAULT_RECOVERY_CONFIG,
} from './recovery';

// Boundary Detection
export {
  type AuthBoundaryConfig,
  type CheckoutDetectionConfig,
  type AuthBoundaryResult,
  type CheckoutDetectionResult,
  DEFAULT_AUTH_BOUNDARY_CONFIG,
  DEFAULT_CHECKOUT_CONFIG,
  detectAuthBoundary,
  detectCheckoutPage,
  isCheckoutElement,
} from './boundary-detection';

// Modal Dismissal
export {
  type ModalDismissalConfig,
  type ModalDismissalResult,
  DEFAULT_MODAL_CONFIG,
  findDismissalTarget,
  detectModalAppearance,
  detectModalDismissed,
} from './modal-dismissal';

// Agent
export {
  PlannerExecutorAgent,
  type PlannerExecutorAgentOptions,
  type PreActionAuthorizer,
  type AuthorizationResult,
  type AgentRuntime,
  type IntentHeuristics,
} from './planner-executor-agent';

// Profiles (extensible categories + task learning)
export type {
  DataDrivenPruningPolicy,
  BrowserAgentProfile,
  ResolvedAgentProfile,
  LearnedTargetFingerprint,
  DomainProfile,
} from './profile-types';
export { EMPTY_RESOLVED_PROFILE } from './profile-types';
export {
  DataDrivenPruningPolicySchema,
  BrowserAgentProfileSchema,
  BrowserAgentProfileArraySchema,
  LearnedTargetFingerprintSchema,
  DomainProfileSchema,
} from './profile-schema';
export { ProfileRegistry } from './profile-registry';
export { pruneWithPolicy } from './data-driven-pruner';
export {
  computeTaskHash,
  extractDomain,
  createFingerprint,
  mergeFingerprint,
  recordFingerprintFailure,
} from './fingerprint-normalizer';
export type { LearningStore } from './learning-store';
export { InMemoryLearningStore } from './learning-store';
export {
  isSensitiveUrl,
  extractFingerprintFromOutcome,
  applyFingerprintFailure,
  applyFingerprintSuccess,
  isFingerprintStale,
  isFingerprintExpired,
  fingerprintToHint,
  computeTaskHash as computeLearningTaskHash,
} from './learning-extractor';
export type { LearningExtractionOptions, LearningExtractionResult } from './learning-extractor';
export {
  TEXT_EXTRACTION_KEYWORDS,
  isTextExtractionTask,
  isExtractionTask,
  getExtractionDomainGuidance,
  buildExtractionPrompt,
} from './extraction-keywords';

// Runtime (Playwright/Chromium)
export {
  PlaywrightRuntime,
  createPlaywrightRuntime,
  type PlaywrightRuntimeOptions,
} from './playwright-runtime';
