/**
 * Recovery: Checkpoint and rollback mechanisms for automation recovery.
 *
 * This module provides state tracking and recovery mechanisms for when
 * automation gets off-track. Key concepts:
 *
 * - RecoveryCheckpoint: A snapshot of known-good state (URL, step, digest)
 * - RecoveryState: Tracks checkpoints and manages recovery attempts
 *
 * Recovery flow:
 * 1. After each successful step verification, record a checkpoint
 * 2. If verification fails repeatedly, attempt recovery to last checkpoint
 * 3. Navigate back to checkpoint URL and re-verify
 * 4. If recovery succeeds, resume from checkpoint step
 */

import type { PredicateSpec, Snapshot } from './plan-models';
import { evaluatePredicates } from './predicates';

/**
 * Configuration for recovery navigation.
 */
export interface RecoveryNavigationConfig {
  /** Whether recovery is enabled (default: true) */
  enabled: boolean;
  /** Maximum recovery attempts per run (default: 2) */
  maxRecoveryAttempts: number;
  /** Whether to track successful URLs for recovery (default: true) */
  trackSuccessfulUrls: boolean;
  /** Maximum checkpoints to retain (default: 10) */
  maxCheckpoints: number;
}

/**
 * Default recovery navigation configuration.
 */
export const DEFAULT_RECOVERY_CONFIG: RecoveryNavigationConfig = {
  enabled: true,
  maxRecoveryAttempts: 2,
  trackSuccessfulUrls: true,
  maxCheckpoints: 10,
};

/**
 * Checkpoint for rollback recovery.
 *
 * Created after each successful step verification to enable rollback
 * if subsequent steps fail.
 */
export interface RecoveryCheckpoint {
  /** The URL at this checkpoint */
  url: string;
  /** The step index that was just completed (0-indexed) */
  stepIndex: number;
  /** Hash of the snapshot for state verification */
  snapshotDigest: string;
  /** When the checkpoint was created */
  timestamp: Date;
  /** Labels of predicates that passed at this checkpoint */
  predicatesPassed: string[];
  /** Structured predicate specs preserved for recovery verification */
  verificationPredicates?: PredicateSpec[];
}

function normalizeRecoveryUrl(url: string): string {
  return (url || '').trim().replace(/\/+$/, '').toLowerCase();
}

function splitRecoveryArgs(rawArgs: string): string[] {
  return rawArgs
    .split(',')
    .map(arg => arg.trim())
    .filter(Boolean)
    .map(arg => arg.replace(/^['"]|['"]$/g, ''));
}

function parseRecoveryPredicate(label: string): PredicateSpec | null {
  const trimmed = (label || '').trim();
  const match = trimmed.match(/^([a-z_]+)\((.*)\)$/i);
  if (!match) {
    return null;
  }

  return {
    predicate: match[1].toLowerCase(),
    args: splitRecoveryArgs(match[2]),
  };
}

export function recoveryUrlsMatch(expectedUrl: string, actualUrl: string): boolean {
  const expected = normalizeRecoveryUrl(expectedUrl);
  const actual = normalizeRecoveryUrl(actualUrl);
  if (!expected || !actual) {
    return false;
  }

  return actual.includes(expected) || expected.includes(actual);
}

export function verifyRecoveryCheckpoint(
  checkpoint: RecoveryCheckpoint,
  snapshot: Snapshot | null
): boolean {
  if (!snapshot) {
    return false;
  }

  if (!recoveryUrlsMatch(checkpoint.url, snapshot.url || '')) {
    return false;
  }

  const predicateSpecs =
    Array.isArray(checkpoint.verificationPredicates) && checkpoint.verificationPredicates.length > 0
      ? checkpoint.verificationPredicates
      : checkpoint.predicatesPassed
          .map(parseRecoveryPredicate)
          .filter((value): value is PredicateSpec => value !== null);
  if (predicateSpecs.length === 0) {
    return true;
  }

  return evaluatePredicates(predicateSpecs, snapshot);
}

/**
 * Tracks recovery state for rollback mechanism.
 *
 * Checkpoints are created after each successful step verification.
 * Recovery can be attempted when steps fail repeatedly.
 *
 * @example
 * ```typescript
 * const state = new RecoveryState({ maxRecoveryAttempts: 2 });
 *
 * // After successful step
 * state.recordCheckpoint({
 *   url: 'https://shop.com/cart',
 *   stepIndex: 2,
 *   snapshotDigest: 'abc123',
 *   predicatesPassed: ['url_contains("/cart")'],
 * });
 *
 * // On repeated failure
 * if (state.canRecover()) {
 *   const checkpoint = state.consumeRecoveryAttempt();
 *   // Navigate to checkpoint.url and resume
 * }
 * ```
 */
export class RecoveryState {
  /** List of recorded checkpoints (most recent last) */
  private checkpoints: RecoveryCheckpoint[] = [];

  /** Number of recovery attempts consumed */
  private recoveryAttemptsUsed: number = 0;

  /** Maximum allowed recovery attempts */
  readonly maxRecoveryAttempts: number;

  /** The checkpoint being recovered to (if any) */
  currentRecoveryTarget: RecoveryCheckpoint | null = null;

  /** Maximum checkpoints to retain */
  readonly maxCheckpoints: number;

  constructor(config: Partial<RecoveryNavigationConfig> = {}) {
    this.maxRecoveryAttempts =
      config.maxRecoveryAttempts ?? DEFAULT_RECOVERY_CONFIG.maxRecoveryAttempts;
    this.maxCheckpoints = config.maxCheckpoints ?? DEFAULT_RECOVERY_CONFIG.maxCheckpoints;
  }

  /**
   * Record a successful checkpoint.
   *
   * Called after step verification passes to enable future rollback.
   *
   * @param checkpoint - Checkpoint data (without timestamp)
   * @returns The created RecoveryCheckpoint
   */
  recordCheckpoint(data: Omit<RecoveryCheckpoint, 'timestamp'>): RecoveryCheckpoint {
    const checkpoint: RecoveryCheckpoint = {
      ...data,
      timestamp: new Date(),
      predicatesPassed: data.predicatesPassed || [],
      verificationPredicates: Array.isArray(data.verificationPredicates)
        ? JSON.parse(JSON.stringify(data.verificationPredicates))
        : [],
    };

    this.checkpoints.push(checkpoint);

    // Keep only last N checkpoints to bound memory
    if (this.checkpoints.length > this.maxCheckpoints) {
      this.checkpoints = this.checkpoints.slice(-this.maxCheckpoints);
    }

    return checkpoint;
  }

  /**
   * Get the most recent checkpoint for recovery.
   *
   * @returns Most recent RecoveryCheckpoint, or null if no checkpoints exist
   */
  getRecoveryTarget(): RecoveryCheckpoint | null {
    if (this.checkpoints.length === 0) {
      return null;
    }
    return this.checkpoints[this.checkpoints.length - 1];
  }

  /**
   * Get checkpoint at a specific step index.
   *
   * @param stepIndex - The step index to find
   * @returns RecoveryCheckpoint at that step, or null if not found
   */
  getCheckpointAtStep(stepIndex: number): RecoveryCheckpoint | null {
    for (let i = this.checkpoints.length - 1; i >= 0; i--) {
      if (this.checkpoints[i].stepIndex === stepIndex) {
        return this.checkpoints[i];
      }
    }
    return null;
  }

  /**
   * Get the most recent checkpoint before a given step.
   *
   * @param stepIndex - The step index to find checkpoint before
   * @returns Most recent checkpoint with stepIndex < given index, or null
   */
  getCheckpointBeforeStep(stepIndex: number): RecoveryCheckpoint | null {
    for (let i = this.checkpoints.length - 1; i >= 0; i--) {
      if (this.checkpoints[i].stepIndex < stepIndex) {
        return this.checkpoints[i];
      }
    }
    return null;
  }

  /**
   * Check if recovery is still possible.
   *
   * @returns True if recovery attempts remain and checkpoints exist
   */
  canRecover(): boolean {
    return this.recoveryAttemptsUsed < this.maxRecoveryAttempts && this.checkpoints.length > 0;
  }

  /**
   * Consume a recovery attempt and return target checkpoint.
   *
   * Increments recoveryAttemptsUsed and sets currentRecoveryTarget.
   *
   * @returns The checkpoint to recover to, or null if recovery not possible
   */
  consumeRecoveryAttempt(): RecoveryCheckpoint | null {
    if (!this.canRecover()) {
      return null;
    }

    this.recoveryAttemptsUsed++;
    this.currentRecoveryTarget = this.getRecoveryTarget();
    return this.currentRecoveryTarget;
  }

  /**
   * Clear the current recovery target after recovery completes.
   */
  clearRecoveryTarget(): void {
    this.currentRecoveryTarget = null;
  }

  /**
   * Reset recovery state for a new run.
   */
  reset(): void {
    this.checkpoints = [];
    this.recoveryAttemptsUsed = 0;
    this.currentRecoveryTarget = null;
  }

  /**
   * Remove and return the most recent checkpoint.
   *
   * Useful when recovery fails and we want to try an earlier checkpoint.
   *
   * @returns The removed checkpoint, or null if no checkpoints exist
   */
  popCheckpoint(): RecoveryCheckpoint | null {
    if (this.checkpoints.length === 0) {
      return null;
    }
    return this.checkpoints.pop() || null;
  }

  /**
   * Get the URL of the most recent successful checkpoint.
   */
  get lastSuccessfulUrl(): string | null {
    if (this.checkpoints.length === 0) {
      return null;
    }
    return this.checkpoints[this.checkpoints.length - 1].url;
  }

  /**
   * Get the step index of the most recent successful checkpoint.
   */
  get lastSuccessfulStep(): number | null {
    if (this.checkpoints.length === 0) {
      return null;
    }
    return this.checkpoints[this.checkpoints.length - 1].stepIndex;
  }

  /**
   * Get the number of checkpoints.
   */
  get length(): number {
    return this.checkpoints.length;
  }

  /**
   * Get the number of recovery attempts used.
   */
  get attemptsUsed(): number {
    return this.recoveryAttemptsUsed;
  }

  toJSON(): {
    checkpoints: Array<{
      url: string;
      stepIndex: number;
      snapshotDigest: string;
      timestamp: string;
      predicatesPassed: string[];
      verificationPredicates?: PredicateSpec[];
    }>;
    recoveryAttemptsUsed: number;
  } {
    return {
      checkpoints: this.checkpoints.map(cp => ({
        url: cp.url,
        stepIndex: cp.stepIndex,
        snapshotDigest: cp.snapshotDigest,
        timestamp: cp.timestamp.toISOString(),
        predicatesPassed: cp.predicatesPassed,
        verificationPredicates: cp.verificationPredicates,
      })),
      recoveryAttemptsUsed: this.recoveryAttemptsUsed,
    };
  }

  static fromJSON(data: {
    checkpoints: Array<{
      url: string;
      stepIndex: number;
      snapshotDigest: string;
      timestamp: string;
      predicatesPassed: string[];
      verificationPredicates?: PredicateSpec[];
    }>;
    recoveryAttemptsUsed: number;
    maxRecoveryAttempts: number;
    maxCheckpoints: number;
  }): RecoveryState {
    const state = new RecoveryState({
      maxRecoveryAttempts: data.maxRecoveryAttempts,
      maxCheckpoints: data.maxCheckpoints,
    });
    state.checkpoints = data.checkpoints.map(cp => ({
      url: cp.url,
      stepIndex: cp.stepIndex,
      snapshotDigest: cp.snapshotDigest,
      timestamp: new Date(cp.timestamp),
      predicatesPassed: cp.predicatesPassed,
      verificationPredicates: cp.verificationPredicates,
    }));
    state.recoveryAttemptsUsed = data.recoveryAttemptsUsed;
    return state;
  }
}
