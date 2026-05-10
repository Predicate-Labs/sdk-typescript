/**
 * Vision Fallback Detection
 *
 * Detects when a snapshot is unusable and should trigger vision-based
 * element detection instead of text-based snapshot analysis.
 *
 * Vision fallback is triggered when:
 * - Snapshot has too few elements (< 10 and status indicates issues)
 * - Snapshot status is "require_vision" or "error"
 * - Diagnostics indicate low confidence or canvas page
 */

import type { Snapshot } from './plan-models';

/**
 * Snapshot diagnostics interface (optional data from runtime).
 */
export interface SnapshotDiagnostics {
  /** Confidence score 0-1 (low = unreliable) */
  confidence?: number;
  /** Whether page contains canvas elements */
  hasCanvas?: boolean;
  /** Whether vision is explicitly required */
  requiresVision?: boolean;
}

/**
 * Result of vision fallback detection.
 */
export interface VisionFallbackResult {
  /** Whether vision fallback should be used */
  shouldUseVision: boolean;
  /** Reason for vision fallback (if triggered) */
  reason: string | null;
}

/**
 * Detect if snapshot is unusable and should trigger vision fallback.
 *
 * Returns whether vision-based element detection should be used instead
 * of the text-based snapshot analysis.
 *
 * Note: If we have sufficient elements (10+), we should NOT trigger vision
 * fallback even if diagnostics suggest it. This handles cases where the
 * API incorrectly flags normal HTML pages as requiring vision.
 *
 * @param snapshot - The snapshot to analyze
 * @returns Vision fallback result with shouldUseVision flag and reason
 *
 * @example
 * ```typescript
 * const result = detectSnapshotFailure(snapshot);
 * if (result.shouldUseVision) {
 *   console.log(`Vision fallback needed: ${result.reason}`);
 * }
 * ```
 */
export function detectSnapshotFailure(snapshot: Snapshot | null): VisionFallbackResult {
  // Null snapshot always requires vision
  if (snapshot === null) {
    return { shouldUseVision: true, reason: 'snapshot_null' };
  }

  const elements = snapshot.elements || [];
  const elementCount = elements.length;

  // Check explicit status field (tri-state: success, error, require_vision)
  const status = snapshot.status || 'success';

  if (status === 'require_vision') {
    return { shouldUseVision: true, reason: 'require_vision' };
  }

  if (status === 'error') {
    return { shouldUseVision: true, reason: `snapshot_error` };
  }

  // Check diagnostics if available (from snapshot metadata)
  const diag = (snapshot as unknown as { diagnostics?: SnapshotDiagnostics }).diagnostics;
  if (diag) {
    // Low confidence
    if (diag.confidence !== undefined && diag.confidence < 0.3) {
      return { shouldUseVision: true, reason: 'low_confidence' };
    }

    // Canvas page with few elements
    if (diag.hasCanvas && elementCount < 5) {
      return { shouldUseVision: true, reason: 'canvas_page' };
    }

    // Diagnostics explicitly require vision
    if (diag.requiresVision && elementCount < 5) {
      return { shouldUseVision: true, reason: 'diagnostics_requires_vision' };
    }
  }

  // Too few elements to be useful — always trigger vision
  if (elementCount < 3) {
    return { shouldUseVision: true, reason: 'too_few_elements' };
  }

  // 3+ elements: snapshot is usable unless explicit conditions triggered above
  return { shouldUseVision: false, reason: null };
}

/**
 * Check if vision fallback is needed for a snapshot context.
 *
 * Convenience wrapper that checks both snapshot success and vision requirement.
 *
 * @param snapshotSuccess - Whether snapshot capture succeeded
 * @param requiresVision - Whether vision is already flagged as required
 * @returns true if vision should be used
 */
export function shouldUseVision(snapshotSuccess: boolean, requiresVision: boolean): boolean {
  return !snapshotSuccess || requiresVision;
}
