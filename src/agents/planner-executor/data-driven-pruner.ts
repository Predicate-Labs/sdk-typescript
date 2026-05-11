/**
 * Data-driven pruning using BrowserAgentProfile pruning policies.
 *
 * When a ResolvedAgentProfile contains a pruningPolicy, this module
 * applies it instead of the built-in category-based pruning.
 *
 * @see docs/plans/browser-agent/2026-05-02-extensible-categories-and-task-learning.md
 */

import type { Snapshot, SnapshotElement } from './plan-models';
import type { PruningTaskCategory } from './pruning-types';
import type { DataDrivenPruningPolicy, LearnedTargetFingerprint } from './profile-types';

function roleOf(element: SnapshotElement): string {
  return String(element.role || '').toLowerCase();
}

function textOf(element: SnapshotElement): string {
  return [element.text, element.name, element.ariaLabel, element.href]
    .filter((v): v is string => Boolean(v))
    .join(' ')
    .toLowerCase();
}

function scoreElement(
  element: SnapshotElement,
  goal: string,
  fingerprints: LearnedTargetFingerprint[]
): number {
  let score = Number(element.importance || 0);
  if (element.clickable) score += 15;
  if (element.inDominantGroup) score += 20;

  // Boost elements matching learned fingerprints
  for (const fp of fingerprints) {
    if (fp.confidence > 0.5 && matchesFingerprint(element, fp)) {
      score += Math.round(fp.confidence * 30);
    }
  }

  // Goal-term boost
  const goalTerms = goal
    .toLowerCase()
    .split(/\s+/)
    .filter(t => t.length > 2);
  const text = textOf(element);
  if (goalTerms.some(term => text.includes(term))) {
    score += 10;
  }

  return score;
}

function matchesFingerprint(element: SnapshotElement, fp: LearnedTargetFingerprint): boolean {
  if (fp.role && roleOf(element) !== fp.role.toLowerCase()) return false;
  if (fp.textTokens && fp.textTokens.length > 0) {
    const text = textOf(element);
    if (!fp.textTokens.some(token => text.includes(token.toLowerCase()))) return false;
  }
  if (fp.hrefPathPattern && element.href) {
    if (!String(element.href).includes(fp.hrefPathPattern)) return false;
  }
  return true;
}

export function pruneWithPolicy(
  snapshot: Snapshot,
  policy: DataDrivenPruningPolicy,
  goal: string,
  relaxationLevel: number,
  category: PruningTaskCategory,
  fingerprints: LearnedTargetFingerprint[] = [],
  gatewayRanked = false
): { elements: SnapshotElement[]; maxNodes: number } {
  let maxNodes: number;
  if (relaxationLevel === 0) {
    maxNodes = policy.maxElements;
  } else if (relaxationLevel === 1) {
    maxNodes = policy.maxElementsRelaxed;
  } else {
    maxNodes = policy.maxElementsLoose ?? Math.min(policy.maxElementsRelaxed * 2, 100);
  }

  const allowedRoles = new Set(policy.allowedRoles.map(r => r.toLowerCase()));
  const excludePatterns = (policy.excludeTextPatterns ?? []).map(p => p.toLowerCase());
  const includePatterns = policy.includeTextPatterns ?? [];

  const filtered = (snapshot.elements || []).filter(element => {
    if (allowedRoles.size > 0 && !allowedRoles.has(roleOf(element))) {
      return false;
    }

    const text = textOf(element);

    if (excludePatterns.some(p => text.includes(p))) {
      return false;
    }

    if (includePatterns.length > 0) {
      if (!includePatterns.some(p => text.includes(p.toLowerCase()))) {
        return false;
      }
    }

    return true;
  });

  const sorted = filtered.sort(
    (a, b) => scoreElement(b, goal, fingerprints) - scoreElement(a, goal, fingerprints)
  );
  const elements = gatewayRanked ? sorted : sorted.slice(0, maxNodes);

  return { elements, maxNodes };
}
