/**
 * Predicate System for Step Verification
 *
 * Predicates are used to verify step outcomes and enable pre-step verification
 * (skipping steps if the desired state is already achieved).
 *
 * Supported predicates:
 * - url_contains: Check if URL contains a substring
 * - url_equals: Check if URL equals a target URL
 * - url_matches: Check if URL matches a regex pattern
 * - exists: Check if element with text/selector exists
 * - not_exists: Check if element does not exist
 * - element_exists: Check if element with role and optional text/label exists
 * - element_count: Check element count within range
 * - any_of: Any of the sub-predicates passes
 * - all_of: All sub-predicates pass
 */

import type { Snapshot, SnapshotElement, PredicateSpec } from './plan-models';

// ---------------------------------------------------------------------------
// Predicate Interface
// ---------------------------------------------------------------------------

/**
 * A predicate that can be evaluated against a snapshot.
 */
export interface Predicate {
  /** Predicate type name */
  readonly name: string;
  /** Evaluate predicate against a snapshot */
  evaluate(snapshot: Snapshot): boolean;
}

// ---------------------------------------------------------------------------
// URL Predicates
// ---------------------------------------------------------------------------

/**
 * Check if URL contains a substring.
 */
export function urlContains(substring: string): Predicate {
  const needle = substring.trim();
  return {
    name: 'url_contains',
    evaluate(snapshot: Snapshot): boolean {
      if (!needle) {
        return false;
      }
      const url = snapshot.url || '';
      return url.toLowerCase().includes(needle.toLowerCase());
    },
  };
}

/**
 * Check if URL matches a regex pattern.
 */
export function urlMatches(pattern: string): Predicate {
  return {
    name: 'url_matches',
    evaluate(snapshot: Snapshot): boolean {
      const url = snapshot.url || '';
      try {
        const regex = new RegExp(pattern, 'i');
        return regex.test(url);
      } catch {
        // Invalid regex, fall back to substring match
        return url.toLowerCase().includes(pattern.toLowerCase());
      }
    },
  };
}

/**
 * Check if URL equals a target URL, ignoring trailing slash differences.
 */
export function urlEquals(targetUrl: string): Predicate {
  const target = normalizeUrlForEquality(targetUrl);
  return {
    name: 'url_equals',
    evaluate(snapshot: Snapshot): boolean {
      if (!target) {
        return false;
      }
      return normalizeUrlForEquality(snapshot.url || '') === target;
    },
  };
}

function normalizeUrlForEquality(url: string): string {
  const trimmed = url.trim();
  if (!trimmed) {
    return '';
  }
  try {
    const parsed = new URL(trimmed);
    parsed.hash = '';
    return parsed.toString().replace(/\/$/, '').toLowerCase();
  } catch {
    return trimmed.replace(/\/$/, '').toLowerCase();
  }
}

// ---------------------------------------------------------------------------
// Element Predicates
// ---------------------------------------------------------------------------

/**
 * Check if element matching selector/text exists.
 */
export function exists(selectorOrText: string): Predicate {
  return {
    name: 'exists',
    evaluate(snapshot: Snapshot): boolean {
      const elements = snapshot.elements || [];
      return elements.some(el => elementMatches(el, selectorOrText));
    },
  };
}

/**
 * Check if element matching selector/text does NOT exist.
 */
export function notExists(selectorOrText: string): Predicate {
  return {
    name: 'not_exists',
    evaluate(snapshot: Snapshot): boolean {
      const elements = snapshot.elements || [];
      return !elements.some(el => elementMatches(el, selectorOrText));
    },
  };
}

/**
 * Check element count is within range.
 */
export function elementCount(
  selectorOrText: string,
  minCount: number = 0,
  maxCount?: number
): Predicate {
  return {
    name: 'element_count',
    evaluate(snapshot: Snapshot): boolean {
      const elements = snapshot.elements || [];
      const matching = elements.filter(el => elementMatches(el, selectorOrText));
      const count = matching.length;
      if (count < minCount) return false;
      if (maxCount !== undefined && count > maxCount) return false;
      return true;
    },
  };
}

/**
 * Helper to check if element matches selector/text.
 */
function elementMatches(element: SnapshotElement, selectorOrText: string): boolean {
  const text = (element.text || '').toLowerCase();
  const role = (element.role || '').toLowerCase();
  const ariaLabel = (element.ariaLabel || '').toLowerCase();
  const query = selectorOrText.toLowerCase();

  // Direct text match
  if (text.includes(query)) return true;

  // Aria label match
  if (ariaLabel.includes(query)) return true;

  // Role-based selector (e.g., "button", "link")
  if (role === query) return true;

  // Combined role:text selector (e.g., "button:submit")
  if (selectorOrText.includes(':')) {
    const [roleQuery, textQuery] = selectorOrText.split(':', 2);
    if (role === roleQuery.toLowerCase() && text.includes(textQuery.toLowerCase())) {
      return true;
    }
  }

  return false;
}

// ---------------------------------------------------------------------------
// Composite Predicates
// ---------------------------------------------------------------------------

/**
 * Any of the sub-predicates passes (OR).
 */
export function anyOf(...predicates: Predicate[]): Predicate {
  return {
    name: 'any_of',
    evaluate(snapshot: Snapshot): boolean {
      return predicates.some(p => p.evaluate(snapshot));
    },
  };
}

/**
 * All sub-predicates pass (AND).
 */
export function allOf(...predicates: Predicate[]): Predicate {
  return {
    name: 'all_of',
    evaluate(snapshot: Snapshot): boolean {
      return predicates.every(p => p.evaluate(snapshot));
    },
  };
}

// ---------------------------------------------------------------------------
// Predicate Builder
// ---------------------------------------------------------------------------

/**
 * Build a Predicate from a specification.
 *
 * @param spec - Predicate specification (from plan step verify array)
 * @returns Predicate instance
 *
 * @example
 * ```typescript
 * const pred = buildPredicate({ predicate: 'url_contains', args: ['/cart'] });
 * const passes = pred.evaluate(snapshot);
 * ```
 */
export function buildPredicate(spec: PredicateSpec): Predicate {
  const { predicate: name, args } = spec;

  switch (name) {
    case 'url_contains':
      return urlContains(String(args[0] || ''));

    case 'url_equals':
      return urlEquals(String(args[0] || ''));

    case 'url_matches':
      return urlMatches(String(args[0] || ''));

    case 'exists':
      return exists(String(args[0] || ''));

    case 'not_exists':
      return notExists(String(args[0] || ''));

    case 'element_count': {
      const selector = String(args[0] || '');
      const minCount = typeof args[1] === 'number' ? args[1] : 0;
      const maxCount = typeof args[2] === 'number' ? args[2] : undefined;
      return elementCount(selector, minCount, maxCount);
    }

    case 'element_exists': {
      const role = String(args[0] || '').toLowerCase();
      const textQuery = args[1] ? String(args[1]).toLowerCase() : '';
      return {
        name: 'element_exists',
        evaluate(snapshot: Snapshot): boolean {
          const elements = snapshot.elements || [];
          return elements.some(el => {
            const elRole = (el.role || '').toLowerCase();
            if (role && elRole !== role) return false;
            if (!textQuery) return true;
            const elText = (el.text || '').toLowerCase();
            const elName = (el.name || '').toLowerCase();
            const elAria = (el.ariaLabel || '').toLowerCase();
            return (
              elText.includes(textQuery) || elName.includes(textQuery) || elAria.includes(textQuery)
            );
          });
        },
      };
    }

    case 'any_of':
      return anyOf(...(args as PredicateSpec[]).map(buildPredicate));

    case 'all_of':
      return allOf(...(args as PredicateSpec[]).map(buildPredicate));

    default:
      // Unknown predicates must fail closed so pre-step verification cannot skip real work.
      return {
        name: `unknown:${name}`,
        evaluate(): boolean {
          return false;
        },
      };
  }
}

/**
 * Evaluate all predicates against a snapshot.
 *
 * @param predicates - Array of predicate specifications
 * @param snapshot - Snapshot to evaluate against
 * @returns true if all predicates pass
 */
export function evaluatePredicates(predicates: PredicateSpec[], snapshot: Snapshot): boolean {
  for (const spec of predicates) {
    try {
      const pred = buildPredicate(spec);
      if (!pred.evaluate(snapshot)) {
        return false;
      }
    } catch {
      // On error, assume predicate fails
      return false;
    }
  }
  return true;
}
