/**
 * Plan Utilities for PlannerExecutorAgent
 *
 * Action parsing, plan normalization, and validation utilities.
 */

import type { ParsedAction, Plan, SnapshotElement } from './plan-models';
import type { PrunedSnapshotContext } from './pruning-types';

// ---------------------------------------------------------------------------
// Action Parsing
// ---------------------------------------------------------------------------

/**
 * Parse action from executor response.
 *
 * Handles various LLM output formats:
 * - CLICK(42)
 * - - CLICK(42)  (with leading dash/bullet)
 * - TYPE(42, "text")
 * - - TYPE(42, "Logitech mouse")
 * - SCROLL(down)
 * - PRESS('Enter')
 * - NONE (executor couldn't find element)
 *
 * @param text - Raw executor response
 * @returns Parsed action with type and arguments
 */
export function parseAction(text: string): ParsedAction {
  let cleaned = text.trim();

  // Strip <think>...</think> tags (Qwen/DeepSeek reasoning output)
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  // Some local models leak reasoning without the opening tag but still close it before the answer.
  const closingThinkIndex = cleaned.toLowerCase().lastIndexOf('</think>');
  if (closingThinkIndex !== -1) {
    cleaned = cleaned.slice(closingThinkIndex + '</think>'.length).trim();
  }
  // If <think> never closed, strip from first <think> to end
  cleaned = cleaned.replace(/<think>[\s\S]*$/gi, '').trim();

  // If after stripping think tags we have empty content, return NONE
  // This happens when the model only outputs thinking without an actual action
  if (!cleaned || cleaned.length === 0) {
    return { action: 'NONE', args: ['empty response after stripping think tags'] };
  }

  const directMatch = parseExactActionLine(cleaned);
  if (directMatch) {
    return directMatch;
  }

  const actionLines = cleaned
    .split(/\r?\n/)
    .map(normalizeActionCandidateLine)
    .filter((line): line is string => line !== null)
    .map(parseExactActionLine)
    .filter((line): line is ParsedAction => line !== null);

  if (actionLines.length === 1) {
    return actionLines[0];
  }

  return { action: 'UNKNOWN', args: [text] };
}

function normalizeActionCandidateLine(line: string): string | null {
  const trimmed = line.trim();
  if (!trimmed || /^```/.test(trimmed)) {
    return null;
  }

  const withoutBullet = trimmed.replace(/^[-*•]\s*/, '');
  const withoutLabel = withoutBullet.replace(
    /^(?:final\s+action|action|output|answer|return)\s*:\s*/i,
    ''
  );

  return withoutLabel.trim() || null;
}

function parseExactActionLine(line: string): ParsedAction | null {
  // CLICK(<id>)
  const clickMatch = line.match(/^CLICK\((\d+)\)$/);
  if (clickMatch) {
    return { action: 'CLICK', args: [parseInt(clickMatch[1], 10)] };
  }

  // TYPE(<id>, "text") - also handle without quotes
  const typeMatch = line.match(/^TYPE\((\d+),\s*["']?([^"']+?)["']?\)$/);
  if (typeMatch) {
    return { action: 'TYPE', args: [parseInt(typeMatch[1], 10), typeMatch[2].trim()] };
  }

  // PRESS('key')
  const pressMatch = line.match(/^PRESS\(['"]?(.+?)['"]?\)$/);
  if (pressMatch) {
    return { action: 'PRESS', args: [pressMatch[1]] };
  }

  // SCROLL(direction)
  const scrollMatch = line.match(/^SCROLL\((\w+)\)$/);
  if (scrollMatch) {
    return { action: 'SCROLL', args: [scrollMatch[1]] };
  }

  // FINISH()
  if (line === 'FINISH()') {
    return { action: 'FINISH', args: [] };
  }

  // DONE
  if (line.toUpperCase() === 'DONE') {
    return { action: 'DONE', args: [] };
  }

  // NONE - executor couldn't find a suitable element
  if (line.toUpperCase() === 'NONE') {
    return { action: 'NONE', args: [] };
  }

  // CLICK_XY(x, y) — vision fallback coordinate click
  const clickXyMatch = line.match(
    /^CLICK_XY\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$/i
  );
  if (clickXyMatch) {
    return { action: 'CLICK_XY', args: [Number(clickXyMatch[1]), Number(clickXyMatch[2])] };
  }

  // CLICK_RECT(x, y, w, h) — vision fallback rect click
  const clickRectMatch = line.match(
    /^CLICK_RECT\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)$/i
  );
  if (clickRectMatch) {
    return {
      action: 'CLICK_RECT',
      args: [
        Number(clickRectMatch[1]),
        Number(clickRectMatch[2]),
        Number(clickRectMatch[3]),
        Number(clickRectMatch[4]),
      ],
    };
  }

  // TYPE_AT("text") — vision fallback: type at current position
  const typeAtMatch = line.match(/^TYPE_AT\s*\(\s*["']([\s\S]*?)["']\s*\)$/i);
  if (typeAtMatch) {
    return { action: 'TYPE_AT', args: [typeAtMatch[1]] };
  }

  return null;
}

// ---------------------------------------------------------------------------
// JSON Extraction
// ---------------------------------------------------------------------------

/**
 * Strip thinking tags from LLM response (Qwen, DeepSeek, etc.)
 */
function stripThinkingTags(content: string): string {
  let cleaned = content;
  // Strip complete <think>...</think> tags
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  // If <think> never closed, strip from first <think> to end
  cleaned = cleaned.replace(/<think>[\s\S]*$/gi, '').trim();
  return cleaned;
}

/**
 * Repair common JSON issues from LLM output.
 *
 * Handles:
 * - Unquoted object keys (valid JS but not valid JSON): `reasoning:"..."` → `"reasoning":"..."`
 * - Single-quoted strings: `'text'` → `"text"`
 * - Trailing commas before `}` or `]`
 *
 * @param text - Raw text that looks like JSON but may have syntax issues
 * @returns Repaired JSON string
 */
function repairJson(text: string): string {
  let repaired = text;

  // Fix backslash-escaped quotes that make JSON unparseable.
  // Some LLMs output: {"action":"CLICK",...,\"reasoning\":\"text\"}
  // where the outer quotes are real but inner key/value quotes are escaped.
  // Heuristic: if the text has unescaped quotes AND backslash-escaped quotes
  // mixed together, unescape the backslash-escaped ones.
  const hasUnescapedQuotes = /[^\\]"/.test(repaired);
  const hasEscapedQuotes = /\\"/.test(repaired);
  if (hasUnescapedQuotes && hasEscapedQuotes) {
    repaired = repaired.replace(/\\"/g, '"');
  }

  // Add double quotes around unquoted object keys
  // Matches: word-characters followed by colon (not already inside a string)
  // Pattern: start of object `{` or comma `,`, optional whitespace, then unquoted key, then `:`
  repaired = repaired.replace(/([{,]\s*)([\w$]+)\s*:/g, '$1"$2":');

  // Replace single-quoted strings with double-quoted strings
  // This is a simple heuristic — it won't handle escaped single quotes inside strings,
  // but it handles the common case of LLMs outputting 'text' instead of "text"
  repaired = repaired.replace(/'([^']*)'/g, '"$1"');

  // Remove trailing commas before } or ]
  repaired = repaired.replace(/,\s*([}\]])/g, '$1');

  return repaired;
}

/**
 * Extract the first balanced JSON object from a string.
 *
 * Handles cases where the LLM outputs multiple concatenated JSON objects:
 *   {"action":"EXTRACT",...},{"action":"CLICK",...}
 *
 * Uses brace-depth counting to find the first complete `{...}` block.
 *
 * @returns The first balanced JSON object string, or null if not found
 */
function extractFirstJsonObject(text: string): string | null {
  const start = text.indexOf('{');
  if (start === -1) return null;

  let depth = 0;
  let inString = false;
  let escape = false;

  for (let i = start; i < text.length; i++) {
    const ch = text[i];

    if (escape) {
      escape = false;
      continue;
    }

    if (ch === '\\' && inString) {
      escape = true;
      continue;
    }

    if (ch === '"') {
      inString = !inString;
      continue;
    }

    if (inString) continue;

    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) {
        return text.slice(start, i + 1);
      }
    }
  }

  return null;
}

/**
 * Extract JSON from LLM response that may contain markdown or prose.
 *
 * Handles:
 * - Pure JSON responses
 * - JSON wrapped in ```json code blocks
 * - JSON embedded in prose text
 * - Qwen/DeepSeek <think>...</think> tags
 *
 * @param content - Raw LLM response
 * @returns Parsed JSON object
 * @throws Error if no valid JSON found
 */
export function extractJson(content: string): Record<string, unknown> {
  // Strip thinking tags first (Qwen, DeepSeek models)
  const cleaned = stripThinkingTags(content);

  // Try direct parse first
  try {
    return JSON.parse(cleaned);
  } catch {
    // Continue to extraction methods
  }

  // Try to extract from code block
  const codeBlockMatch = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    try {
      return JSON.parse(codeBlockMatch[1].trim());
    } catch {
      // Continue to other methods
    }
  }

  // Try to find JSON object in text
  // First try to extract just the FIRST balanced JSON object to handle
  // cases where LLM outputs multiple objects like: {"action":"EXTRACT",...},{"action":"CLICK",...}
  const firstObj = extractFirstJsonObject(cleaned);
  if (firstObj) {
    try {
      return JSON.parse(firstObj);
    } catch {
      // Try to repair common JSON issues: unquoted keys (valid JS but not valid JSON)
      try {
        const repaired = repairJson(firstObj);
        return JSON.parse(repaired);
      } catch {
        // Continue to last resort
      }
    }
  }

  // Fallback: greedy match (original behavior)
  const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch {
      // Try to repair common JSON issues: unquoted keys (valid JS but not valid JSON)
      try {
        const repaired = repairJson(jsonMatch[0]);
        return JSON.parse(repaired);
      } catch {
        // Continue to last resort
      }
    }
  }

  // Last resort: try to repair the entire cleaned content as a JS object literal
  try {
    const repaired = repairJson(cleaned);
    return JSON.parse(repaired);
  } catch {
    // Give up
  }

  throw new Error(`Failed to extract JSON from response: ${cleaned.slice(0, 200)}`);
}

// ---------------------------------------------------------------------------
// Plan Normalization
// ---------------------------------------------------------------------------

/**
 * Action name aliases to normalize LLM output variations.
 */
const ACTION_ALIASES: Record<string, string> = {
  CLICK_ELEMENT: 'CLICK',
  CLICK_BUTTON: 'CLICK',
  CLICK_LINK: 'CLICK',
  CLICK_XY: 'CLICK',
  TYPE_AT: 'TYPE',
  INPUT: 'TYPE_AND_SUBMIT',
  TYPE_TEXT: 'TYPE_AND_SUBMIT',
  ENTER_TEXT: 'TYPE_AND_SUBMIT',
  EXTRACT_TEXT: 'EXTRACT',
  GOTO: 'NAVIGATE',
  GO_TO: 'NAVIGATE',
  OPEN: 'NAVIGATE',
  SCROLL_DOWN: 'SCROLL',
  SCROLL_UP: 'SCROLL',
  SCROLL_TO: 'SCROLL',
  SCROLL_INTO_VIEW: 'SCROLL',
};

/**
 * Parse a string predicate into a normalized object.
 *
 * LLMs sometimes output predicates as strings like:
 * - "url_contains('amazon.com')" -> {predicate: "url_contains", args: ["amazon.com"]}
 * - "exists(role=button)" -> {predicate: "exists", args: ["role=button"]}
 */
function parseStringPredicate(predStr: string): Record<string, unknown> | null {
  const cleaned = predStr.trim();

  // Try to match function-call style: predicate_name(args)
  // Use [\s\S] instead of . with 's' flag for cross-browser compatibility
  const match = cleaned.match(/^(\w+)\s*\(\s*([\s\S]+?)\s*\)$/);
  if (match) {
    const predName = match[1];
    let argsStr = match[2].trim();

    // Strip quotes from args if present
    if (
      (argsStr.startsWith("'") && argsStr.endsWith("'")) ||
      (argsStr.startsWith('"') && argsStr.endsWith('"'))
    ) {
      argsStr = argsStr.slice(1, -1);
    }

    return {
      predicate: predName,
      args: [argsStr],
    };
  }

  // Try simple predicate name without args
  if (/^[\w_]+$/.test(cleaned)) {
    return {
      predicate: cleaned,
      args: [],
    };
  }

  return null;
}

/**
 * Normalize a verify predicate to the expected format.
 *
 * LLMs may output predicates in various formats:
 * - {"url_contains": "amazon.com"} -> {"predicate": "url_contains", "args": ["amazon.com"]}
 * - {"predicate": "url_contains", "input": "x"} -> {"predicate": "url_contains", "args": ["x"]}
 */
function normalizeVerifyPredicate(pred: Record<string, unknown>): Record<string, unknown> {
  const result = { ...pred };

  // Handle "type" field as alternative to "predicate"
  if ('type' in result && !('predicate' in result)) {
    result.predicate = result.type;
    delete result.type;
  }

  // Already has predicate field - normalize args
  if ('predicate' in result) {
    if (!result.args || (Array.isArray(result.args) && result.args.length === 0)) {
      if ('input' in result) {
        result.args = [result.input];
        delete result.input;
      } else if ('value' in result) {
        result.args = [result.value];
        delete result.value;
      } else if ('pattern' in result) {
        result.args = [result.pattern];
        delete result.pattern;
      } else if ('substring' in result) {
        result.args = [result.substring];
        delete result.substring;
      } else if ('selector' in result) {
        result.args = [result.selector];
        delete result.selector;
      }
    }
    return result;
  }

  // Predicate type is a key in the dict (e.g., {"url_contains": "amazon.com"})
  const knownPredicates = [
    'url_contains',
    'url_equals',
    'url_matches',
    'exists',
    'not_exists',
    'element_count',
    'element_visible',
    'any_of',
    'all_of',
    'text_contains',
    'text_equals',
  ];

  for (const predType of knownPredicates) {
    if (predType in result) {
      return {
        predicate: predType,
        args: result[predType] ? [result[predType]] : [],
      };
    }
  }

  // Unknown format - return as-is
  return result;
}

function normalizeHeuristicHint(hint: unknown): Record<string, unknown> {
  if (typeof hint !== 'object' || hint === null) {
    return {};
  }

  const raw = { ...(hint as Record<string, unknown>) };
  return {
    intentPattern:
      typeof raw.intentPattern === 'string'
        ? raw.intentPattern
        : typeof raw.intent_pattern === 'string'
          ? raw.intent_pattern
          : '',
    textPatterns: Array.isArray(raw.textPatterns)
      ? raw.textPatterns
      : Array.isArray(raw.text_patterns)
        ? raw.text_patterns
        : [],
    roleFilter: Array.isArray(raw.roleFilter)
      ? raw.roleFilter
      : Array.isArray(raw.role_filter)
        ? raw.role_filter
        : [],
    priority: typeof raw.priority === 'number' ? raw.priority : 0,
    attributePatterns:
      typeof raw.attributePatterns === 'object' && raw.attributePatterns !== null
        ? raw.attributePatterns
        : typeof raw.attribute_patterns === 'object' && raw.attribute_patterns !== null
          ? raw.attribute_patterns
          : {},
  };
}

function normalizeStep(step: Record<string, unknown>): Record<string, unknown> {
  const normalizedStep = { ...step };

  if ('action' in normalizedStep && typeof normalizedStep.action === 'string') {
    const action = normalizedStep.action.toUpperCase();
    normalizedStep.action = ACTION_ALIASES[action] || action;
  }

  if ('url' in normalizedStep && !('target' in normalizedStep)) {
    normalizedStep.target = normalizedStep.url;
    delete normalizedStep.url;
  }

  if ('target' in normalizedStep && typeof normalizedStep.target === 'number') {
    normalizedStep.target = String(normalizedStep.target);
  }
  if (
    'target' in normalizedStep &&
    typeof normalizedStep.target === 'object' &&
    normalizedStep.target !== null
  ) {
    normalizedStep.target = JSON.stringify(normalizedStep.target);
  }
  if ('target' in normalizedStep && normalizedStep.target === null) {
    delete normalizedStep.target;
  }
  if ('input' in normalizedStep && typeof normalizedStep.input !== 'string') {
    const inputValue = normalizedStep.input;
    if (inputValue === null || inputValue === undefined) {
      delete normalizedStep.input;
    } else if (
      typeof inputValue === 'number' ||
      typeof inputValue === 'boolean' ||
      typeof inputValue === 'bigint'
    ) {
      normalizedStep.input = String(inputValue);
    } else {
      normalizedStep.input = JSON.stringify(inputValue);
    }
  }

  if ('id' in normalizedStep && typeof normalizedStep.id === 'string') {
    const parsed = parseInt(normalizedStep.id, 10);
    if (!isNaN(parsed)) {
      normalizedStep.id = parsed;
    }
  }

  if ('verify' in normalizedStep && Array.isArray(normalizedStep.verify)) {
    normalizedStep.verify = normalizedStep.verify.map((pred: unknown) => {
      if (typeof pred === 'object' && pred !== null) {
        return normalizeVerifyPredicate(pred as Record<string, unknown>);
      }
      if (typeof pred === 'string') {
        const parsed = parseStringPredicate(pred);
        return parsed ?? { predicate: 'unknown', args: [pred] };
      }
      return pred;
    });
  }

  if ('optional_substeps' in normalizedStep && Array.isArray(normalizedStep.optional_substeps)) {
    normalizedStep.optionalSubsteps = normalizedStep.optional_substeps.map((substep: unknown) =>
      normalizeStep(substep as Record<string, unknown>)
    );
    delete normalizedStep.optional_substeps;
  } else if (
    'optionalSubsteps' in normalizedStep &&
    Array.isArray(normalizedStep.optionalSubsteps)
  ) {
    normalizedStep.optionalSubsteps = normalizedStep.optionalSubsteps.map((substep: unknown) =>
      normalizeStep(substep as Record<string, unknown>)
    );
  }

  if ('stop_if_true' in normalizedStep) {
    normalizedStep.stopIfTrue = normalizedStep.stop_if_true;
    delete normalizedStep.stop_if_true;
  }

  if ('heuristic_hints' in normalizedStep && Array.isArray(normalizedStep.heuristic_hints)) {
    normalizedStep.heuristicHints = normalizedStep.heuristic_hints.map(normalizeHeuristicHint);
    delete normalizedStep.heuristic_hints;
  } else if ('heuristicHints' in normalizedStep && Array.isArray(normalizedStep.heuristicHints)) {
    normalizedStep.heuristicHints = normalizedStep.heuristicHints.map(normalizeHeuristicHint);
  }

  return normalizedStep;
}

function normalizeNumericId(value: unknown): unknown {
  if (typeof value === 'string') {
    const parsed = parseInt(value, 10);
    if (!isNaN(parsed)) {
      return parsed;
    }
  }
  return value;
}

/**
 * Normalize plan dictionary to handle LLM output variations.
 *
 * Handles:
 * - url vs target field names
 * - action aliases (click vs CLICK)
 * - step id variations (string vs int)
 * - verify predicate format variations
 *
 * @param planDict - Raw plan dictionary from LLM
 * @returns Normalized plan dictionary
 */
export function normalizePlan(planDict: Record<string, unknown>): Record<string, unknown> {
  const result = { ...planDict };

  if ('steps' in result && Array.isArray(result.steps)) {
    result.steps = result.steps.map((step: Record<string, unknown>) => normalizeStep(step));
  }

  return result;
}

export function normalizeReplanPatch(patchDict: Record<string, unknown>): Record<string, unknown> {
  const result = { ...patchDict };

  if ('replace_steps' in result && Array.isArray(result.replace_steps)) {
    result.replaceSteps = result.replace_steps.map((item: unknown) => {
      const raw =
        typeof item === 'object' && item !== null ? { ...(item as Record<string, unknown>) } : {};
      return {
        ...raw,
        id: normalizeNumericId(raw.id),
        step:
          typeof raw.step === 'object' && raw.step !== null
            ? normalizeStep(raw.step as Record<string, unknown>)
            : raw.step,
      };
    });
    delete result.replace_steps;
  } else if ('replaceSteps' in result && Array.isArray(result.replaceSteps)) {
    result.replaceSteps = result.replaceSteps.map((item: unknown) => {
      const raw =
        typeof item === 'object' && item !== null ? { ...(item as Record<string, unknown>) } : {};
      return {
        ...raw,
        id: normalizeNumericId(raw.id),
        step:
          typeof raw.step === 'object' && raw.step !== null
            ? normalizeStep(raw.step as Record<string, unknown>)
            : raw.step,
      };
    });
  }

  return result;
}

// ---------------------------------------------------------------------------
// Plan Validation
// ---------------------------------------------------------------------------

/**
 * Validate plan quality and smoothness.
 *
 * Checks for common issues that indicate a low-quality plan:
 * - Missing verification predicates
 * - Consecutive same actions
 * - Empty or too short plans
 * - Missing required fields
 *
 * @param plan - Parsed Plan object
 * @returns List of warning strings (empty if plan is smooth)
 */
export function validatePlanSmoothness(plan: Plan): string[] {
  const warnings: string[] = [];

  // Check for empty plan
  if (!plan.steps || plan.steps.length === 0) {
    warnings.push('Plan has no steps');
    return warnings;
  }

  // Check for very short plans (might be incomplete)
  if (plan.steps.length < 2) {
    warnings.push('Plan has only one step - might be incomplete');
  }

  // Check each step
  let prevAction: string | null = null;
  let prevId: number | undefined = undefined;
  for (const step of plan.steps) {
    const stepLabel = step.id ?? '?';
    // Check for missing verification
    if ((!step.verify || step.verify.length === 0) && step.required !== false) {
      warnings.push(`Step ${stepLabel} has no verification predicates`);
    }

    // Check for consecutive same actions (might indicate loop)
    if (step.action === prevAction && step.action === 'CLICK') {
      warnings.push(`Steps ${prevId ?? '?'} and ${stepLabel} both use ${step.action}`);
    }

    // Check for NAVIGATE without target
    if (step.action === 'NAVIGATE' && !step.target) {
      warnings.push(`Step ${stepLabel} is NAVIGATE but has no target URL`);
    }

    // Check for CLICK without intent
    if (step.action === 'CLICK' && !step.intent) {
      warnings.push(`Step ${stepLabel} is CLICK but has no intent hint`);
    }

    // Check for TYPE_AND_SUBMIT without input
    if (step.action === 'TYPE_AND_SUBMIT' && !step.input) {
      warnings.push(`Step ${stepLabel} is TYPE_AND_SUBMIT but has no input`);
    }

    prevAction = step.action;
    prevId = step.id;
  }

  return warnings;
}

// ---------------------------------------------------------------------------
// Context Formatting
// ---------------------------------------------------------------------------

/**
 * Format snapshot elements for LLM context.
 *
 * Uses compact format: id|role|text|importance|is_primary|bg|clickable|nearby_text|ord|DG|href
 *
 * Uses multi-strategy selection (like Python SDK) to ensure diverse element coverage:
 * 1. Top elements by importance (captures high-priority navigation)
 * 2. Elements from dominant group (captures product listings)
 * 3. Top elements by position (captures visible content regardless of importance)
 *
 * @param elements - Array of snapshot elements
 * @param limit - Maximum number of elements to include
 * @returns Compact string representation
 */
export function selectContextElements(
  elements: SnapshotElement[],
  limit: number = 200
): SnapshotElement[] {
  // Filter to interactive elements
  const interactiveRoles = new Set([
    'button',
    'link',
    'textbox',
    'searchbox',
    'combobox',
    'checkbox',
    'radio',
    'slider',
    'tab',
    'menuitem',
    'option',
    'switch',
    'cell',
    'a',
    'input',
    'select',
    'textarea',
  ]);

  // Roles that should be prioritized (input elements for typing)
  const inputRoles = new Set(['textbox', 'searchbox', 'combobox', 'input', 'textarea']);

  // Include elements that are:
  // 1. Have an interactive role (button, link, textbox, etc.)
  // 2. Are marked as clickable
  // 3. Have an href (links, even if role is not 'link')
  const filtered = elements.filter(el => {
    const role = (el.role || '').toLowerCase();
    const isInteractive = interactiveRoles.has(role);
    const hasHref = Boolean(el.href);
    return isInteractive || el.clickable || hasHref;
  });

  // === Multi-strategy selection (like Python SDK) ===
  const selectedIds = new Set<number>();
  const selected: SnapshotElement[] = [];

  // Helper to add element if not already selected
  const addElement = (el: SnapshotElement): boolean => {
    if (el.id === undefined || selectedIds.has(el.id)) return false;
    selectedIds.add(el.id);
    selected.push(el);
    return true;
  };

  // 1. First, add all input elements (searchbox, textbox, etc.) - these are critical
  for (const el of filtered) {
    const role = (el.role || '').toLowerCase();
    if (inputRoles.has(role)) {
      addElement(el);
    }
  }

  // 2. Top 40 by importance (like Python SDK)
  const byImportance = [...filtered].sort((a, b) => (b.importance || 0) - (a.importance || 0));
  for (const el of byImportance.slice(0, 40)) {
    addElement(el);
  }

  // 3. Elements from dominant group (product listings typically have inDominantGroup=true)
  const dominantGroup = filtered.filter(el => el.inDominantGroup);
  for (const el of dominantGroup) {
    if (selected.length >= 80) break; // Cap at ~80 from dominant group
    addElement(el);
  }

  // 4. Top 30 by position (elements appearing earlier in document, like Python SDK)
  // This ensures we capture visible products even if they have low importance
  // Sort by element ID as proxy for document order (lower ID = earlier in DOM)
  const byPosition = [...filtered].sort((a, b) => {
    // Use element ID as proxy for document order
    return (a.id || 0) - (b.id || 0);
  });
  for (const el of byPosition.slice(0, 30)) {
    addElement(el);
  }

  // 5. Product-like links: links with longer text and product URLs
  // This captures product cards that might have low importance but are clickable
  const productLinks = filtered.filter(el => {
    const role = (el.role || '').toLowerCase();
    const text = (el.text || '').trim();
    const href = (el.href || '').toLowerCase();
    // Links with substantial text (>15 chars) or product URL patterns
    return (
      role === 'link' &&
      (text.length > 15 ||
        href.includes('/dp/') ||
        href.includes('/product/') ||
        href.includes('/item/') ||
        href.includes('/p/'))
    );
  });
  for (const el of productLinks.slice(0, 20)) {
    addElement(el);
  }

  // 6. Fill remaining slots with any remaining elements up to limit
  for (const el of filtered) {
    if (selected.length >= limit) break;
    addElement(el);
  }

  // Format each element
  // Format: id|role|text|importance|is_primary|bg|clickable|nearby_text|ord|DG|href
  // (matches Python SDK format exactly)
  return selected;
}

export function formatContext(elements: SnapshotElement[], limit: number = 200): string {
  const selected = selectContextElements(elements, limit);
  const lines: string[] = [];

  // Add header row
  lines.push('id|role|text|importance|is_primary|bg|clickable|nearby_text|ord|DG|href');
  for (const el of selected) {
    // If element has href, treat as link (like Python SDK)
    const role = el.href ? 'link' : el.role || '';

    // Truncate text to 30 chars (like Python SDK)
    const text = truncateText(el.text || el.name || '', 30);

    // Build line in Python SDK order
    const parts = [
      el.id,
      role,
      text,
      el.importance || 0,
      el.isPrimary ? '1' : '0',
      '', // bg (background color name)
      el.clickable ? '1' : '0',
      truncateText(el.nearbyText || '', 20),
      el.ordinal || '',
      el.inDominantGroup ? '1' : '0',
      compressHref(el.href || ''),
    ];
    lines.push(parts.join('|'));
  }

  return lines.join('\n');
}

export function formatPrunedContext(
  context: Pick<
    PrunedSnapshotContext,
    | 'category'
    | 'elements'
    | 'relaxationLevel'
    | 'rawElementCount'
    | 'prunedElementCount'
    | 'actionableElementCount'
  >
): string {
  return [
    `Category: ${context.category}`,
    `Relaxation: ${context.relaxationLevel}`,
    `Elements: ${context.prunedElementCount}/${context.rawElementCount}`,
    `Actionable: ${context.actionableElementCount}`,
    formatContext(context.elements, context.prunedElementCount || 1),
  ].join('\n');
}

/**
 * Compress href to last path segment (like Python SDK).
 */
function compressHref(href: string): string {
  if (!href) return '';
  href = href.trim();

  // Relative URL - get last segment
  if (href.startsWith('/')) {
    const parts = href.split('/');
    const last = parts[parts.length - 1] || '';
    return last.slice(0, 20);
  }

  // Absolute URL - try to parse
  try {
    const url = new URL(href);
    if (url.pathname && url.pathname !== '/') {
      const parts = url.pathname.replace(/\/$/, '').split('/');
      const last = parts[parts.length - 1] || '';
      return last.slice(0, 20) || url.hostname.slice(0, 15);
    }
    return url.hostname.slice(0, 15);
  } catch {
    return href.slice(0, 20);
  }
}

/**
 * Truncate and sanitize text for LLM context.
 * Removes newlines and excessive whitespace to keep pipe-delimited format intact.
 */
function truncateText(text: string, maxLen: number): string {
  // Replace newlines and multiple spaces with single space
  const sanitized = text
    .replace(/[\r\n]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (sanitized.length <= maxLen) return sanitized;
  return sanitized.slice(0, maxLen - 3) + '...';
}

// Re-export SnapshotElement for backwards compatibility
export type { SnapshotElement } from './plan-models';
