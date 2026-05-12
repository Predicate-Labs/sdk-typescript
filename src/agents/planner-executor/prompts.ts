/**
 * Prompt Builders for PlannerExecutorAgent
 *
 * System and user prompts for Planner and Executor LLMs.
 * Optimized for small local models (4B-8B parameters).
 */

import type { ActionRecord } from './plan-models';
import {
  isExtractionTask,
  isCountingTask,
  getExtractionDomainGuidance,
} from './extraction-keywords';

// ---------------------------------------------------------------------------
// Stepwise Planner Prompt (ReAct-style)
// ---------------------------------------------------------------------------

/**
 * Build system and user prompts for stepwise (ReAct-style) planning.
 *
 * Instead of generating a full plan upfront, this prompt asks the LLM to
 * decide the next single action based on current page state and history.
 *
 * @param goal - The overall task goal
 * @param currentUrl - Current page URL
 * @param pageContext - Compact representation of page elements
 * @param actionHistory - List of previously executed actions
 * @returns Tuple of [systemPrompt, userPrompt]
 */
export function buildStepwisePlannerPrompt(
  goal: string,
  currentUrl: string,
  pageContext: string,
  actionHistory: ActionRecord[]
): [string, string] {
  // Build action history text
  let historyText = '';
  if (actionHistory.length > 0) {
    historyText = 'Actions taken so far:\n';
    for (const rec of actionHistory) {
      const targetStr = rec.target ? `(${rec.target})` : '';
      historyText += `  ${rec.stepNum}. ${rec.action}${targetStr} → ${rec.result}`;
      if (rec.urlAfter) {
        historyText += ` [URL: ${rec.urlAfter.slice(0, 60)}...]`;
      }
      if (rec.extractedData) {
        const preview =
          rec.extractedData.length > 500
            ? rec.extractedData.slice(0, 500) + '...'
            : rec.extractedData;
        historyText += ` [EXTRACTED: ${preview}]`;
      }
      historyText += '\n';
    }
    historyText += '\n';
  }

  // Tight prompt optimized for small local models (7B)
  // NOTE: /no_think at END of system prompt for Qwen3 compatibility
  const system = `You are a browser automation planner. Decide the NEXT action.

Actions:
- NAVIGATE: Go directly to a URL when the next destination is known. Set "target" to the URL.
- CLICK: Click an element. Set "intent" to describe the SPECIFIC element (include label, placeholder, or nearby text, e.g. "email textbox", "display name field", "Next button", NOT just "textbox" or "button"). Set "input" to EXACT text from elements list.
- FILL_FORM: Fill ALL visible form fields and submit. Use for login, signup, checkout, or any multi-field form. Set "fields" to an array of {label, value} pairs. Set "submitText" to the submit button text. Set "verify" to check navigation after submit.
- TYPE: Type text into a SINGLE form field. Prefer FILL_FORM for forms with multiple fields.
- TYPE_AND_SUBMIT: Type text into a search box and submit. Set "input" to the SEARCH QUERY from the goal (NOT the element label).
- SCROLL: Scroll page. Set "direction" to "up" or "down".
- SCROLL_AND_COUNT: Scroll through the ENTIRE page and count items. Use ONLY when the task asks to enumerate items (e.g., "how many listings", "number of results", "count the products"). Do NOT use when "count" is a data value to read (e.g., "calorie count", "word count" = use EXTRACT instead). Set "countTarget" to describe what to count.
- WAIT: Wait for content to appear when a follow-up verification is needed.
- EXTRACT: Extract the requested information from the current page when the task is data collection.
- STUCK: Use only when the page state is blocked and you cannot make safe forward progress.
- DONE: ONLY return DONE when the ENTIRE goal is complete. NOT after just one step.

WHEN TO USE DONE:
- "Add to Cart" task: DONE only AFTER clicking the Add to Cart button
- "Search and click product" task: DONE only AFTER clicking a product link
- "Search only" task: DONE after search results appear
- "Log in" task: DONE only AFTER the page navigates away from /login
- If goal has multiple steps, complete ALL steps before returning DONE

CRITICAL RULE FOR FILL_FORM (PREFERRED for login/signup/checkout):
- Use FILL_FORM when the goal provides values for 2+ form fields (e.g. "username: X, password: Y")
- "fields" is an array of {label, value} where label matches the field's visible text/placeholder
- "submitText" is the text on the submit button (e.g. "Sign in", "Log in", "Submit", "Next")
- The system will find and fill each field by matching label to element text/role
- This is MUCH faster than TYPE one field at a time

CRITICAL RULE FOR TYPE_AND_SUBMIT:
- "input" must be the SEARCH QUERY you want to type (e.g., "wireless headphones")
- "input" is NOT the element label (e.g., NOT "Search Amazon")
- ONLY use if you see a "searchbox" or "textbox" element

CRITICAL RULE FOR CLICK (after search):
- After searching, you are on a RESULTS PAGE. Click a PRODUCT LINK to go to product details.
- Look for LINK elements with product names, prices, or /dp/ URLs
- Set "input" to the product title text from the elements list

CRITICAL RULE FOR ADD TO CART:
- On product page, look for "Add to Cart" or "Add to Bag" buttons
- Set "input" to "Add to Cart" (or exact button text from elements)

Output ONLY valid JSON (no markdown, no \`\`\`):
{"action":"NAVIGATE","target":"https://shop.test/search","verify":[{"predicate":"url_contains","args":["search"]}]}
{"action":"FILL_FORM","fields":[{"label":"username","value":"john@example.com"},{"label":"password","value":"secret123"}],"submitText":"Sign in","verify":[{"predicate":"url_contains","args":["dashboard"]}]}
{"action":"FILL_FORM","fields":[{"label":"email","value":"test@test.com"},{"label":"password","value":"pass123"},{"label":"confirm password","value":"pass123"}],"submitText":"Create Account","verify":[]}
{"action":"TYPE_AND_SUBMIT","intent":"searchbox","input":"wireless headphones","verify":[{"predicate":"url_contains","args":["search"]}]}
{"action":"CLICK","intent":"product link","input":"Sony WH-1000XM4 Wireless...","verify":[]}
{"action":"CLICK","intent":"add to cart button","input":"Add to Cart","verify":[]}
{"action":"SCROLL_AND_COUNT","countTarget":"product listings","goal":"Count total product listings on the page"}
{"action":"DONE","intent":"completed"}

RULES:
1. For TYPE_AND_SUBMIT: "input" = search query from goal (what you want to search for)
2. For CLICK: "input" = exact text from elements list
3. Do NOT type into "email" or "newsletter" fields
4. Do NOT repeat the same action twice
5. Include "verify" when you know a simple URL or element predicate that proves success; otherwise use []
6. Include planner metadata when useful: "target", "required", "stop_if_true", "optional_substeps", "heuristic_hints"
7. "heuristic_hints" entries may use snake_case fields: "intent_pattern", "text_patterns", "role_filter", "attribute_patterns", "priority"
8. Output ONLY JSON - no 时光网 tags, no markdown, no prose
9. Do NOT output 时光网 or any reasoning
10. Do NOT return DONE until ALL parts of the goal are complete
11. Never copy example URLs from these instructions. Only NAVIGATE to a URL from the user's task, the current page, or a visible element.
12. PREFER FILL_FORM for login/signup/checkout forms with 2+ fields. Do NOT use multiple TYPE actions when FILL_FORM can do it in one step.
13. "intent" must be SPECIFIC: describe the element with its label or context (e.g., "email field", "plan dropdown", "Next button on step 2")
14. Treat history results "success", "skipped", and "vision_fallback" as already satisfied. Do not repeat those steps; choose the next incomplete part of the goal.`;

  // Inject extraction-specific guidance when the goal is an extraction task
  const extractionGuidance = isExtractionTask(goal)
    ? getExtractionDomainGuidance(isCountingTask(goal))
    : '';

  // NOTE: /no_think MUST be at the START of user message for Qwen3 models
  const user = `/no_think
Goal: ${goal}

Current URL: ${currentUrl}
${extractionGuidance}
${historyText}Current page elements (ID|role|text|importance|clickable|...):
${pageContext}

Based on the goal and current page state, what is the NEXT action to take?`;

  return [system, user];
}

// ---------------------------------------------------------------------------
// Executor Prompt
// ---------------------------------------------------------------------------

/**
 * Build system and user prompts for the Executor LLM.
 *
 * @param goal - Human-readable goal for this step
 * @param intent - Intent hint for element selection (optional)
 * @param compactContext - Compact representation of page elements
 * @param inputText - For TYPE_AND_SUBMIT: text to type. For CLICK: target text to match (optional)
 * @param category - Task category for category-specific hints (optional)
 * @param actionType - Action type (CLICK, TYPE_AND_SUBMIT, etc.)
 * @returns Tuple of [systemPrompt, userPrompt]
 */
export function buildExecutorPrompt(
  goal: string,
  intent: string | undefined,
  compactContext: string,
  inputText?: string,
  category?: string,
  actionType?: string
): [string, string] {
  const intentLine = intent ? `Intent: ${intent}\n` : '';

  // For CLICK actions, inputText is target to match (not text to type)
  const isTypeAction = actionType === 'TYPE_AND_SUBMIT' || actionType === 'TYPE';
  let inputLine = '';
  if (isTypeAction && inputText) {
    inputLine = `Text to type: "${inputText}"\n`;
  } else if (inputText) {
    inputLine = `Target to find: "${inputText}"\n`;
  }

  // Get category-specific hints
  const categoryHints = getCategoryExecutorHints(category);
  const categoryLine = categoryHints ? `${categoryHints}\n` : '';

  // Build system prompt based on action type
  let system: string;

  if (isTypeAction && inputText) {
    // TYPE action - find the INPUT element (textbox/combobox), not the submit button
    system = `You are an executor for browser automation.
Task: Find the INPUT element (textbox, combobox, searchbox) to type into.
Return ONLY ONE line: TYPE(<id>, "text")
IMPORTANT: Return the ID of the INPUT/TEXTBOX element, NOT the submit button.
CRITICAL - AVOID these fields (they are NOT search boxes):
- Fields with 'email', 'newsletter', 'subscribe', 'signup' in the text
- Fields labeled 'Your email address', 'Email', 'Enter your email'
- Fields in footer/newsletter sections
Treat searchbox/combobox roles as valid search inputs.
Accept search labels such as 'Search', 'Search Amazon', 'Search products', 'Search store', or similar search-specific wording in text, placeholder, name, or aria-label.
If NO search field exists, return NONE instead of guessing.
If you output anything else, the action fails.
Do NOT output <think> or any reasoning.
No prose, no markdown, no extra whitespace.
Example: TYPE(42, "hello world")`;
  } else {
    // CLICK action (most common)
    const searchKeywords = ['search', 'magnify', 'magnifier', 'find'];
    const productKeywords = ['product', 'item', 'result', 'listing'];
    const addToCartKeywords = ['add to cart', 'add to bag', 'add to basket', 'buy now'];

    const isSearchAction =
      (intent && searchKeywords.some(kw => intent.toLowerCase().includes(kw))) ||
      searchKeywords.some(kw => goal.toLowerCase().includes(kw));

    const isProductAction =
      (intent && productKeywords.some(kw => intent.toLowerCase().includes(kw))) ||
      productKeywords.some(kw => goal.toLowerCase().includes(kw));

    const isAddToCartAction =
      (intent && addToCartKeywords.some(kw => intent.toLowerCase().includes(kw))) ||
      addToCartKeywords.some(kw => goal.toLowerCase().includes(kw));

    const isTextMatchingAction = intent && intent.toLowerCase().includes('matching');
    const hasTargetText = Boolean(inputText);

    if (isSearchAction) {
      system = `You are an executor for browser automation.
Return ONLY a single-line CLICK(id) action.
If you output anything else, the action fails.
Do NOT output <think> or any reasoning.
SEARCH ICON HINTS: Look for links/buttons with 'search' in text/href, or icon-only elements (text='0' or empty) with 'search' in href.
Output MUST match exactly: CLICK(<digits>) with no spaces.
Example: CLICK(12)`;
    } else if (isTextMatchingAction || hasTargetText) {
      // When planner specifies target text, executor must match it
      const targetText = inputText || '';
      system = `You are an executor for browser automation.
Return ONLY a single-line CLICK(id) action.
If you output anything else, the action fails.
Do NOT output <think> or any reasoning.
CRITICAL: Find an element with text matching '${targetText}'.
- Look for: product titles, category names, link text, button labels
- Text must contain the target words (case-insensitive partial match is OK)
- If multiple elements match, choose the one with the strongest text overlap and the most specific visible label
- If NO element contains the target text, return NONE instead of clicking something random
Output: CLICK(<digits>) or NONE
Example: CLICK(42) or NONE`;
    } else if (isProductAction) {
      system = `You are an executor for browser automation.
Return ONLY a single-line CLICK(id) action.
If you output anything else, the action fails.
Do NOT output <think> or any reasoning.
PRODUCT CLICK HINTS:
- Look for LINK elements (role=link) with product IDs in href (e.g., /7027762, /dp/B...)
- Prefer links with delivery info text like 'Delivery', 'Ships to Store', 'Get it...'
- These are inside product cards and will navigate to product detail pages
- AVOID buttons like 'Search', 'Shop', category buttons, or filter buttons
- AVOID image slider options (slider image 1, 2, etc.)
Output MUST match exactly: CLICK(<digits>) with no spaces.
Example: CLICK(1268)`;
    } else if (isAddToCartAction) {
      system = `You are an executor for browser automation.
Return ONLY a single-line CLICK(id) action.
If you output anything else, the action fails.
Do NOT output <think> or any reasoning.
ADD TO CART HINTS:
- FIRST: Look for buttons with text: 'Add to Cart', 'Add to Bag', 'Add to Basket', 'Buy Now'
- If found, click that button directly
- FALLBACK: If NO 'Add to Cart' button is visible, you are likely on a SEARCH RESULTS page
  - In this case, click a PRODUCT LINK to go to the product details page first
  - Look for LINK elements with product IDs in href (e.g., /7027762, /dp/B...)
  - Prefer links with product names, prices, or delivery info
- AVOID: 'Search' buttons, category buttons, filter buttons, pagination
Output MUST match exactly: CLICK(<digits>) with no spaces.
Example: CLICK(42)`;
    } else {
      system = `You are an executor for browser automation.
Return ONLY a single-line CLICK(id) action.
If you output anything else, the action fails.
Do NOT output <think> or any reasoning.
No prose, no markdown, no extra whitespace.
Output MUST match exactly: CLICK(<digits>) with no spaces.
Example: CLICK(12)`;
    }
  }

  // Build action instruction based on action type
  let actionInstruction: string;
  if (isTypeAction && inputText) {
    actionInstruction = `Return TYPE(id, "${inputText}"):`;
  } else if (inputText) {
    actionInstruction = `Return CLICK(id) for element matching "${inputText}", or NONE if not found:`;
  } else {
    actionInstruction = 'Return CLICK(id):';
  }

  // NOTE: /no_think MUST be at the START of user message for Qwen3 models
  const user = `/no_think
Goal: ${goal}
${intentLine}${categoryLine}${inputLine}
Elements:
${compactContext}

${actionInstruction}`;

  return [system, user];
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/**
 * Get category-specific hints for the executor.
 */
function getCategoryExecutorHints(category?: string): string {
  if (!category) return '';

  const categoryLower = category.toLowerCase();

  const hints: Record<string, string> = {
    shopping: "Priority: 'Add to Cart', 'Buy Now', 'Add to Bag', product links, price elements.",
    checkout: "Priority: 'Checkout', 'Proceed to Checkout', 'Place Order', payment fields.",
    form_filling: 'Priority: input fields, textboxes, submit/send buttons, form labels.',
    search: 'Priority: search box, search button, result links, filter controls.',
    auth: 'Priority: username/email field, password field, sign in/login button.',
    extraction: 'Priority: data elements, table cells, list items, content containers.',
    navigation: 'Priority: navigation links, menu items, breadcrumbs.',
  };

  return hints[categoryLower] || '';
}

// ---------------------------------------------------------------------------
// Stepwise Planner Response Schema
// ---------------------------------------------------------------------------

/**
 * Expected response format from stepwise planner.
 */
export interface StepwisePlannerResponse {
  id?: number;
  goal?: string;
  action:
    | 'NAVIGATE'
    | 'CLICK'
    | 'TYPE'
    | 'TYPE_AND_SUBMIT'
    | 'FILL_FORM'
    | 'SCROLL'
    | 'SCROLL_AND_COUNT'
    | 'PRESS'
    | 'WAIT'
    | 'EXTRACT'
    | 'STUCK'
    | 'DONE';
  target?: string;
  intent?: string;
  input?: string;
  direction?: 'up' | 'down';
  fields?: Array<{ label: string; value: string }>;
  submitText?: string;
  countTarget?: string;
  verify?: Array<{ predicate: string; args: unknown[] }>;
  required?: boolean;
  stopIfTrue?: boolean;
  optionalSubsteps?: Array<Record<string, unknown>>;
  heuristicHints?: Array<Record<string, unknown>>;
  reasoning?: string;
}

export function buildVisionExecutorPrompt(
  goal: string,
  intent: string | undefined,
  inputText?: string,
  actionType?: string
): [string, string] {
  const isTypeAction = actionType === 'TYPE_AND_SUBMIT' || actionType === 'TYPE';

  const system = `You are a vision-based executor for browser automation.
You receive a screenshot of the current browser viewport.
Your job is to identify the correct location to act on based on the goal and return a coordinate-based action.

SUPPORTED ACTIONS (return ONLY ONE line):
- CLICK_XY(x, y) — click at pixel coordinates (x, y) in the viewport
- CLICK_RECT(x, y, w, h) — click the center of rectangle at (x, y) with width w and height h
- TYPE_AT("text") — type text at the currently focused position
- PRESS("key") — press a keyboard key (e.g., "Enter", "Tab", "Escape")
- NONE — if you cannot determine the correct action from the screenshot

RULES:
- Estimate pixel coordinates from the screenshot
- Coordinates (0,0) are the top-left corner of the viewport
- Be precise with coordinates — aim for the center of the target element
- Do NOT output any reasoning or prose
- Do NOT use CLICK(id) — only coordinate-based actions
- For form fields, match the field by its LABEL or PLACEHOLDER text, not just any input
- Example: CLICK_XY(450, 320)`;

  let actionHint: string;
  if (isTypeAction && inputText) {
    actionHint = `Return CLICK_XY(x, y) for the INPUT FIELD matching: "${intent || goal}". Look at field labels/placeholders to find the correct field.`;
  } else {
    actionHint = `Return CLICK_XY(x, y) for the element matching: "${intent || goal}"`;
  }

  const user = `/no_think
Goal: ${goal}
${intent ? `Intent: ${intent}` : ''}
${inputText ? `Input: "${inputText}"` : ''}

${actionHint}`;

  return [system, user];
}
