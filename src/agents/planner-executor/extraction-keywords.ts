/**
 * Text extraction keyword detection for EXTRACT action routing.
 *
 * Ported from sdk-python predicate/agents/planner_executor_agent.py
 * (_is_text_extraction_task / TEXT_EXTRACTION_KEYWORDS).
 *
 * Determines whether a planner step should use text-based extraction
 * (reading page content as markdown and extracting via LLM) rather
 * than element-targeted actions like CLICK or TYPE.
 */

// ---------------------------------------------------------------------------
// Keyword Categories
// ---------------------------------------------------------------------------

/**
 * Strong extraction verbs that alone indicate a data extraction task.
 * These are unambiguous — if the task says "extract" or "scrape", it's extraction.
 */
const EXTRACTION_VERBS: readonly string[] = [
  'extract',
  'read',
  'parse',
  'scrape',
  'retrieve',
  'capture',
  'grab',
  'copy',
  'pull',
];

/**
 * Ambiguous verbs that need a content noun to confirm extraction intent.
 * E.g., "find" alone is ambiguous, but "find the title" = extraction.
 */
const AMBIGUOUS_VERBS: readonly string[] = [
  'find',
  'get',
  'fetch',
  'list',
  'show',
  'tell',
  'display',
  'provide',
  'report',
  'give',
  'identify',
  'collect',
  'gather',
  'return',
  'output',
];

/**
 * Multi-word phrases that strongly indicate extraction.
 * These are checked first via substring matching.
 */
const EXTRACTION_PHRASES: readonly string[] = [
  'what is',
  'what are',
  "what's",
  'show me',
  'tell me',
  'find the',
  'get the',
  'read the',
  'list the',
  'title of',
  'price of',
  'name of',
  'content of',
  'find the text',
  'find the title',
  'find the price',
  'find the name',
  'how many',
  'how much',
  'sale price',
  'sale prices',
  'first 5',
  'first 10',
  'first 3',
  'top 5',
  'top 10',
  'top 3',
];

/**
 * Content/data nouns that indicate the task wants specific information.
 * Used alongside ambiguous verbs to confirm extraction intent.
 */
const CONTENT_NOUNS: readonly string[] = [
  'title',
  'headline',
  'heading',
  'text',
  'content',
  'body',
  'paragraph',
  'article',
  'post',
  'message',
  'description',
  'summary',
  'excerpt',
  'price',
  'cost',
  'amount',
  'name',
  'label',
  'value',
  'number',
  'date',
  'time',
  'address',
  'email',
  'phone',
  'rating',
  'review',
  'comment',
  'author',
  'username',
  'table',
  'row',
  'column',
  'item',
  'entry',
  'record',
  'population',
  'score',
  'count',
  'total',
  'average',
  'statistic',
  'stat',
  'link',
  'url',
  'image',
  'photo',
  'product',
  'results',
  'listings',
];

/**
 * Legacy keyword list kept for backward compatibility with tests.
 * @deprecated Use the categorised constants above for new code.
 */
export const TEXT_EXTRACTION_KEYWORDS: readonly string[] = [
  ...EXTRACTION_VERBS,
  ...EXTRACTION_PHRASES,
  ...CONTENT_NOUNS,
];

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/**
 * Determine if a task/step description is a text extraction task.
 *
 * Uses a tiered detection strategy to balance precision and recall:
 *
 * 1. **Strong extraction phrases** ("what is", "find the", "title of"):
 *    These multi-word patterns are highly specific → immediate match.
 *
 * 2. **Strong extraction verbs** ("extract", "scrape", "read"):
 *    These verbs unambiguously indicate extraction → match on their own.
 *
 * 3. **Ambiguous verb + content noun combos** ("find" + "title"):
 *    Verbs like "find" or "get" are ambiguous alone, but when paired with
 *    a content noun ("title", "price", "name") they indicate extraction.
 *    This prevents false positives like "find a product" (no content noun).
 *
 * Ported from sdk-python `_is_text_extraction_task()` with improved precision.
 *
 * @param task - The task or step description to analyse
 * @returns true if this looks like a text extraction task
 */
const FORM_FILL_SIGNALS: readonly string[] = [
  'form',
  'fill',
  'submit',
  'onboarding',
  'sign up',
  'signup',
  'register',
  'checkbox',
  'dropdown',
  'radio button',
  'next button',
  'click the',
  'type ',
  'enter ',
];

export function isTextExtractionTask(task: string): boolean {
  if (!task) {
    return false;
  }

  const taskLower = task.toLowerCase();

  // Form-fill negative signal: if the task clearly involves filling a form,
  // it's not extraction even if it contains extraction-like keywords
  // (e.g., "Display name", "email" are field labels, not extraction targets)
  if (FORM_FILL_SIGNALS.some(signal => taskLower.includes(signal))) {
    return false;
  }

  // Tier 1: Strong extraction phrases (multi-word substring match)
  for (const phrase of EXTRACTION_PHRASES) {
    if (taskLower.includes(phrase)) {
      return true;
    }
  }

  // Tier 2: Strong extraction verbs (word boundary match)
  for (const verb of EXTRACTION_VERBS) {
    if (new RegExp(`\\b${escapeRegExp(verb)}(s|ed|ing)?\\b`).test(taskLower)) {
      return true;
    }
  }

  // Tier 3: Ambiguous verb + content noun combo
  // E.g., "find the title" = yes, "find a product" = no, "list all reviews" = yes
  const hasAmbiguousVerb = AMBIGUOUS_VERBS.some(verb =>
    new RegExp(`\\b${escapeRegExp(verb)}\\b`).test(taskLower)
  );
  if (hasAmbiguousVerb) {
    const hasContentNoun = CONTENT_NOUNS.some(noun =>
      new RegExp(`\\b${escapeRegExp(noun)}(s|es)?\\b`).test(taskLower)
    );
    if (hasContentNoun) {
      return true;
    }
  }

  return false;
}

// ---------------------------------------------------------------------------
// Extraction prompt builder
// ---------------------------------------------------------------------------

/**
 * Build the executor prompt for extracting text from page content.
 *
 * @param pageContent - Page content (markdown or compact representation)
 * @param extractQuery - What to extract
 * @returns Tuple of [systemPrompt, userPrompt]
 */
export function buildExtractionPrompt(pageContent: string, extractQuery: string): [string, string] {
  // NOTE: /no_think MUST be at the START of user message for Qwen3 models.
  // Without it, Qwen3 puts the answer in <think/> tags and content is empty.
  const system = `You extract specific text from page content. Return only the extracted text. Do NOT output any thinking, reasoning, or explanation.`;

  const user = `/no_think
You are a text extraction assistant. Given the page content below, extract the specific information requested.

PAGE CONTENT:
${pageContent}

EXTRACTION REQUEST:
${extractQuery}

INSTRUCTIONS:
1. Read the content carefully
2. Find and extract ONLY the specific information requested
3. Return ONLY the extracted text, nothing else
4. If the information is not found, return "NOT_FOUND"

EXTRACTED TEXT:`;

  return [system, user];
}

/**
 * Check if a task description contains extraction-related keywords
 * that should trigger extraction-specific planner guidance.
 */
export function isExtractionTask(task: string): boolean {
  if (!task) return false;
  const taskLower = task.toLowerCase();
  return (
    taskLower.includes('extract') ||
    taskLower.includes('get the') ||
    taskLower.includes('what is') ||
    taskLower.includes('read the') ||
    taskLower.includes('find the text') ||
    taskLower.includes('scrape') ||
    taskLower.includes('title of') ||
    taskLower.includes('price of') ||
    taskLower.includes('name of') ||
    taskLower.includes('content of') ||
    taskLower.includes('headline of') ||
    taskLower.includes('rating of') ||
    taskLower.includes('review of')
  );
}

/**
 * Get extraction-specific domain guidance for the planner prompt.
 *
 * This is injected into the planner system prompt when the task
 * is detected as an extraction task, instructing the planner to
 * use EXTRACT instead of CLICK for data that is already visible.
 */
export function getExtractionDomainGuidance(): string {
  return `

IMPORTANT: Extraction Task Planning Rules
========================================

STEP 1 - CHECK CURRENT URL:
Before choosing an action, compare the Current URL to the goal.
- Does the current page contain the data requested?
- If the goal mentions a specific section/page (e.g., "show hn", "top stories", "/show"), check if the URL matches.
- If you are NOT on the right page, NAVIGATE to the correct URL first.

STEP 2 - EXTRACT VISIBLE DATA:
If the data is VISIBLE in the page context:
- Use EXTRACT directly - no clicking needed
- The EXTRACT action reads visible text from the current page

CRITICAL: Do NOT click on links to external sites when extracting.
- Post/article titles often link to EXTERNAL sites
- To extract a title that is visible, use EXTRACT directly on the current page
- Only click if you need to navigate to a detail page (e.g., for comments)

Example - wrong page, need to navigate first:
Goal: "extract the title of the first showhn post on hackernews show"
Current URL: news.ycombinator.com/news (wrong page, need /show)
{"action":"NAVIGATE","target":"https://news.ycombinator.com/show","verify":[{"predicate":"url_contains","args":["show"]}],"reasoning":"navigate to Show HN page"}

Example - on correct page, extract directly:
Goal: "extract the title of the first showhn post"
Current URL: news.ycombinator.com/show (correct page, data visible)
{"action":"EXTRACT","target":"first ShowHN post title","goal":"Extract the title of the first ShowHN post","verify":[],"reasoning":"data is visible on current page"}

Example - product price on listing page:
Goal: "find the price of the first laptop"
Current URL: store.com/laptops (correct page, prices visible)
{"action":"EXTRACT","target":"price of first laptop","goal":"Extract the price of the first laptop listing","verify":[],"reasoning":"prices are visible in listing elements"}
`;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Escape special regex characters in a string.
 */
function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
