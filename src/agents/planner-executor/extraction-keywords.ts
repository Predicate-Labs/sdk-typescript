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
  'note',
  'record',
  'summarize',
  'outline',
  'compare',
  'calculate',
];

/**
 * Multi-word phrases that strongly indicate extraction.
 * These are checked first via substring matching.
 */
const EXTRACTION_PHRASES: readonly string[] = [
  'what is',
  'what are',
  "what's",
  'what does',
  'show me',
  'tell me',
  'find the',
  'get the',
  'read the',
  'list the',
  'note down',
  'write down',
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
  'pricing',
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
  'benefits',
  'services',
  'courses',
  'games',
  'guidelines',
  'steps',
  'tips',
  'events',
  'options',
  'perks',
  'faqs',
  'specifications',
  'features',
  'formats',
  'cities',
  'countries',
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
  const hasFormFillSignal = FORM_FILL_SIGNALS.some(signal => {
    if (signal.endsWith(' ')) {
      return taskLower.includes(signal);
    }
    return new RegExp(`\\b${escapeRegExp(signal)}\\b`).test(taskLower);
  });
  if (hasFormFillSignal) {
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
    taskLower.includes('review of') ||
    taskLower.includes('summarize') ||
    taskLower.includes('note the') ||
    taskLower.includes('list the')
  );
}

/**
 * Get extraction-specific domain guidance for the planner prompt.
 *
 * This is injected into the planner system prompt when the task
 * is detected as an extraction task, instructing the planner to
 * use EXTRACT instead of CLICK for data that is already visible.
 */
export function getExtractionDomainGuidance(includeCounting = false): string {
  const countingSection = includeCounting
    ? `

STEP 3 - COUNTING ACROSS FULL PAGE:
If the task asks to COUNT items (e.g., "how many listings", "number of results", "count the products", "total number of entries"):
- Use SCROLL_AND_COUNT instead of EXTRACT
- Set "countTarget" to describe what to count (e.g., "listings", "products", "articles")
- The system will scroll through the entire page and sum up counts
- Do NOT use EXTRACT for counting tasks — EXTRACT only sees the current viewport

Example - count all listings:
Goal: "note how many listings are available"
Current URL: alibaba.com/search?SearchText=smartphones (correct page)
{"action":"SCROLL_AND_COUNT","countTarget":"product listings","goal":"Count total product listings","verify":[]}
`
    : '';

  return `

IMPORTANT: Extraction Task Planning Rules
=======================================

STEP 1 - CHECK CURRENT URL:
Before choosing an action, compare the Current URL to the goal.
- Does the current page contain the data requested?
- If the goal mentions a specific section/page (e.g., "show hn", "top stories", "/show"), check if the URL matches.
- If you are NOT on the right page, NAVIGATE or CLICK to navigate to the correct page first.

STEP 2 - EXTRACT VISIBLE DATA:
If the data is VISIBLE in the page context:
- Use EXTRACT directly - no clicking needed
- The EXTRACT action reads visible text from the current page

CRITICAL: Do NOT click on links to external sites when extracting.
- Post/article titles often link to EXTERNAL sites
- To extract a title that is visible, use EXTRACT directly on the current page
- Only click if you need to navigate to a detail page to access the data (e.g., nutritional info on a recipe page)

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

Example - data on a detail page, need to click first:
Goal: "summarize the calorie count from a recipe's nutritional information"
Current URL: allrecipes.com/search?q=cookies (search results, not a recipe page)
{"action":"CLICK","intent":"first recipe link","input":"Best Chocolate Chip Cookies","verify":[],"reasoning":"need to navigate to recipe detail page for nutritional info"}
${countingSection}
`;
}

// ---------------------------------------------------------------------------
// Counting Task Detection
// ---------------------------------------------------------------------------

const COUNTING_PHRASES: readonly string[] = [
  'how many',
  'how much',
  'number of',
  'count the',
  'count all',
  'count each',
  'count every',
  'total number',
  'total count of',
  'how numerous',
];

// NOTE: "count of" is intentionally excluded from COUNTING_PHRASES because
// it is ambiguous: "count of items" (verb phrase) vs "word count of the
// article" (noun compound). The "number of" phrase covers the counting
// semantics of "count of". Bare "count" is handled by Tier 3 below.

const COUNTING_VERBS: readonly string[] = ['tally', 'enumerate'];

// Words that can syntactically precede "count" when it is used as a VERB
// (imperative, infinitive, or after an auxiliary/modal). This set is finite
// and well-defined in English grammar. Any word NOT in this set, appearing
// immediately before "count", indicates a noun compound like "calorie count",
// "word count", "error count" — regardless of what the modifier noun is.
const COUNT_VERB_PRECEDERS = new Set([
  // Infinitive marker
  'to',
  // Modals and auxiliaries
  'can',
  'could',
  'will',
  'would',
  'shall',
  'should',
  'must',
  'may',
  'might',
  'do',
  'did',
  'does',
  'have',
  'has',
  'had',
  // Polite / adverbial markers
  'please',
  'just',
  'also',
  'even',
  'still',
  'not',
  'never',
  'only',
  // Conjunctions that continue an action sequence
  'and',
  'or',
  'but',
  'then',
  // Verbs that take infinitive complements
  'going',
  'try',
  'want',
  'need',
  'help',
  'let',
  'plan',
  'attempt',
  'aim',
  'start',
  'begin',
]);

export function isCountingTask(task: string): boolean {
  if (!task) return false;
  const taskLower = task.toLowerCase();

  // Tier 1: Unambiguous counting phrases (substring match)
  if (COUNTING_PHRASES.some(phrase => taskLower.includes(phrase))) {
    return true;
  }

  // Tier 2: Unambiguous counting verbs (word boundary match)
  if (
    COUNTING_VERBS.some(verb =>
      new RegExp(`\\b${escapeRegExp(verb)}(s|ed|ing)?\\b`).test(taskLower)
    )
  ) {
    return true;
  }

  // Tier 3: Context-aware "count" — distinguish verb from noun compound.
  //
  // In English, "[noun] count" is a compound noun meaning "the count of
  // [noun]" (e.g., "calorie count", "word count", "page count", "error
  // count"). As a VERB, "count" appears in imperative position (start of
  // clause) or after auxiliaries/modals/infinitive markers.
  //
  // Strategy: find every "count" preceded by a word. If ALL preceding words
  // are NOT in COUNT_VERB_PRECEDERS, every occurrence is a noun compound and
  // this is NOT a counting task. If any occurrence has no preceding word
  // (imperative) or is preceded by a verb preceder, it IS a counting task.
  if (/\bcount(s|ed|ing)?\b/i.test(taskLower)) {
    const precedingMatches = [...taskLower.matchAll(/\b([a-z]+)\s+count(s|ed|ing)?\b/g)];
    if (precedingMatches.length > 0) {
      const allAreNounCompounds = precedingMatches.every(m => !COUNT_VERB_PRECEDERS.has(m[1]));
      if (allAreNounCompounds) {
        return false;
      }
    }
    // "count" with no preceding word (imperative) or preceded by a verb marker
    return true;
  }

  return false;
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
