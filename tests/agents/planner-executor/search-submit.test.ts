import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  StepStatus,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';
import {
  isSearchLikeTypeAndSubmit,
  isUrlChangeRelevantToIntent,
} from '../../../src/agents/planner-executor/boundary-detection';
import { ReplanPatchSchema } from '../../../src/agents/planner-executor/plan-models';
import { normalizeReplanPatch } from '../../../src/agents/planner-executor/plan-utils';
import type { SnapshotElement } from '../../../src/agents/planner-executor/plan-models';

class ProviderStub extends LLMProvider {
  private responses: string[];
  public calls: Array<{ system?: string; user?: string; options?: any }> = [];
  public imageCalls: Array<{ system?: string; user?: string; imageBase64: string; options?: any }> =
    [];
  private readonly vision: boolean;

  constructor(responses: string[] = [], options: { vision?: boolean } = {}) {
    super();
    this.responses = [...responses];
    this.vision = options.vision ?? false;
  }

  get modelName(): string {
    return 'stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return this.vision;
  }

  async generate(
    system?: string,
    user?: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    this.calls.push({ system, user, options });
    const content = this.responses.length
      ? this.responses.shift()!
      : JSON.stringify({ action: 'DONE' });
    return {
      content,
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    };
  }

  async generateWithImage(
    system: string,
    user: string,
    imageBase64: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    this.imageCalls.push({ system, user, imageBase64, options });
    const content = this.responses.length ? this.responses.shift()! : 'NONE';
    return {
      content,
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    };
  }
}

class AdaptiveProviderStub extends LLMProvider {
  public calls: Array<{ system?: string; user?: string; options?: any }> = [];

  get modelName(): string {
    return 'adaptive-stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  async generate(
    system?: string,
    user?: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    this.calls.push({ system, user, options });
    const content =
      this.calls.length === 1
        ? JSON.stringify({
            action: 'TYPE',
            intent: 'email field',
            input: 'user@example.com',
            verify: [{ predicate: 'element_exists', args: ['textbox', 'Display name'] }],
          })
        : user?.includes('TYPE(user@example.com) → skipped')
          ? JSON.stringify({ action: 'DONE', reasoning: 'stale email skip was accepted' })
          : JSON.stringify({
              action: 'TYPE',
              intent: 'email field',
              input: 'user@example.com',
              verify: [{ predicate: 'element_exists', args: ['textbox', 'Email'] }],
            });

    return {
      content,
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    };
  }
}

class RuntimeStub implements AgentRuntime {
  public currentUrl: string;
  public gotoCalls: string[] = [];
  public clickCalls: number[] = [];
  public coordinateClickCalls: Array<{ x: number; y: number }> = [];
  public typeCalls: Array<{ elementId: number; text: string }> = [];
  public selectCalls: Array<{ elementId: number; value: string }> = [];
  public coordinateTypeCalls: string[] = [];
  public keyCalls: string[] = [];

  constructor(
    initialUrl: string,
    private readonly snapshotFactory: (runtime: RuntimeStub) => Snapshot | null,
    private readonly handlers: {
      onClick?: (elementId: number, runtime: RuntimeStub) => Promise<void> | void;
      onType?: (elementId: number, text: string, runtime: RuntimeStub) => Promise<void> | void;
      onPressKey?: (key: string, runtime: RuntimeStub) => Promise<void> | void;
    } = {}
  ) {
    this.currentUrl = initialUrl;
  }

  async snapshot(): Promise<Snapshot | null> {
    const snap = this.snapshotFactory(this);
    if (snap?.url) {
      this.currentUrl = snap.url;
    }
    return snap;
  }

  async goto(url: string): Promise<void> {
    this.gotoCalls.push(url);
    this.currentUrl = url;
  }

  async click(elementId: number): Promise<void> {
    this.clickCalls.push(elementId);
    await this.handlers.onClick?.(elementId, this);
  }

  async clickCoordinate(x: number, y: number): Promise<void> {
    this.coordinateClickCalls.push({ x, y });
  }

  async type(elementId: number, text: string): Promise<void> {
    this.typeCalls.push({ elementId, text });
    await this.handlers.onType?.(elementId, text, this);
  }

  async selectOption(elementId: number, value: string): Promise<void> {
    this.selectCalls.push({ elementId, value });
  }

  async typeCoordinate(text: string): Promise<void> {
    this.coordinateTypeCalls.push(text);
  }

  async pressKey(key: string): Promise<void> {
    this.keyCalls.push(key);
    await this.handlers.onPressKey?.(key, this);
  }

  async scroll(): Promise<void> {}

  async getCurrentUrl(): Promise<string> {
    return this.currentUrl;
  }

  async getViewportHeight(): Promise<number> {
    return 1000;
  }

  async scrollBy(): Promise<boolean> {
    return true;
  }
}

class MarkdownRuntimeStub extends RuntimeStub {
  constructor(
    initialUrl: string,
    snapshotFactory: (runtime: RuntimeStub) => Snapshot | null,
    private readonly markdown: string
  ) {
    super(initialUrl, snapshotFactory);
  }

  async readMarkdown(): Promise<string> {
    return this.markdown;
  }
}

class NullMarkdownRuntimeStub extends RuntimeStub {
  async readMarkdown(): Promise<string | null> {
    return null;
  }
}

function makeSnapshot(
  url: string,
  elements: Snapshot['elements'],
  extra: Partial<Snapshot> = {}
): Snapshot {
  return {
    url,
    title: 'Test Page',
    elements,
    ...extra,
  };
}

describe('PlannerExecutorAgent search submission parity', () => {
  it('applies search filter controls in mixed FILL_FORM plans before submitting', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [
          { label: 'any_field', value: 'Space images' },
          { label: 'All mediatypes', value: 'IMAGES' },
        ],
        submitText: 'Search',
        verify: [{ predicate: 'url_contains', args: ['search'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted once' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt =>
        makeSnapshot(rt.currentUrl, [
          {
            id: 5,
            role: 'textbox',
            text: 'optional_field3_q',
            name: 'optional_field3_q',
            clickable: false,
          },
          { id: 8, role: 'searchbox', text: 'any field', name: 'q', clickable: true },
          { id: 15, role: 'radio', text: 'texts', name: 'mediatype', nearbyText: 'All mediatypes' },
          { id: 16, role: 'radio', text: 'image', name: 'mediatype', nearbyText: 'All mediatypes' },
          { id: 17, role: 'button', text: 'Search', clickable: true },
        ]),
      {
        onClick: (elementId, rt) => {
          if (elementId === 17) {
            rt.currentUrl = 'https://example.test/search?q=Space+images&mediatype=image';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 8, text: 'Space images' }]);
    expect(runtime.clickCalls).toEqual([16, 17]);
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'FILL_FORM(TYPE(8, "Space images") -> CLICK(16, IMAGES) -> CLICK(17))'
    );
  });

  it('sets select-style search filters in mixed FILL_FORM plans before submitting', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [
          { label: 'any_field', value: 'Space images' },
          { label: 'All mediatypes', value: 'IMAGES' },
        ],
        submitText: 'Search',
        verify: [{ predicate: 'url_contains', args: ['search'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted once' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 8, role: 'searchbox', text: 'any field', name: 'q', clickable: true },
          {
            id: 14,
            role: 'select',
            text: 'All mediatypes Texts Collection Software Image Audio',
            name: 'mediatype',
            nearbyText: 'All mediatypes',
          },
          { id: 17, role: 'button', text: 'Search', clickable: true },
        ]),
      {
        onClick: (elementId, rt) => {
          if (elementId === 17) {
            rt.currentUrl = 'https://example.test/search?q=Space+images&mediatype=image';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 8, text: 'Space images' }]);
    expect(runtime.selectCalls).toEqual([{ elementId: 14, value: 'IMAGES' }]);
    expect(runtime.clickCalls).toEqual([17]);
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'FILL_FORM(TYPE(8, "Space images") -> SELECT(14, IMAGES) -> CLICK(17))'
    );
  });

  it('prefers the visible any-field query input over raw q fields in advanced search forms', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [
          { label: 'any_field', value: 'Space images' },
          { label: 'All mediatypes', value: 'image' },
        ],
        submitText: 'Search',
        verify: [{ predicate: 'url_contains', args: ['Space'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'filtered search submitted once' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 8, role: 'searchbox', text: 'q', name: 'q', clickable: true },
          { id: 13, role: 'textbox', text: 'any_field', name: 'any_field', clickable: true },
          {
            id: 41,
            role: 'select',
            text: 'All mediatypes Texts Collection Software Image Audio',
            name: 'mediatype',
            nearbyText: 'All mediatypes',
          },
          { id: 36, role: 'submit', text: 'Search', name: 'Search', clickable: true },
        ]),
      {
        onClick: (elementId, rt) => {
          if (elementId === 36) {
            const typedAnyField = rt.typeCalls.some(
              call => call.elementId === 13 && call.text === 'Space images'
            );
            rt.currentUrl = typedAnyField
              ? 'https://example.test/search?query=(Space%20images)%20AND%20mediatype:(image)'
              : 'https://example.test/search?query=mediatype:(image)';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 13, text: 'Space images' }]);
    expect(runtime.typeCalls).not.toContainEqual({ elementId: 8, text: 'Space images' });
    expect(result.stepOutcomes[0].urlAfter).toContain('Space%20images');
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'FILL_FORM(TYPE(13, "Space images") -> SELECT(41, image) -> CLICK(36))'
    );
  });

  it('escalates to find an explicit submit control after applying search filters', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [
          { label: 'any_field', value: 'Space images' },
          { label: 'All mediatypes', value: 'image' },
        ],
        submitText: 'Search',
        verify: [{ predicate: 'url_contains', args: ['mediatype=image'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'filtered search submitted once' }),
    ]);
    const executor = new ProviderStub();
    let snapshotCalls = 0;
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt => {
        snapshotCalls += 1;
        const baseElements: Snapshot['elements'] = [
          { id: 8, role: 'searchbox', text: 'any field', name: 'q', clickable: true },
          {
            id: 14,
            role: 'select',
            text: 'All mediatypes Texts Collection Software Image Audio',
            name: 'mediatype',
            nearbyText: 'All mediatypes',
          },
        ];
        return makeSnapshot(
          rt.currentUrl,
          snapshotCalls === 1
            ? baseElements
            : [...baseElements, { id: 17, role: 'button', text: 'Search', clickable: true }]
        );
      },
      {
        onClick: (elementId, rt) => {
          if (elementId === 17) {
            rt.currentUrl = 'https://example.test/search?q=Space+images&mediatype=image';
          }
        },
        onPressKey: (_key, rt) => {
          rt.currentUrl = 'https://example.test/search?q=Space+images';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.selectCalls).toEqual([{ elementId: 14, value: 'image' }]);
    expect(runtime.clickCalls).toEqual([17]);
    expect(runtime.keyCalls).toEqual([]);
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'FILL_FORM(TYPE(8, "Space images") -> SELECT(14, image) -> CLICK(17))'
    );
  });

  it('treats native submit inputs as explicit submit controls after applying search filters', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [
          { label: 'any_field', value: 'Space images' },
          { label: 'All mediatypes', value: 'image' },
        ],
        submitText: 'Search',
        verify: [{ predicate: 'url_contains', args: ['mediatype=image'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'filtered search submitted once' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 8, role: 'searchbox', text: 'any field', name: 'q', clickable: true },
          {
            id: 14,
            role: 'select',
            text: 'All mediatypes Texts Collection Software Image Audio',
            name: 'mediatype',
            nearbyText: 'All mediatypes',
          },
          { id: 17, role: 'submit', text: 'Search', name: 'Search', clickable: false },
        ]),
      {
        onClick: (elementId, rt) => {
          if (elementId === 17) {
            rt.currentUrl = 'https://example.test/search?q=Space+images&mediatype=image';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([17]);
    expect(runtime.keyCalls).toEqual([]);
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'FILL_FORM(TYPE(8, "Space images") -> SELECT(14, image) -> CLICK(17))'
    );
  });

  it('accepts FILL_FORM repair steps because stepwise execution supports them', () => {
    const normalized = normalizeReplanPatch({
      mode: 'patch',
      replace_steps: [
        {
          id: 2,
          step: {
            id: 2,
            action: 'FILL_FORM',
            fields: [{ label: 'q', value: 'Space images' }],
            submitText: 'Search',
            verify: [],
          },
        },
      ],
    });

    expect(() => ReplanPatchSchema.parse(normalized)).not.toThrow();
  });

  it('does not submit mixed FILL_FORM search plans when filter fields are unmatched', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [
          { label: 'q', value: 'Space images' },
          { label: 'mediatype', value: 'image' },
        ],
        submitText: 'Search',
        verify: [{ predicate: 'url_contains', args: ['Space'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted once' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub('https://example.test/advancedsearch', rt =>
      makeSnapshot(rt.currentUrl, [
        {
          id: 5,
          role: 'textbox',
          text: 'optional field 3',
          name: 'optional_field3_q',
          clickable: true,
        },
        { id: 8, role: 'searchbox', text: 'Search', name: 'q', clickable: true },
        { id: 9, role: 'button', text: 'Search', clickable: true },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(false);
    expect(runtime.typeCalls).toEqual([{ elementId: 8, text: 'Space images' }]);
    expect(runtime.typeCalls).not.toContainEqual({ elementId: 5, text: 'Space images' });
    expect(runtime.clickCalls).toEqual([]);
    expect(runtime.keyCalls).toEqual([]);
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'FILL_FORM(TYPE(8, "Space images") -> UNMATCHED(mediatype))'
    );
  });

  it('routes single-value FILL_FORM search queries to the primary search input, not AND fields', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [{ label: 'optional field 3', value: 'Space images' }],
        verify: [{ predicate: 'url_contains', args: ['Space'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted once' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 1, role: 'searchbox', text: 'Search', name: 'q', clickable: true },
          {
            id: 2,
            role: 'textbox',
            text: 'optional field 3',
            name: 'optional_field3_q',
            clickable: true,
          },
          {
            id: 3,
            role: 'textbox',
            text: 'optional field 4',
            name: 'optional_field4_q',
            clickable: true,
          },
        ]),
      {
        onPressKey: (_key, rt) => {
          rt.currentUrl = 'https://example.test/search?q=Space+images';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 1, text: 'Space images' }]);
    expect(runtime.typeCalls).not.toContainEqual({ elementId: 2, text: 'Space images' });
    expect(runtime.keyCalls).toEqual(['Enter']);
    expect(result.stepOutcomes[0].actionTaken).toBe('FILL_FORM(TYPE(1, "Space images") -> ENTER)');
  });

  it('escalates FILL_FORM search routing instead of typing into visible AND fields', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [{ label: 'optional field 3', value: 'Space images' }],
        verify: [{ predicate: 'url_contains', args: ['Space'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted once' }),
    ]);
    const executor = new ProviderStub();
    let snapshotCalls = 0;
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt => {
        snapshotCalls += 1;
        if (snapshotCalls === 1) {
          return makeSnapshot(rt.currentUrl, [
            {
              id: 5,
              role: 'textbox',
              text: 'optional field 3',
              name: 'optional_field3_q',
              clickable: true,
            },
          ]);
        }
        return makeSnapshot(rt.currentUrl, [
          {
            id: 5,
            role: 'textbox',
            text: 'optional field 3',
            name: 'optional_field3_q',
            clickable: true,
          },
          { id: 8, role: 'searchbox', text: 'Search', name: 'q', clickable: true },
        ]);
      },
      {
        onPressKey: (_key, rt) => {
          rt.currentUrl = 'https://example.test/search?q=Space+images';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 8, text: 'Space images' }]);
    expect(runtime.typeCalls).not.toContainEqual({ elementId: 5, text: 'Space images' });
    expect(runtime.keyCalls).toEqual(['Enter']);
  });

  it('treats duplicate-value FILL_FORM on search pages as a single primary search submission', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'FILL_FORM',
        fields: [
          { label: 'optional field 3', value: 'Space images' },
          { label: 'optional field 4', value: 'Space images' },
        ],
        verify: [{ predicate: 'url_contains', args: ['Space'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted once' }),
    ]);
    const executor = new ProviderStub();
    const runtime = new RuntimeStub(
      'https://example.test/advancedsearch',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 1, role: 'searchbox', text: 'Search', name: 'q', clickable: true },
          {
            id: 2,
            role: 'textbox',
            text: 'optional field 3',
            name: 'optional_field3_q',
            clickable: true,
          },
          {
            id: 3,
            role: 'textbox',
            text: 'optional field 4',
            name: 'optional_field4_q',
            clickable: true,
          },
        ]),
      {
        onPressKey: (_key, rt) => {
          rt.currentUrl = 'https://example.test/search?q=Space+images';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 1, text: 'Space images' }]);
    expect(runtime.keyCalls).toEqual(['Enter']);
    expect(result.stepOutcomes[0].actionTaken).toBe('FILL_FORM(TYPE(1, "Space images") -> ENTER)');
  });

  it('falls back to snapshot content when markdown extraction is unavailable', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'EXTRACT',
        goal: 'capture dates and titles of the first 10 images',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub(['2020-01-01T00:00:00Z - Space photo']);
    const runtime = new NullMarkdownRuntimeStub('https://example.test/search', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 1, role: 'article', text: '2020-01-01T00:00:00Z - Space photo' },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[0].extractedData).toEqual({
      text: '2020-01-01T00:00:00Z - Space photo',
      query: 'capture dates and titles of the first 10 images',
    });
  });

  it('falls back to snapshot content when markdown extraction returns not found', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'EXTRACT',
        goal: 'capture dates and titles of the first 10 images',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub([
      'NOT_FOUND',
      '2024-01-01T00:00:00Z - Space image one\n2024-01-02T00:00:00Z - Space image two',
    ]);
    const runtime = new MarkdownRuntimeStub(
      'https://example.test/search?query=(Space%20images)%20AND%20mediatype:(image)',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 1, role: 'link', text: 'Space image one' },
          { id: 2, role: 'text', text: '2024-01-01T00:00:00Z' },
          { id: 3, role: 'link', text: 'Space image two' },
          { id: 4, role: 'text', text: '2024-01-02T00:00:00Z' },
        ]),
      'Search results page'
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(executor.calls).toHaveLength(2);
    expect(executor.calls[1]?.user).toContain('Space image one');
    expect(result.stepOutcomes[0].extractedData).toEqual({
      text: '2024-01-01T00:00:00Z - Space image one\n2024-01-02T00:00:00Z - Space image two',
      query: 'capture dates and titles of the first 10 images',
    });
  });

  it('uses vision extraction when text context is unavailable but a screenshot is present', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'EXTRACT',
        goal: 'capture dates and titles of the first 10 images',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub(['2024-01-01T00:00:00Z - Space image one'], {
      vision: true,
    });
    const runtime = new NullMarkdownRuntimeStub(
      'https://example.test/search?query=(Space%20images)%20AND%20mediatype:(image)',
      rt => makeSnapshot(rt.currentUrl, [], { screenshot: 'base64-screenshot' })
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(executor.imageCalls).toHaveLength(1);
    expect(executor.imageCalls[0]?.imageBase64).toBe('base64-screenshot');
    expect(result.stepOutcomes[0].usedVision).toBe(true);
    expect(result.stepOutcomes[0].extractedData).toEqual({
      text: '2024-01-01T00:00:00Z - Space image one',
      query: 'capture dates and titles of the first 10 images',
    });
  });

  it('treats page-chrome-only markdown as unavailable and uses vision extraction', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'EXTRACT',
        goal: 'capture dates and titles of the first 10 images',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub(['2024-01-01T00:00:00Z - Space image one'], {
      vision: true,
    });
    const runtime = new MarkdownRuntimeStub(
      'https://example.test/search?query=(Space%20images)%20AND%20mediatype:(image)',
      rt => makeSnapshot(rt.currentUrl, [], { screenshot: 'base64-screenshot' }),
      'Skip to main content (https://example.test/#maincontent)'
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(executor.calls).toHaveLength(0);
    expect(executor.imageCalls).toHaveLength(1);
    expect(result.stepOutcomes[0].usedVision).toBe(true);
    expect(result.stepOutcomes[0].extractedData).toEqual({
      text: '2024-01-01T00:00:00Z - Space image one',
      query: 'capture dates and titles of the first 10 images',
    });
  });

  it('coerces SCROLL_AND_COUNT into EXTRACT for first-N extraction tasks without scrolling', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'SCROLL_AND_COUNT',
        countTarget: 'image listings',
        goal: 'Count image listings',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub(['2024-01-01T00:00:00Z - Space image one']);
    const runtime = new RuntimeStub(
      'https://example.test/search?query=(Space%20images)%20AND%20mediatype:(image)',
      rt => makeSnapshot(rt.currentUrl, [{ id: 1, role: 'link', text: 'Space image one' }])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].actionTaken).toBe('EXTRACT');
    expect(result.stepOutcomes[0].extractedData).toEqual({
      text: '2024-01-01T00:00:00Z - Space image one',
      query:
        'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });
    expect(await runtime.getCurrentUrl()).toBe(
      'https://example.test/search?query=(Space%20images)%20AND%20mediatype:(image)'
    );
  });

  it('does not rewrite site URLs during extraction', async () => {
    const currentUrl =
      'https://archive.org/advancedsearch.php?q=Space+images&fl%5B%5D=identifier&rows=50&page=1&output=json&callback=callback&save=yes';
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'EXTRACT',
        goal: 'capture dates and titles of the first 10 images',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub(['2020-01-01T00:00:00Z - Space photo']);
    const runtime = new MarkdownRuntimeStub(
      currentUrl,
      rt => makeSnapshot(rt.currentUrl, []),
      'callback({"response":{"docs":[{"date":"2020-01-01T00:00:00Z","title":"Space photo","mediatype":"image"}]}})'
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images on archive.org and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes).toHaveLength(1);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(runtime.gotoCalls).toHaveLength(0);
    expect(result.stepOutcomes[0].urlAfter).toBe(currentUrl);
  });

  it('preserves the full task constraints when planner gives a shortened extract goal', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'EXTRACT',
        goal: 'capture dates and titles',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub(['2020-01-01T00:00:00Z - Space photo']);
    const runtime = new MarkdownRuntimeStub(
      'https://example.test/search',
      rt => makeSnapshot(rt.currentUrl, []),
      '{"response":{"docs":[{"date":"2020-01-01T00:00:00Z","title":"Space photo"}]}}'
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    expect(result.success).toBe(true);
    expect(executor.calls[0]?.user).toContain('first 10 images listed');
    expect(executor.calls[0]?.user).toContain('If the request asks for first/top/last N items');
    expect(result.stepOutcomes[0].extractedData).toEqual({
      text: '2020-01-01T00:00:00Z - Space photo',
      query:
        'capture dates and titles\nOverall task: Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });
  });

  it('enforces first N extraction limits when the extractor returns extra rows', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'EXTRACT',
        goal: 'capture dates and titles',
        verify: [],
      }),
    ]);
    const executor = new ProviderStub([
      [
        '1994-04-01T00:00:00Z - Space Radar Image',
        '1957-05-21T00:00:00Z - Moon Rocket Cantata',
        '2022-12-02T00:00:00Z - Deltadel Ebro',
        '2000-12-16T00:00:00Z - Ganymede and Europa',
        '2001-12-09T00:00:00Z - Praxidike',
        '2006-01-01T00:00:00Z - photo-jsc2006e15538',
        '2008-11-13T00:00:00Z - Young Scientist Challenge',
        '2003-04-03T00:00:00Z - Dusty Star',
        '1967-05-08T00:00:00Z - Recruiting Brochure',
        '1967-06-22T00:00:00Z - Breadboard Checkout',
        '2003-01-01T00:00:00Z - extra row',
      ].join('\n'),
    ]);
    const runtime = new MarkdownRuntimeStub(
      'https://example.test/search',
      rt => makeSnapshot(rt.currentUrl, []),
      '{"response":{"docs":[]}}'
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Use advanced search for Space images and output the capture dates and titles of the first 10 images listed.',
    });

    const text = (result.stepOutcomes[0].extractedData as { text: string }).text;
    expect(result.success).toBe(true);
    expect(text.split('\n')).toHaveLength(10);
    expect(text).not.toContain('extra row');
  });

  it('identifies search-like TYPE_AND_SUBMIT actions and rejects unrelated URL changes', () => {
    const searchbox: SnapshotElement = {
      id: 1,
      role: 'searchbox',
      ariaLabel: 'Search products',
      clickable: true,
    };

    expect(
      isSearchLikeTypeAndSubmit(
        { action: 'TYPE_AND_SUBMIT', intent: 'search for trail shoes', input: 'trail shoes' },
        searchbox
      )
    ).toBe(true);

    expect(
      isSearchLikeTypeAndSubmit(
        { action: 'TYPE_AND_SUBMIT', intent: 'enter email address', input: 'user@example.com' },
        { role: 'textbox', ariaLabel: 'Email address' }
      )
    ).toBe(false);

    expect(
      isUrlChangeRelevantToIntent('https://shop.test/', 'https://shop.test/promo-overlay', {
        action: 'TYPE_AND_SUBMIT',
        intent: 'search for trail shoes',
        input: 'trail shoes',
        verify: [{ predicate: 'url_contains', args: ['/search'] }],
      })
    ).toBe(false);

    expect(
      isUrlChangeRelevantToIntent('https://shop.test/', 'https://shop.test/search?q=trail+shoes', {
        action: 'TYPE_AND_SUBMIT',
        intent: 'search for trail shoes',
        input: 'trail shoes',
        verify: [{ predicate: 'url_contains', args: ['/search'] }],
      })
    ).toBe(true);

    expect(
      isUrlChangeRelevantToIntent(
        'https://www.amazon.com/s?k=noise+canceling+earbuds',
        'https://www.amazon.com/ref=nav_logo_prime',
        {
          action: 'CLICK',
          intent: 'product link',
          input: 'noise canceling earbuds',
          verify: [],
        }
      )
    ).toBe(false);

    expect(
      isUrlChangeRelevantToIntent(
        'https://www.amazon.com/s?k=noise+canceling+earbuds',
        'https://www.amazon.com/dp/B012345678',
        {
          action: 'CLICK',
          intent: 'product link',
          input: 'noise canceling earbuds',
          verify: [],
        }
      )
    ).toBe(true);
  });

  it('prefers Enter for search inputs and retries with the explicit submit control when the first URL change is unrelated', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE_AND_SUBMIT',
        intent: 'search for trail shoes',
        input: 'trail shoes',
        verify: [{ predicate: 'url_contains', args: ['/search'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted' }),
    ]);
    const executor = new ProviderStub(['TYPE(1, "trail shoes")']);
    const runtime = new RuntimeStub(
      'https://shop.test/',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 1, role: 'searchbox', ariaLabel: 'Search', clickable: true, importance: 100 },
          { id: 2, role: 'button', text: 'Search', clickable: true, importance: 90 },
          { id: 3, role: 'button', text: 'Advanced Search', clickable: true, importance: 70 },
        ]),
      {
        onPressKey: () => {
          runtime.currentUrl = 'https://shop.test/promo-overlay';
        },
        onClick: elementId => {
          if (elementId === 2) {
            runtime.currentUrl = 'https://shop.test/search?q=trail+shoes';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, { task: 'Search for trail shoes' });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(runtime.typeCalls).toEqual([{ elementId: 1, text: 'trail shoes' }]);
    expect(runtime.keyCalls).toEqual(['Enter']);
    expect(runtime.clickCalls).toEqual([2]);
    expect(runtime.currentUrl).toContain('/search');
  });

  it('uses deterministic searchbox heuristics when the executor returns NONE for TYPE_AND_SUBMIT', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE_AND_SUBMIT',
        intent: 'searchbox',
        input: 'noise canceling earbuds',
        verify: [{ predicate: 'url_contains', args: ['/s?k='] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub(
      'https://www.amazon.com/',
      rt =>
        makeSnapshot(rt.currentUrl, [
          {
            id: 10,
            role: 'searchbox',
            name: 'Search Amazon',
            ariaLabel: 'Search Amazon',
            text: 'field-keywords',
            importance: 100,
          },
          { id: 11, role: 'button', text: 'Go', clickable: true, importance: 90 },
        ]),
      {
        onPressKey: () => {
          runtime.currentUrl = 'https://www.amazon.com/s?k=noise+canceling+earbuds';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Search for noise canceling earbuds',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 10, text: 'noise canceling earbuds' }]);
    expect(runtime.keyCalls).toEqual(['Enter']);
    expect(executor.calls).toHaveLength(0);
  });

  it('uses deterministic searchbox heuristics for planner TYPE actions and submits search-like inputs', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE',
        intent: 'searchbox',
        input: 'noise canceling earbuds',
        verify: [{ predicate: 'url_contains', args: ['/s?k='] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub(
      'https://www.amazon.com/',
      rt =>
        makeSnapshot(rt.currentUrl, [
          {
            id: 10,
            role: 'searchbox',
            name: 'Search Amazon',
            ariaLabel: 'Search Amazon',
            text: 'field-keywords',
            importance: 100,
          },
          { id: 11, role: 'button', text: 'Go', clickable: true, importance: 90 },
        ]),
      {
        onPressKey: () => {
          runtime.currentUrl = 'https://www.amazon.com/s?k=noise+canceling+earbuds';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Search for noise canceling earbuds',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 10, text: 'noise canceling earbuds' }]);
    expect(runtime.keyCalls).toEqual(['Enter']);
    expect(executor.calls).toHaveLength(0);
  });

  it('uses deterministic field heuristics on sparse multi-step form pages', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'email field completed' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub('https://forms.test/signup', rt =>
      makeSnapshot(rt.currentUrl, [
        {
          id: 20,
          role: 'textbox',
          ariaLabel: 'Email',
          text: 'Email',
          clickable: true,
          importance: 100,
        },
        { id: 21, role: 'button', text: 'Next', clickable: true, importance: 90 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Fill the multi-step signup form with user@example.com, then continue',
    });

    expect(result.success).toBe(true);
    expect(runtime.typeCalls).toEqual([{ elementId: 20, text: 'user@example.com' }]);
    expect(executor.calls).toHaveLength(0);
  });

  it('types a form field and clicks Next when verification expects the next step', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Display name'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'advanced to display name step' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    let stage: 'email' | 'displayName' = 'email';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'email'
            ? [
                {
                  id: 20,
                  role: 'textbox',
                  ariaLabel: 'Email',
                  text: 'Email',
                  clickable: true,
                  importance: 100,
                },
                { id: 21, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
            : [
                {
                  id: 30,
                  role: 'textbox',
                  ariaLabel: 'Display name',
                  text: 'Display name',
                  clickable: true,
                  importance: 100,
                },
              ],
          { status: 'require_vision' }
        ),
      {
        onClick: elementId => {
          if (elementId === 21) {
            stage = 'displayName';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Fill the multi-step signup form with user@example.com, then continue',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].actionTaken).toBe('TYPE(20, "user@example.com") -> CLICK(21)');
    expect(runtime.typeCalls).toEqual([{ elementId: 20, text: 'user@example.com' }]);
    expect(runtime.clickCalls).toEqual([21]);
    expect(executor.calls).toHaveLength(0);
  });

  it('clicks Next after vision coordinate typing when verification expects the next form step', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Display name'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'advanced to display name step' }),
    ]);
    const executor = new ProviderStub(['CLICK_XY(499, 337)'], { vision: true });
    let stage: 'email' | 'displayName' = 'email';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'email'
            ? [{ id: 21, role: 'button', text: 'Next', clickable: true, importance: 90 }]
            : [
                {
                  id: 30,
                  role: 'textbox',
                  ariaLabel: 'Display name',
                  text: 'Display name',
                  clickable: true,
                  importance: 100,
                },
              ],
          { status: 'require_vision', screenshot: 'ZmFrZQ==' }
        ),
      {
        onClick: elementId => {
          if (elementId === 21) {
            stage = 'displayName';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Fill the multi-step signup form with user@example.com, then continue',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'CLICK_XY(499, 337) + TYPE_AT("user@example.com") -> CLICK(21)'
    );
    expect(runtime.coordinateClickCalls).toEqual([{ x: 499, y: 337 }]);
    expect(runtime.coordinateTypeCalls).toEqual(['user@example.com']);
    expect(runtime.clickCalls).toEqual([21]);
    expect(executor.imageCalls).toHaveLength(1);
  });

  it('treats a same-url wizard pane transition as success after vision typing and Next', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Display name'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'advanced to next wizard pane' }),
    ]);
    const executor = new ProviderStub(['CLICK_XY(499, 337)'], { vision: true });
    let stage: 'email' | 'displayName' = 'email';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'email'
            ? [{ id: 21, role: 'button', text: 'Next', clickable: true, importance: 90 }]
            : [
                { id: 30, role: 'textbox', text: 'Llama Rider', clickable: true, importance: 100 },
                { id: 20, role: 'button', text: 'Back', clickable: true, importance: 80 },
                { id: 21, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ],
          { status: 'require_vision', screenshot: 'ZmFrZQ==' }
        ),
      {
        onClick: elementId => {
          if (elementId === 21) {
            stage = 'displayName';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Fill the multi-step signup form with user@example.com, then continue',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].actionTaken).toBe(
      'CLICK_XY(499, 337) + TYPE_AT("user@example.com") -> CLICK(21)'
    );
    expect(runtime.coordinateTypeCalls).toEqual(['user@example.com']);
    expect(runtime.clickCalls).toEqual([21]);
  });

  it('treats a same-url Next click as progress before final Submit', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Next button on step 1',
        verify: [{ predicate: 'element_exists', args: ['heading', 'Review'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'submitted onboarding form' }),
    ]);
    const executor = new ProviderStub(['CLICK(2)', 'CLICK(6)']);
    let stage: 'terms' | 'review' | 'submitted' = 'terms';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'terms'
            ? [
                {
                  id: 1,
                  role: 'checkbox',
                  text: 'Agree to terms',
                  clickable: true,
                  importance: 100,
                },
                { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
            : stage === 'review'
              ? [
                  { id: 5, role: 'button', text: 'Back', clickable: true, importance: 80 },
                  { id: 6, role: 'button', text: 'Submit', clickable: true, importance: 100 },
                ]
              : [{ id: 7, role: 'status', text: 'Submitted', importance: 100 }]
        ),
      {
        onClick: elementId => {
          if (elementId === 2) {
            stage = 'review';
          }
          if (elementId === 6) {
            stage = 'submitted';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Complete onboarding, agree to the terms, and lastly submit the multi-step form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(2)',
      verificationPassed: true,
    });
    expect(result.stepOutcomes[1]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(6)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([2, 6]);
  });

  it('does not pre-skip a same-url Next click just because the downstream Submit is visible', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Next button on step 2',
        verify: [{ predicate: 'element_exists', args: ['button', 'Submit'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'form submitted' }),
    ]);
    const executor = new ProviderStub(['CLICK(2)', 'CLICK(4)']);
    let stage: 'plan' | 'review' | 'submitted' = 'plan';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'plan'
            ? [
                { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
                { id: 3, role: 'checkbox', text: 'Terms', clickable: true, importance: 80 },
                { id: 4, role: 'button', text: 'Submit', clickable: true, importance: 40 },
              ]
            : stage === 'review'
              ? [
                  { id: 4, role: 'button', text: 'Submit', clickable: true, importance: 100 },
                  { id: 5, role: 'button', text: 'Back', clickable: true, importance: 80 },
                ]
              : [{ id: 6, role: 'status', text: 'Submitted', importance: 100 }]
        ),
      {
        onClick: elementId => {
          if (elementId === 2) {
            stage = 'review';
          } else if (elementId === 4) {
            stage = 'submitted';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Complete a same-url onboarding wizard and lastly submit the form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(2)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([2, 4]);
  });

  it('does not treat a Next click as successful final submission', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
    ]);
    const executor = new ProviderStub(['CLICK(2)']);
    let stage: 'terms' | 'review' = 'terms';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'terms'
            ? [
                { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
                { id: 3, role: 'button', text: 'Back', clickable: true, importance: 80 },
              ]
            : [
                { id: 4, role: 'button', text: 'Back', clickable: true, importance: 80 },
                { id: 6, role: 'button', text: 'Submit', clickable: true, importance: 100 },
              ]
        ),
      {
        onClick: elementId => {
          if (elementId === 2) {
            stage = 'review';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Lastly, submit the multi-step form',
    });

    expect(result.success).toBe(false);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.FAILED,
      actionTaken: 'CLICK(2)',
      verificationPassed: false,
    });
    expect(runtime.clickCalls).toEqual([2]);
    expect(stage).toBe('review');
  });

  it('treats a real Submit click as terminal when confirmation text differs from planner predicate', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'form submitted' }),
    ]);
    const executor = new ProviderStub(['CLICK(6)']);
    let stage: 'review' | 'complete' = 'review';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'review'
            ? [
                { id: 5, role: 'button', text: 'Back', clickable: true, importance: 80 },
                { id: 6, role: 'button', text: 'Submit', clickable: true, importance: 100 },
              ]
            : [
                {
                  id: 8,
                  role: 'heading',
                  text: 'Thanks for completing onboarding',
                  importance: 100,
                },
                { id: 9, role: 'text', text: 'Your Pro plan is ready.', importance: 80 },
              ]
        ),
      {
        onClick: elementId => {
          if (elementId === 6) {
            stage = 'complete';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Lastly, submit the multi-step form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(6)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([6]);
  });

  it('treats a same-pane Submit click as terminal when a confirmation status appears', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'form submitted' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    let submitted = false;
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          submitted
            ? [
                { id: 1, role: 'button', text: 'Submit', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
                { id: 3, role: 'status', text: 'Onboarding complete', importance: 100 },
              ]
            : [
                { id: 1, role: 'button', text: 'Submit', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
              ]
        ),
      {
        onClick: elementId => {
          if (elementId === 1) {
            submitted = true;
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Lastly, submit the multi-step form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(1)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([1]);
  });

  it('treats a Submit click as terminal when Submit disappears and only Back remains', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'form submitted' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    let submitted = false;
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          submitted
            ? [
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
                { id: 3, role: 'heading', text: 'Welcome aboard', importance: 100 },
              ]
            : [
                { id: 1, role: 'button', text: 'Submit', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
              ]
        ),
      {
        onClick: elementId => {
          if (elementId === 1) {
            submitted = true;
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Lastly, submit the multi-step form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(1)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([1]);
  });

  it('treats a Submit click as terminal when the submit control changes to Done', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'form submitted' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    let submitted = false;
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          submitted
            ? [
                { id: 1, role: 'button', text: 'Done', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
              ]
            : [
                { id: 1, role: 'button', text: 'Submit', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
              ]
        ),
      {
        onClick: elementId => {
          if (elementId === 1) {
            submitted = true;
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Lastly, submit the multi-step form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(1)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([1]);
  });

  it('does not require clicking Done after a terminal Submit succeeds', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Submit button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'Done button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Done'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'form submitted' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)', 'CLICK(1)']);
    let submitted = false;
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          submitted
            ? [
                { id: 1, role: 'button', text: 'Done', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
              ]
            : [
                { id: 1, role: 'button', text: 'Submit', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Back', clickable: true, importance: 80 },
              ]
        ),
      {
        onClick: elementId => {
          if (elementId === 1) {
            submitted = true;
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Lastly, submit the multi-step form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes).toHaveLength(1);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(1)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([1]);
    expect(planner.calls).toHaveLength(1);
  });

  it('treats other common final form buttons as terminal submissions', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Confirm registration button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Registered'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'registration completed' }),
    ]);
    const executor = new ProviderStub(['CLICK(10)']);
    let stage: 'review' | 'complete' = 'review';
    const runtime = new RuntimeStub(
      'https://forms.test/register',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'review'
            ? [
                { id: 9, role: 'button', text: 'Back', clickable: true, importance: 80 },
                {
                  id: 10,
                  role: 'button',
                  text: 'Confirm registration',
                  clickable: true,
                  importance: 100,
                },
              ]
            : [{ id: 11, role: 'heading', text: 'Account complete', importance: 100 }]
        ),
      {
        onClick: elementId => {
          if (elementId === 10) {
            stage = 'complete';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Confirm registration and complete the form',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(10)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([10]);
  });

  it('does not target Next with strict submit intent heuristics', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'submit',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub('https://forms.test/signup', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
        { id: 3, role: 'button', text: 'Back', clickable: true, importance: 80 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Lastly, submit the multi-step form',
    });

    expect(result.success).toBe(false);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.FAILED,
      verificationPassed: false,
      error: 'Executor could not find suitable element',
    });
    expect(runtime.clickCalls).toEqual([]);
  });

  it('does not reload a same-url wizard checkpoint during recovery', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'Next button on email step',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Display name'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'stale email field',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Email'] }],
      }),
    ]);
    const executor = new ProviderStub(['CLICK(2)', 'NONE']);
    let stage: 'email' | 'displayName' = 'email';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'email'
            ? [
                { id: 1, role: 'textbox', text: 'Email', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
            : [
                { id: 3, role: 'textbox', text: 'Display name', clickable: true, importance: 100 },
                { id: 4, role: 'button', text: 'Back', clickable: true, importance: 80 },
                { id: 5, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
        ),
      {
        onClick: elementId => {
          if (elementId === 2) {
            stage = 'displayName';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Complete a same-url onboarding wizard',
    });

    expect(result.stepOutcomes.length).toBeGreaterThanOrEqual(2);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[1].status).toBe(StepStatus.FAILED);
    expect(runtime.gotoCalls).toEqual([]);
    expect(stage).toBe('displayName');
  });

  it('skips repeated completed form-field intents on later wizard panes', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Display name'] }],
      }),
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Email'] }],
      }),
      JSON.stringify({
        action: 'TYPE',
        intent: 'display name field',
        input: 'Tony W',
        verify: [{ predicate: 'element_exists', args: ['button', 'Next'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'continued past stale email repeat' }),
    ]);
    const executor = new ProviderStub(['TYPE(1, "user@example.com")', 'TYPE(3, "Tony W")']);
    let stage: 'email' | 'displayName' = 'email';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'email'
            ? [
                { id: 1, role: 'textbox', text: 'Email', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
            : [
                { id: 3, role: 'textbox', text: 'Display name', clickable: true, importance: 100 },
                { id: 4, role: 'button', text: 'Back', clickable: true, importance: 80 },
                { id: 5, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
        ),
      {
        onType: (elementId, _text) => {
          if (elementId === 1) {
            stage = 'displayName';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Complete a same-url onboarding wizard',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[1]).toMatchObject({
      status: StepStatus.SKIPPED,
      actionTaken: 'SKIPPED(previously_completed_form_step)',
      verificationPassed: true,
    });
    expect(runtime.typeCalls).toEqual([
      { elementId: 1, text: 'user@example.com' },
      { elementId: 3, text: 'Tony W' },
    ]);
  });

  it('records previously completed form skips as skipped so the planner can move on', async () => {
    const planner = new AdaptiveProviderStub();
    const executor = new ProviderStub(['TYPE(1, "user@example.com")']);
    let stage: 'email' | 'displayName' = 'email';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'email'
            ? [
                { id: 1, role: 'textbox', text: 'Email', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
            : [
                { id: 3, role: 'textbox', text: 'Display name', clickable: true, importance: 100 },
                { id: 4, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
        ),
      {
        onType: (elementId, _text) => {
          if (elementId === 1) {
            stage = 'displayName';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
        stepwise: { maxSteps: 4 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Complete a same-url onboarding wizard',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[1]).toMatchObject({
      status: StepStatus.SKIPPED,
      actionTaken: 'SKIPPED(previously_completed_form_step)',
    });
    expect(planner.calls[2]?.user).toContain('TYPE(user@example.com) → skipped');
    expect(result.error).toBeUndefined();
  });

  it('stops repeated completed form skip loops before exhausting max steps', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Display name'] }],
      }),
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Email'] }],
      }),
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Email'] }],
      }),
      JSON.stringify({
        action: 'TYPE',
        intent: 'email field',
        input: 'user@example.com',
        verify: [{ predicate: 'element_exists', args: ['textbox', 'Email'] }],
      }),
    ]);
    const executor = new ProviderStub(['TYPE(1, "user@example.com")']);
    let stage: 'email' | 'displayName' = 'email';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'email'
            ? [
                { id: 1, role: 'textbox', text: 'Email', clickable: true, importance: 100 },
                { id: 2, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
            : [
                { id: 3, role: 'textbox', text: 'Display name', clickable: true, importance: 100 },
                { id: 4, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
        ),
      {
        onType: (elementId, _text) => {
          if (elementId === 1) {
            stage = 'displayName';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
        stepwise: { maxSteps: 10 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Complete a same-url onboarding wizard',
    });

    expect(result.success).toBe(false);
    expect(result.error).toContain('Planner repeated already completed form step');
    expect(result.error).not.toContain('Exceeded maximum steps');
    expect(result.stepOutcomes).toHaveLength(4);
  });

  it('skips narrower repeated plan-choice intents after the plan step succeeds', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'plan radio button',
        verify: [{ predicate: 'element_exists', args: ['checkbox', 'Terms'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'Pro plan radio button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Plan confirmed'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'terms checkbox',
        verify: [{ predicate: 'element_exists', args: ['status', 'Submitted'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'plan selected and terms accepted' }),
    ]);
    const executor = new ProviderStub(['CLICK(4)', 'CLICK(5)']);
    let stage: 'plan' | 'terms' | 'submitted' = 'plan';
    const runtime = new RuntimeStub(
      'https://forms.test/signup',
      rt =>
        makeSnapshot(
          rt.currentUrl,
          stage === 'plan'
            ? [
                { id: 4, role: 'radio', text: 'Pro', clickable: true, importance: 100 },
                { id: 6, role: 'button', text: 'Next', clickable: true, importance: 90 },
              ]
            : stage === 'terms'
              ? [
                  { id: 5, role: 'checkbox', text: 'Terms', clickable: true, importance: 100 },
                  { id: 7, role: 'button', text: 'Submit', clickable: true, importance: 90 },
                ]
              : [{ id: 8, role: 'status', text: 'Submitted', importance: 100 }]
        ),
      {
        onClick: elementId => {
          if (elementId === 4) {
            stage = 'terms';
          }
          if (elementId === 5) {
            stage = 'submitted';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Complete onboarding with the Pro plan and accept terms',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[1]).toMatchObject({
      status: StepStatus.SKIPPED,
      actionTaken: 'SKIPPED(previously_completed_form_step)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([4, 5]);
  });

  it('does not pre-skip a form choice just because its next-step verification is already visible', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'pro plan radio button',
        verify: [{ predicate: 'element_exists', args: ['checkbox', 'Terms'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'plan selected' }),
    ]);
    const executor = new ProviderStub(['CLICK(4)']);
    const runtime = new RuntimeStub('https://forms.test/signup', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 4, role: 'radio', text: 'Pro', clickable: true, importance: 100 },
        { id: 5, role: 'radio', text: 'Basic', clickable: true, importance: 80 },
        { id: 6, role: 'checkbox', text: 'Terms', clickable: true, importance: 90 },
        { id: 7, role: 'button', text: 'Next', clickable: true, importance: 70 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Choose the Pro plan',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(4)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([4]);
  });

  it('treats clicking the intended radio option as success when verification is too strict', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'pro plan radio button',
        verify: [{ predicate: 'element_exists', args: ['status', 'Plan selected'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'plan selected' }),
    ]);
    const executor = new ProviderStub(['CLICK(4)']);
    const runtime = new RuntimeStub('https://forms.test/signup', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 4, role: 'radio', text: 'Pro', clickable: true, importance: 100 },
        { id: 5, role: 'radio', text: 'Basic', clickable: true, importance: 90 },
        { id: 6, role: 'button', text: 'Next', clickable: true, importance: 80 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Choose the Pro plan',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0]).toMatchObject({
      status: StepStatus.SUCCESS,
      actionTaken: 'CLICK(4)',
      verificationPassed: true,
    });
    expect(runtime.clickCalls).toEqual([4]);
  });

  it('normalizes repair optional substep aliases and numeric inputs', () => {
    expect(
      normalizeReplanPatch({
        replaceSteps: [
          {
            id: '1',
            step: {
              action: 'CLICK',
              intent: 'repair plan step',
              optionalSubsteps: [
                { action: 'TYPE', intent: 'retry field', input: 4 },
                { action: 'SCROLL_TO', intent: 'scroll to submit' },
                { action: 'SCROLL_INTO_VIEW', intent: 'scroll submit into view' },
              ],
            },
          },
        ],
      })
    ).toMatchObject({
      replaceSteps: [
        {
          id: 1,
          step: {
            optionalSubsteps: [
              { action: 'TYPE', input: '4' },
              { action: 'SCROLL' },
              { action: 'SCROLL' },
            ],
          },
        },
      ],
    });
  });

  it('treats a relevant search URL change as success even when planner verification is too strict', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE_AND_SUBMIT',
        intent: 'searchbox',
        input: 'noise canceling earbuds',
        verify: [{ predicate: 'url_contains', args: ['/search'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search submitted' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub(
      'https://www.amazon.com/',
      rt =>
        makeSnapshot(rt.currentUrl, [
          {
            id: 10,
            role: 'searchbox',
            name: 'Search Amazon',
            ariaLabel: 'Search Amazon',
            text: 'field-keywords',
            importance: 100,
          },
          { id: 11, role: 'button', text: 'Go', clickable: true, importance: 90 },
        ]),
      {
        onPressKey: () => {
          runtime.currentUrl = 'https://www.amazon.com/s?k=noise+canceling+earbuds&ref=nb_sb_noss';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Search for noise canceling earbuds, then pick a product',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[0].verificationPassed).toBe(true);
    expect(runtime.currentUrl).toContain('/s?k=noise+canceling+earbuds');
  });

  it('does not retry submission when Enter satisfies verification without changing the URL', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE_AND_SUBMIT',
        intent: 'search for trail shoes',
        input: 'trail shoes',
        verify: [{ predicate: 'exists', args: ['Result item'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search results are visible' }),
    ]);
    const executor = new ProviderStub(['TYPE(1, "trail shoes")']);
    let submitted = false;
    const runtime = new RuntimeStub(
      'https://shop.test/search',
      () =>
        makeSnapshot('https://shop.test/search', [
          { id: 1, role: 'searchbox', ariaLabel: 'Search', clickable: true, importance: 100 },
          { id: 2, role: 'button', text: 'Search', clickable: true, importance: 90 },
          ...(submitted
            ? [{ id: 3, role: 'text', text: 'Result item', importance: 80 } as SnapshotElement]
            : []),
        ]),
      {
        onPressKey: () => {
          submitted = true;
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, { task: 'Search for trail shoes' });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(runtime.keyCalls).toEqual(['Enter']);
    expect(runtime.clickCalls).toEqual([]);
  });

  it('still consults the executor for non-search type-and-submit actions on multi-input pages', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'TYPE_AND_SUBMIT',
        intent: 'enter email address',
        input: 'user@example.com',
        verify: [{ predicate: 'exists', args: ['Signed in'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'submitted sign-in form' }),
    ]);
    const executor = new ProviderStub(['TYPE(2, "user@example.com")']);
    let signedIn = false;
    const runtime = new RuntimeStub(
      'https://shop.test/account',
      () =>
        makeSnapshot('https://shop.test/account', [
          { id: 1, role: 'searchbox', ariaLabel: 'Search', clickable: true, importance: 100 },
          { id: 2, role: 'textbox', ariaLabel: 'Email address', clickable: true, importance: 95 },
          { id: 3, role: 'button', text: 'Sign In', clickable: true, importance: 90 },
          ...(signedIn
            ? [{ id: 4, role: 'text', text: 'Signed in', importance: 80 } as SnapshotElement]
            : []),
        ]),
      {
        onClick: elementId => {
          if (elementId === 3) {
            signedIn = true;
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0, executorRepairAttempts: 1 },
        recovery: { enabled: false },
      },
    });

    const result = await agent.runStepwise(runtime, { task: 'Sign in with email' });

    expect(result.success).toBe(true);
    expect(executor.calls.length).toBeGreaterThan(0);
    expect(runtime.typeCalls).toEqual([{ elementId: 2, text: 'user@example.com' }]);
    expect(runtime.clickCalls).toEqual([3]);
  });
});
