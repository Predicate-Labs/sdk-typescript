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
import type { SnapshotElement } from '../../../src/agents/planner-executor/plan-models';

class ProviderStub extends LLMProvider {
  private responses: string[];
  public calls: Array<{ system?: string; user?: string; options?: any }> = [];

  constructor(responses: string[] = []) {
    super();
    this.responses = [...responses];
  }

  get modelName(): string {
    return 'stub';
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
}

class RuntimeStub implements AgentRuntime {
  public currentUrl: string;
  public clickCalls: number[] = [];
  public typeCalls: Array<{ elementId: number; text: string }> = [];
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
    this.currentUrl = url;
  }

  async click(elementId: number): Promise<void> {
    this.clickCalls.push(elementId);
    await this.handlers.onClick?.(elementId, this);
  }

  async type(elementId: number, text: string): Promise<void> {
    this.typeCalls.push({ elementId, text });
    await this.handlers.onType?.(elementId, text, this);
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

function makeSnapshot(url: string, elements: Snapshot['elements']): Snapshot {
  return {
    url,
    title: 'Test Page',
    elements,
  };
}

describe('PlannerExecutorAgent search submission parity', () => {
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
