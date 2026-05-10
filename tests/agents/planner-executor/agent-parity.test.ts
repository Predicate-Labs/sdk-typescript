import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  StepStatus,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';

class ProviderStub extends LLMProvider {
  private responses: string[];
  public calls: Array<{ system: string; user: string; options?: any }> = [];
  public imageCalls: Array<{ system: string; user: string; imageBase64: string; options?: any }> =
    [];
  private readonly vision: boolean;

  constructor(responses: string[] = [], options: { vision?: boolean } = {}) {
    super();
    this.responses = [...responses];
    this.vision = options.vision ?? false;
  }

  get modelName(): string {
    return this.vision ? 'vision-stub' : 'stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return this.vision;
  }

  async generate(
    systemPrompt: string,
    userPrompt: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    this.calls.push({ system: systemPrompt, user: userPrompt, options });
    const content = this.responses.length ? this.responses.shift()! : 'DONE';
    return {
      content,
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    };
  }

  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    imageBase64: string,
    options: Record<string, any> = {}
  ): Promise<LLMResponse> {
    this.imageCalls.push({ system: systemPrompt, user: userPrompt, imageBase64, options });
    const content = this.responses.length ? this.responses.shift()! : 'CLICK(1)';
    return {
      content,
      modelName: this.modelName,
      promptTokens: 12,
      completionTokens: 6,
      totalTokens: 18,
    };
  }
}

class ThrowingProvider extends LLMProvider {
  get modelName(): string {
    return 'throwing-stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return false;
  }

  async generate(): Promise<LLMResponse> {
    throw new Error('LLM request timed out after 120000ms');
  }
}

class TimeoutThenDoneProvider extends LLMProvider {
  private called = false;

  get modelName(): string {
    return 'timeout-then-done-stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return false;
  }

  async generate(): Promise<LLMResponse> {
    if (!this.called) {
      this.called = true;
      throw new Error('LLM request timed out after 120000ms');
    }
    return {
      content: JSON.stringify({ action: 'DONE', reasoning: 'fallback search submitted' }),
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
  public gotoCalls: string[] = [];
  public scrollCalls: Array<'up' | 'down'> = [];
  public snapshotCalls = 0;

  constructor(
    initialUrl: string,
    private readonly snapshotFactory: (runtime: RuntimeStub) => Snapshot | null,
    private readonly handlers: {
      onClick?: (elementId: number, runtime: RuntimeStub) => Promise<void> | void;
      onType?: (elementId: number, text: string, runtime: RuntimeStub) => Promise<void> | void;
      onGoto?: (url: string, runtime: RuntimeStub) => Promise<void> | void;
      onPressKey?: (key: string, runtime: RuntimeStub) => Promise<void> | void;
    } = {}
  ) {
    this.currentUrl = initialUrl;
  }

  async snapshot(): Promise<Snapshot | null> {
    this.snapshotCalls += 1;
    const snap = this.snapshotFactory(this);
    if (snap) {
      this.currentUrl = snap.url || this.currentUrl;
    }
    return snap;
  }

  async goto(url: string): Promise<void> {
    this.gotoCalls.push(url);
    this.currentUrl = url;
    await this.handlers.onGoto?.(url, this);
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

  async scroll(direction: 'up' | 'down'): Promise<void> {
    this.scrollCalls.push(direction);
  }

  async getCurrentUrl(): Promise<string> {
    return this.currentUrl;
  }

  async getViewportHeight(): Promise<number> {
    return 1000;
  }

  async scrollBy(_dy: number): Promise<boolean> {
    return true;
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

describe('PlannerExecutorAgent parity', () => {
  it('bootstraps TYPE_AND_SUBMIT on step 1 when planner times out and searchbox is visible', async () => {
    const planner = new TimeoutThenDoneProvider();
    const executor = new ProviderStub([]);
    const runtime = new RuntimeStub(
      'https://www.amazon.com/',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 1, role: 'searchbox', text: 'Search Amazon', clickable: true, importance: 100 },
        ]),
      {
        onPressKey: key => {
          if (key === 'Enter') {
            runtime.currentUrl = 'https://www.amazon.com/s?k=noise+canceling+earbuds';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        stepwise: { maxSteps: 12 },
      },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Search for noise canceling earbuds on Amazon',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[0].actionTaken).toBe('TYPE(1, "noise canceling earbuds")');
    expect(runtime.typeCalls).toEqual([{ elementId: 1, text: 'noise canceling earbuds' }]);
    expect(runtime.keyCalls).toContain('Enter');
    expect(result.stepOutcomes[0].verificationPassed).toBe(true);
  });

  it('aborts early when planner times out without searchable context', async () => {
    const planner = new ThrowingProvider();
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub('https://www.amazon.com/', () =>
      makeSnapshot('https://www.amazon.com/', [
        { id: 1, role: 'link', text: 'Home', clickable: true, importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        stepwise: { maxSteps: 12 },
      },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Search for noise canceling earbuds on Amazon',
    });

    expect(result.success).toBe(false);
    expect(result.stepsTotal).toBe(1);
    expect(result.stepOutcomes).toHaveLength(1);
    expect(result.stepOutcomes[0].goal).toBe('Call planner LLM');
    expect(result.error).toContain('Planner unavailable');
  });

  it('skips execution when pre-step verification already passes', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'checkout',
        input: 'Checkout',
        verify: [{ predicate: 'url_contains', args: ['/cart'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'already satisfied' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    const runtime = new RuntimeStub('https://shop.test/cart', () =>
      makeSnapshot('https://shop.test/cart', [
        { id: 1, role: 'button', text: 'Checkout', clickable: true, importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Go to checkout' });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SKIPPED);
    expect(result.stepOutcomes[0].actionTaken).toBe('SKIPPED(pre_verification_passed)');
    expect(runtime.clickCalls).toEqual([]);
    expect(executor.calls).toHaveLength(0);
  });

  it('marks a step failed when post-action verification predicates do not pass', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'checkout',
        input: 'Checkout',
        verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'stop' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    const runtime = new RuntimeStub('https://shop.test/cart', () =>
      makeSnapshot('https://shop.test/cart', [
        { id: 1, role: 'button', text: 'Checkout', clickable: true, importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1 } },
    });
    const result = await agent.runStepwise(runtime, { task: 'Go to checkout' });

    expect(result.stepOutcomes[0].status).toBe(StepStatus.FAILED);
    expect(result.stepOutcomes[0].verificationPassed).toBe(false);
  });

  it('treats a relevant click URL change as success when planner verification is too strict', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'learn more link',
        input: 'Learn more',
        verify: [{ predicate: 'url_contains', args: ['/learn-more'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'link opened' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    const runtime = new RuntimeStub(
      'https://example.com/',
      rt =>
        makeSnapshot(rt.currentUrl, [
          {
            id: 1,
            role: 'link',
            text: 'Learn more',
            href: 'https://www.iana.org/help/example-domains',
            clickable: true,
            importance: 100,
          },
        ]),
      {
        onClick: id => {
          if (id === 1) {
            runtime.currentUrl = 'https://www.iana.org/help/example-domains';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 } },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Find the Learn more link and click it',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[0].verificationPassed).toBe(true);
    expect(result.stepOutcomes[0].urlAfter).toBe('https://www.iana.org/help/example-domains');
  });

  it('retries product-link click with best navigation candidate when first click does not navigate', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'product link',
        input: 'noise canceling earbuds',
        verify: [],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'product opened' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    const runtime = new RuntimeStub(
      'https://shop.test/search?q=noise+canceling+earbuds',
      rt =>
        makeSnapshot(rt.currentUrl, [
          {
            id: 1,
            role: 'link',
            text: 'Home',
            href: 'https://shop.test/home',
            clickable: true,
            importance: 100,
          },
          {
            id: 2,
            role: 'link',
            text: 'Noise Canceling Earbuds Product',
            href: 'https://shop.test/products/noise-canceling-earbuds-123',
            clickable: true,
            importance: 90,
          },
        ]),
      {
        onClick: id => {
          if (id === 1) {
            runtime.currentUrl = 'https://shop.test/search?q=noise+canceling+earbuds';
          }
          if (id === 2) {
            runtime.currentUrl = 'https://shop.test/products/noise-canceling-earbuds-123';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 } },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Search for noise canceling earbuds on amazon and open a product link',
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1, 2]);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[0].actionTaken).toBe('CLICK(1) -> CLICK(2)');
    expect(result.stepOutcomes[0].urlAfter).toBe(
      'https://shop.test/products/noise-canceling-earbuds-123'
    );
  });

  it('clicks a visible product result when the planner keeps choosing non-progress scrolling', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'SCROLL',
        direction: 'down',
        intent: 'find product result',
        verify: [{ predicate: 'exists', args: ['Cooling Towel'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'product detail opened' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub(
      'https://www.amazon.com/s?k=Cooling+Towels&ref=nb_sb_noss',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 10, role: 'searchbox', ariaLabel: 'Search Amazon', clickable: true },
          {
            id: 42,
            role: 'link',
            text: 'Cooling Towel 4 Pack for Neck and Face',
            href: '/dp/B0COOLTOWEL',
            clickable: true,
            importance: 100,
            inDominantGroup: true,
          },
        ]),
      {
        onClick: id => {
          if (id === 42) {
            runtime.currentUrl = 'https://www.amazon.com/dp/B0COOLTOWEL';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 } },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Search for Cooling Towels on amazon and click a product in search results to open its detail page',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[0].actionTaken).toBe('CLICK(42)');
    expect(runtime.scrollCalls).toEqual([]);
    expect(runtime.clickCalls).toEqual([42]);
    expect(result.stepOutcomes[0].urlAfter).toBe('https://www.amazon.com/dp/B0COOLTOWEL');
  });

  it('keeps click success when post-click effects fail during navigation handoff', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'product link',
        input: 'Cooling Towel product link',
        verify: [{ predicate: 'url_contains', args: ['/dp/'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'product detail opened' }),
    ]);
    const executor = new ProviderStub(['CLICK(42)']);
    let clickTriggered = false;
    const runtime: AgentRuntime = {
      currentUrl: 'https://www.amazon.com/s?k=cooling+towels',
      async snapshot(): Promise<Snapshot | null> {
        if (clickTriggered) {
          throw new Error('Could not establish connection. Receiving end does not exist.');
        }
        return makeSnapshot('https://www.amazon.com/s?k=cooling+towels', [
          {
            id: 42,
            role: 'link',
            text: 'Cooling Towel 4 Pack',
            href: '/dp/B0COOLTOWEL',
            clickable: true,
            importance: 100,
            inDominantGroup: true,
          },
        ]);
      },
      async goto(url: string): Promise<void> {
        this.currentUrl = url;
      },
      async click(elementId: number): Promise<void> {
        if (elementId === 42) {
          clickTriggered = true;
          this.currentUrl = 'https://www.amazon.com/dp/B0COOLTOWEL';
        }
      },
      async type(): Promise<void> {},
      async pressKey(): Promise<void> {},
      async scroll(): Promise<void> {},
      async getCurrentUrl(): Promise<string> {
        return this.currentUrl;
      },
      async getViewportHeight(): Promise<number> {
        return 1000;
      },
      async scrollBy(): Promise<boolean> {
        return true;
      },
    } as AgentRuntime & { currentUrl: string };

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 } },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Click a cooling towel product link and open its detail page',
    });

    expect(result.success).toBe(true);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SUCCESS);
    expect(result.stepOutcomes[0].actionTaken).toBe('CLICK(42)');
    expect(result.stepOutcomes[0].urlAfter).toBe('https://www.amazon.com/dp/B0COOLTOWEL');
  });

  it('does not navigate to copied example.com planner prompt URLs for unrelated tasks', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'NAVIGATE',
        target: 'https://example.com/search',
        verify: [{ predicate: 'url_contains', args: ['search'] }],
        reasoning: 'copied prompt example',
      }),
      JSON.stringify({
        action: 'TYPE_AND_SUBMIT',
        intent: 'searchbox',
        input: 'cooling towels',
        verify: [{ predicate: 'url_contains', args: ['/s?k='] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'search completed' }),
    ]);
    const executor = new ProviderStub(['NONE']);
    const runtime = new RuntimeStub(
      'https://www.amazon.com/',
      rt =>
        makeSnapshot(rt.currentUrl, [
          {
            id: 11,
            role: 'searchbox',
            ariaLabel: 'Search Amazon',
            clickable: true,
            importance: 100,
          },
        ]),
      {
        onPressKey: () => {
          runtime.currentUrl = 'https://www.amazon.com/s?k=cooling+towels&ref=nb_sb_noss';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
        recovery: { enabled: false },
      },
    });
    const result = await agent.runStepwise(runtime, {
      task: 'Search for cooling towels, then pick a product and click it to go to its detail page',
      startUrl: 'https://www.amazon.com/',
    });

    expect(runtime.gotoCalls).toEqual(['https://www.amazon.com/']);
    expect(result.stepOutcomes[0].status).toBe(StepStatus.SKIPPED);
    expect(result.stepOutcomes[0].actionTaken).toBe('SKIPPED(placeholder_navigation)');
    expect(result.stepOutcomes[1].actionTaken).toBe('TYPE(11, "cooling towels")');
    expect(runtime.currentUrl).toContain('www.amazon.com/s?k=cooling+towels');
  });

  it('uses vision execution when snapshot requires vision and executor supports it', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'continue', input: 'Continue' }),
      JSON.stringify({ action: 'DONE', reasoning: 'done' }),
    ]);
    const executor = new ProviderStub(['CLICK(7)'], { vision: true });
    let clicked = false;
    const runtime = new RuntimeStub(
      'https://shop.test/canvas',
      () =>
        makeSnapshot(
          clicked ? 'https://shop.test/after' : 'https://shop.test/canvas',
          [{ id: 7, role: 'button', text: 'Continue', clickable: true, importance: 100 }],
          { status: 'require_vision', screenshot: 'ZmFrZQ==' }
        ),
      {
        onClick: id => {
          if (id === 7) clicked = true;
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Continue on canvas page' });

    expect(result.success).toBe(true);
    expect(result.fallbackUsed).toBe(true);
    expect(result.stepOutcomes[0].usedVision).toBe(true);
    expect(executor.imageCalls).toHaveLength(1);
    expect(runtime.clickCalls).toEqual([7]);
  });

  it('records checkpoints and recovers to the last known good URL after a failure', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'open cart', input: 'Cart' }),
      JSON.stringify({ action: 'CLICK', intent: 'broken action', input: 'Broken' }),
      JSON.stringify({ action: 'DONE', reasoning: 'recovered' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)', 'CLICK(2)']);
    const runtime = new RuntimeStub(
      'https://shop.test/home',
      rt => {
        if (rt.currentUrl.includes('/cart')) {
          return makeSnapshot('https://shop.test/cart', [
            { id: 2, role: 'button', text: 'Broken', clickable: true, importance: 90 },
          ]);
        }
        return makeSnapshot('https://shop.test/home', [
          { id: 1, role: 'link', text: 'Cart', clickable: true, importance: 100 },
        ]);
      },
      {
        onClick: id => {
          if (id === 1) {
            runtime.currentUrl = 'https://shop.test/cart';
            return;
          }
          if (id === 2) {
            runtime.currentUrl = 'https://shop.test/error';
            throw new Error('click failed');
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Open cart and continue' });

    expect(result.success).toBe(true);
    expect(runtime.gotoCalls).toContain('https://shop.test/cart');
  });

  it('continues through add-to-cart drawer by clicking checkout control from modal integration', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'add to cart button', input: 'Add to Cart' }),
      JSON.stringify({ action: 'DONE', reasoning: 'checkout drawer handled' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    let stage: 'product' | 'drawer' | 'checkout' = 'product';
    const runtime = new RuntimeStub(
      'https://shop.test/product',
      () => {
        if (stage === 'drawer') {
          return makeSnapshot('https://shop.test/product', [
            { id: 1, role: 'button', text: 'Add to Cart', clickable: true, importance: 100 },
            {
              id: 9,
              role: 'button',
              text: 'Proceed to Checkout',
              clickable: true,
              importance: 110,
            },
            { id: 10, role: 'button', text: 'No Thanks', clickable: true, importance: 80 },
            { id: 11, role: 'text', text: 'Added to cart', importance: 20 },
            { id: 12, role: 'text', text: 'Subtotal', importance: 20 },
            { id: 13, role: 'text', text: 'Protection plan', importance: 20 },
            { id: 14, role: 'text', text: 'Drawer footer', importance: 20 },
          ]);
        }
        if (stage === 'checkout') {
          return makeSnapshot('https://shop.test/checkout', [
            { id: 20, role: 'heading', text: 'Checkout', importance: 100 },
          ]);
        }
        return makeSnapshot('https://shop.test/product', [
          { id: 1, role: 'button', text: 'Add to Cart', clickable: true, importance: 100 },
          { id: 2, role: 'text', text: 'Product Title', importance: 50 },
        ]);
      },
      {
        onClick: id => {
          if (id === 1) {
            stage = 'drawer';
            return;
          }
          if (id === 9) {
            stage = 'checkout';
            runtime.currentUrl = 'https://shop.test/checkout';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Add item to cart and continue' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1, 9]);
    expect(runtime.currentUrl).toContain('/checkout');
  });

  it('treats authentication boundaries as graceful success states', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'sign in button', input: 'Sign In' }),
    ]);
    const executor = new ProviderStub(['CLICK(5)']);
    const runtime = new RuntimeStub(
      'https://shop.test/ap/signin',
      () =>
        makeSnapshot('https://shop.test/ap/signin', [
          { id: 5, role: 'button', text: 'Sign In', clickable: true, importance: 100 },
        ]),
      {
        onClick: () => {
          throw new Error('credentials required');
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Continue checkout' });

    expect(result.success).toBe(true);
    expect(result.error).toContain('authentication boundary');
  });
});
