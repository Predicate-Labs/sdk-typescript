import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  TaskCategory,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';

class ProviderStub extends LLMProvider {
  public calls: Array<{ system: string; user: string }> = [];
  public imageCalls: Array<{ system: string; user: string; imageBase64: string }> = [];

  constructor(
    private readonly responses: string[] = [],
    private readonly options: { vision?: boolean } = {}
  ) {
    super();
  }

  get modelName(): string {
    return this.options.vision ? 'vision-provider' : 'provider';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return this.options.vision ?? false;
  }

  async generate(systemPrompt: string, userPrompt: string): Promise<LLMResponse> {
    this.calls.push({ system: systemPrompt, user: userPrompt });
    return {
      content: this.responses.length ? this.responses.shift()! : 'DONE',
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 4,
      totalTokens: 14,
    };
  }

  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    imageBase64: string
  ): Promise<LLMResponse> {
    this.imageCalls.push({ system: systemPrompt, user: userPrompt, imageBase64 });
    return {
      content: this.responses.length ? this.responses.shift()! : 'CLICK(1)',
      modelName: this.modelName,
      promptTokens: 12,
      completionTokens: 5,
      totalTokens: 17,
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
    if (snap) {
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

describe('Composable heuristics integration', () => {
  it('uses planner heuristic_hints before invoking the executor', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'add_to_cart',
        input: 'Add to Cart',
        heuristic_hints: [
          {
            intent_pattern: 'add_to_cart',
            text_patterns: ['add to cart'],
            role_filter: ['button'],
            priority: 10,
          },
        ],
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(999)']);
    const runtime = new RuntimeStub('https://shop.test/product', () =>
      makeSnapshot('https://shop.test/product', [
        { id: 1, role: 'button', text: 'Add to Cart', clickable: true },
        { id: 2, role: 'link', text: 'Product Details', clickable: true },
        { id: 3, role: 'button', text: 'Wishlist', clickable: true },
      ])
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Add the item to cart' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1]);
    expect(executor.calls).toHaveLength(0);
  });

  it('falls back to the executor when heuristics cannot resolve the element', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'target_primary_cta',
        input: 'Promo',
        heuristic_hints: [
          {
            intent_pattern: 'target_primary_cta',
            text_patterns: ['definitely absent'],
            role_filter: ['button'],
            priority: 10,
          },
        ],
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(7)']);
    const runtime = new RuntimeStub('https://shop.test', () =>
      makeSnapshot('https://shop.test', [
        { id: 7, role: 'button', text: 'Open Promo', clickable: true },
        { id: 8, role: 'link', text: 'Home', clickable: true },
        { id: 9, role: 'button', text: 'Dismiss', clickable: true },
      ])
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Open the promo modal' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([7]);
    expect(executor.calls).toHaveLength(1);
  });

  it('uses task-category defaults when planner hints are absent', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'complete purchase',
        input: 'Place Order',
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(99)']);
    const runtime = new RuntimeStub('https://shop.test/checkout', () =>
      makeSnapshot('https://shop.test/checkout', [
        { id: 5, role: 'button', text: 'Place Order', clickable: true },
        { id: 6, role: 'link', text: 'Edit Cart', clickable: true },
        { id: 7, role: 'button', text: 'Apply Coupon', clickable: true },
      ])
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, {
      task: 'Complete the purchase',
      category: TaskCategory.TRANSACTION,
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([5]);
    expect(executor.calls).toHaveLength(0);
  });

  it('keeps the existing vision fallback path when the snapshot requires vision', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'continue',
        input: 'Continue',
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(3)'], { vision: true });
    const runtime = new RuntimeStub('https://shop.test/canvas', () =>
      makeSnapshot(
        'https://shop.test/canvas',
        [{ id: 3, role: 'button', text: 'Continue', clickable: true }],
        { status: 'require_vision', screenshot: 'ZmFrZQ==' }
      )
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    await agent.runStepwise(runtime, { task: 'Continue through the canvas step' });

    expect(executor.imageCalls).toHaveLength(1);
    expect(runtime.clickCalls).toEqual([3]);
  });
});
