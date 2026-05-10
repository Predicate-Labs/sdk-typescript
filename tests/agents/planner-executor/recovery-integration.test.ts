import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';

class ProviderStub extends LLMProvider {
  private responses: string[];

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

  async generate(): Promise<LLMResponse> {
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
  public gotoCalls: string[] = [];

  constructor(
    initialUrl: string,
    private readonly snapshotFactory: (runtime: RuntimeStub) => Snapshot | null,
    private readonly handlers: {
      onGoto?: (url: string, runtime: RuntimeStub) => Promise<void> | void;
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
    await this.handlers.onGoto?.(url, this);
  }

  async click(): Promise<void> {
    this.currentUrl = this.currentUrl.replace(/\/(checkout|cart)/, '/error');
    throw new Error('executor click failed');
  }

  async type(): Promise<void> {}

  async pressKey(): Promise<void> {}

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

describe('PlannerExecutorAgent recovery integration parity', () => {
  it('tries the latest checkpoint first, falls back to an earlier verified checkpoint, and only resumes after verification', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'NAVIGATE',
        target: 'https://shop.test/cart',
        verify: [
          { predicate: 'url_contains', args: ['/cart'] },
          { predicate: 'exists', args: ['Cart Summary'] },
        ],
      }),
      JSON.stringify({
        action: 'NAVIGATE',
        target: 'https://shop.test/checkout',
        verify: [
          { predicate: 'url_contains', args: ['/checkout'] },
          { predicate: 'exists', args: ['Review Order'] },
        ],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'broken checkout control',
        verify: [{ predicate: 'exists', args: ['Order placed'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'recovered to safe checkpoint' }),
    ]);
    const executor = new ProviderStub(['CLICK(99)']);
    let revisitingCheckoutFromRecovery = false;
    const runtime = new RuntimeStub(
      'https://shop.test/home',
      rt => {
        if (rt.currentUrl.includes('/checkout')) {
          return makeSnapshot(rt.currentUrl, [
            {
              id: 1,
              role: 'heading',
              text: revisitingCheckoutFromRecovery ? 'Checkout shell' : 'Review Order',
              importance: 100,
            },
          ]);
        }
        if (rt.currentUrl.includes('/cart')) {
          return makeSnapshot(rt.currentUrl, [
            { id: 2, role: 'heading', text: 'Cart Summary', importance: 100 },
          ]);
        }
        return makeSnapshot(rt.currentUrl, [
          { id: 3, role: 'link', text: 'Start', clickable: true, importance: 50 },
        ]);
      },
      {
        onGoto: url => {
          if (url.includes('/checkout') && runtime.gotoCalls.length >= 3) {
            revisitingCheckoutFromRecovery = true;
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
        recovery: { enabled: true, maxRecoveryAttempts: 1, maxCheckpoints: 5 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Recover from a broken checkout click',
    });

    expect(result.success).toBe(true);
    expect(runtime.gotoCalls).toEqual([
      'https://shop.test/cart',
      'https://shop.test/checkout',
      'https://shop.test/checkout',
      'https://shop.test/cart',
    ]);
    expect(runtime.currentUrl).toBe('https://shop.test/cart');
  });

  it('preserves structured verification predicates when deciding whether to fall back from the latest checkpoint', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'NAVIGATE',
        target: 'https://shop.test/cart',
        verify: [{ predicate: 'exists', args: ['Cart Summary'] }],
      }),
      JSON.stringify({
        action: 'NAVIGATE',
        target: 'https://shop.test/checkout',
        verify: [
          {
            predicate: 'any_of',
            args: [
              { predicate: 'exists', args: ['Review Order'] },
              { predicate: 'exists', args: ['Confirm Purchase'] },
            ],
          },
        ],
      }),
      JSON.stringify({
        action: 'CLICK',
        intent: 'broken checkout control',
        verify: [{ predicate: 'exists', args: ['Order placed'] }],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'fell back from invalid structured checkpoint' }),
    ]);
    const executor = new ProviderStub(['CLICK(99)']);
    let revisitingCheckoutFromRecovery = false;
    const runtime = new RuntimeStub(
      'https://shop.test/home',
      rt => {
        if (rt.currentUrl.includes('/checkout')) {
          return makeSnapshot('https://shop.test/checkout', [
            {
              id: 3,
              role: 'heading',
              text: revisitingCheckoutFromRecovery ? 'Checkout shell' : 'Review Order',
              importance: 100,
            },
          ]);
        }
        if (rt.currentUrl.includes('/cart')) {
          return makeSnapshot('https://shop.test/cart', [
            { id: 2, role: 'heading', text: 'Cart Summary', importance: 100 },
          ]);
        }
        return makeSnapshot(rt.currentUrl, [
          { id: 1, role: 'link', text: 'Start', clickable: true, importance: 50 },
        ]);
      },
      {
        onGoto: url => {
          if (url.includes('/checkout') && runtime.gotoCalls.length >= 3) {
            revisitingCheckoutFromRecovery = true;
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
        recovery: { enabled: true, maxRecoveryAttempts: 1, maxCheckpoints: 5 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Recover with a structured checkpoint predicate',
    });

    expect(result.success).toBe(true);
    expect(runtime.gotoCalls).toEqual([
      'https://shop.test/cart',
      'https://shop.test/checkout',
      'https://shop.test/checkout',
      'https://shop.test/cart',
    ]);
    expect(runtime.currentUrl).toBe('https://shop.test/cart');
  });
});
