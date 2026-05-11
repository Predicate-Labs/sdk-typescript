import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import { PlannerExecutorAgent, type AgentRuntime } from '../../../src/agents/planner-executor';
import {
  PruningTaskCategory,
  pruneSnapshotForTask,
} from '../../../src/agents/planner-executor/category-pruner';
import {
  fullSnapshotContainsIntent,
  pruneWithRecovery,
} from '../../../src/agents/planner-executor/pruning-recovery';
import type { Snapshot, SnapshotElement } from '../../../src/agents/planner-executor/plan-models';

class ProviderStub extends LLMProvider {
  private responses: string[];
  public calls: Array<{ system: string; user: string; mode: 'text' | 'vision' }> = [];

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

  supportsVision(): boolean {
    return true;
  }

  async generate(systemPrompt: string, userPrompt: string): Promise<LLMResponse> {
    this.calls.push({ system: systemPrompt, user: userPrompt, mode: 'text' });
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

  async generateWithImage(systemPrompt: string, userPrompt: string): Promise<LLMResponse> {
    this.calls.push({ system: systemPrompt, user: userPrompt, mode: 'vision' });
    const content = this.responses.length ? this.responses.shift()! : 'CLICK(99)';
    return {
      content,
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    };
  }
}

class PromptAwareExecutorStub extends LLMProvider {
  public calls: Array<{ system: string; user: string; mode: 'text' | 'vision' }> = [];

  get modelName(): string {
    return 'prompt-aware-stub';
  }

  supportsJsonMode(): boolean {
    return true;
  }

  supportsVision(): boolean {
    return true;
  }

  async generate(systemPrompt: string, userPrompt: string): Promise<LLMResponse> {
    this.calls.push({ system: systemPrompt, user: userPrompt, mode: 'text' });
    return {
      content: userPrompt.includes('Add to Wishlist') ? 'CLICK(42)' : 'NONE',
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
  public snapshotCalls = 0;
  public snapshotLimits: number[] = [];
  public scrollByCalls = 0;

  constructor(
    initialUrl: string,
    private readonly snapshotFactory: (runtime: RuntimeStub) => Snapshot | null
  ) {
    this.currentUrl = initialUrl;
  }

  async snapshot(options?: { limit?: number }): Promise<Snapshot | null> {
    this.snapshotCalls += 1;
    this.snapshotLimits.push(options?.limit ?? -1);
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
    this.scrollByCalls++;
    return true;
  }
}

function makeElement(
  overrides: Partial<SnapshotElement> & Pick<SnapshotElement, 'id'>
): SnapshotElement {
  return {
    role: 'button',
    text: '',
    importance: 0,
    clickable: false,
    ...overrides,
  };
}

function makeSnapshot(elements: SnapshotElement[], overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    url: 'https://shop.test/page',
    title: 'Test Page',
    elements,
    screenshot: 'ZmFrZS1pbWFnZQ==',
    ...overrides,
  };
}

describe('pruning recovery', () => {
  it('relaxes when strict pruning keeps only non-actionable nodes', () => {
    const snapshot = makeSnapshot([
      makeElement({
        id: 1,
        role: 'text',
        text: 'Trail Shoe',
        importance: 900,
        inDominantGroup: true,
      }),
      makeElement({
        id: 2,
        role: 'text',
        text: '$129.00',
        importance: 850,
        nearbyText: 'Price',
      }),
      makeElement({
        id: 3,
        role: 'text',
        text: 'Free shipping',
        importance: 700,
        inDominantGroup: true,
      }),
      makeElement({
        id: 4,
        role: 'button',
        text: 'Add to Wishlist',
        importance: 300,
        clickable: true,
      }),
    ]);

    const strict = pruneSnapshotForTask(snapshot, {
      goal: 'add the product to cart',
      category: PruningTaskCategory.SHOPPING,
      relaxationLevel: 0,
    });
    const recovered = pruneWithRecovery(snapshot, {
      goal: 'add the product to cart',
      category: PruningTaskCategory.SHOPPING,
    });

    expect(strict.prunedElementCount).toBe(3);
    expect(strict.promptBlock).not.toContain('Add to Wishlist');
    expect(recovered.relaxationLevel).toBeGreaterThan(0);
    expect(recovered.promptBlock).toContain('Add to Wishlist');
  });

  it('relaxes pruning when the strict pass is too sparse', () => {
    const snapshot = makeSnapshot([
      makeElement({
        id: 1,
        role: 'button',
        text: 'Add to Wishlist',
        importance: 400,
        clickable: true,
      }),
      makeElement({ id: 2, role: 'button', text: 'Compare', importance: 350, clickable: true }),
      makeElement({
        id: 3,
        role: 'link',
        text: 'Privacy Policy',
        href: '/privacy',
        clickable: true,
      }),
    ]);

    const strict = pruneSnapshotForTask(snapshot, {
      goal: 'add the product to cart',
      category: PruningTaskCategory.SHOPPING,
      relaxationLevel: 0,
    });
    const recovered = pruneWithRecovery(snapshot, {
      goal: 'add the product to cart',
      category: PruningTaskCategory.SHOPPING,
    });

    expect(strict.prunedElementCount).toBeLessThan(recovered.prunedElementCount);
    expect(recovered.relaxationLevel).toBeGreaterThan(0);
  });

  it('retries with relaxed pruning before falling back to vision when the full snapshot still contains the target', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'add to wishlist button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['NONE', 'CLICK(42)']);
    const runtime = new RuntimeStub('https://shop.test/product', () =>
      makeSnapshot([
        makeElement({
          id: 40,
          role: 'link',
          text: 'Trail Shoe',
          href: '/product/trail-shoe',
          importance: 900,
          clickable: true,
          inDominantGroup: true,
        }),
        makeElement({
          id: 41,
          role: 'button',
          text: 'Add to Cart',
          importance: 950,
          clickable: true,
        }),
        makeElement({
          id: 42,
          role: 'button',
          text: 'Add to Wishlist',
          importance: 300,
          clickable: true,
        }),
        makeElement({
          id: 45,
          role: 'text',
          text: '$129.00',
          importance: 850,
          nearbyText: 'Price',
        }),
        makeElement({
          id: 43,
          role: 'link',
          text: 'Privacy Policy',
          href: '/privacy',
          importance: 50,
          clickable: true,
        }),
        makeElement({
          id: 44,
          role: 'link',
          text: 'Shipping details',
          href: '/shipping',
          importance: 75,
          clickable: true,
        }),
        makeElement({
          id: 46,
          role: 'link',
          text: 'Product details',
          href: '/details',
          importance: 250,
          clickable: true,
          inDominantGroup: true,
        }),
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitMax: 50,
          limitStep: 10,
          pruningMaxRelaxation: 0,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Add the product to the wishlist',
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([42]);
    expect(executor.calls.map(call => call.mode)).toEqual(['text', 'text']);
    expect(executor.calls[0].user).not.toContain('Add to Wishlist');
    expect(executor.calls[1].user).toContain('Add to Wishlist');
  });

  it('retries with relaxed pruning even when strict pruning yields zero actionable context', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'add to wishlist button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['NONE', 'CLICK(4)']);
    const runtime = new RuntimeStub('https://shop.test/product', () =>
      makeSnapshot([
        makeElement({
          id: 1,
          role: 'text',
          text: 'Trail Shoe',
          importance: 900,
          inDominantGroup: true,
        }),
        makeElement({
          id: 2,
          role: 'text',
          text: '$129.00',
          importance: 850,
          nearbyText: 'Price',
        }),
        makeElement({
          id: 3,
          role: 'text',
          text: 'Free shipping',
          importance: 700,
          inDominantGroup: true,
        }),
        makeElement({
          id: 4,
          role: 'button',
          text: 'Add to Wishlist',
          importance: 300,
          clickable: true,
        }),
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitMax: 50,
          limitStep: 10,
          pruningMaxRelaxation: 0,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Add the product to the wishlist',
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([4]);
    expect(executor.calls.map(call => call.mode)).toEqual(['text', 'text']);
    expect(executor.calls[0].user).toContain('Actionable: 0');
    expect(executor.calls[0].user).not.toContain('Add to Wishlist');
    expect(executor.calls[1].user).toContain('Add to Wishlist');
  });

  it('escalates to vision after relaxed pruning is exhausted', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'add to wishlist button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['NONE', 'NONE', 'CLICK(42)']);
    const runtime = new RuntimeStub('https://shop.test/product', () =>
      makeSnapshot([
        makeElement({
          id: 40,
          role: 'link',
          text: 'Trail Shoe',
          href: '/product/trail-shoe',
          importance: 900,
          clickable: true,
          inDominantGroup: true,
        }),
        makeElement({
          id: 41,
          role: 'button',
          text: 'Add to Cart',
          importance: 950,
          clickable: true,
        }),
        makeElement({
          id: 42,
          role: 'button',
          text: 'Add to Wishlist',
          importance: 300,
          clickable: true,
        }),
        makeElement({
          id: 45,
          role: 'text',
          text: '$129.00',
          importance: 850,
          nearbyText: 'Price',
        }),
        makeElement({
          id: 43,
          role: 'link',
          text: 'Privacy Policy',
          href: '/privacy',
          importance: 50,
          clickable: true,
        }),
        makeElement({
          id: 44,
          role: 'link',
          text: 'Shipping details',
          href: '/shipping',
          importance: 75,
          clickable: true,
        }),
        makeElement({
          id: 46,
          role: 'link',
          text: 'Product details',
          href: '/details',
          importance: 250,
          clickable: true,
          inDominantGroup: true,
        }),
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitMax: 50,
          limitStep: 10,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Add the product to the wishlist',
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([42]);
    expect(executor.calls.map(call => call.mode)).toEqual(['text', 'text', 'vision']);
  });

  it('continues limit escalation when raw snapshot count is high but pruned context is unusable', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'add to wishlist button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new PromptAwareExecutorStub();
    const runtime = new RuntimeStub('https://shop.test/product', rt => {
      const latestLimit = rt.snapshotLimits[rt.snapshotLimits.length - 1] ?? 0;
      const fillerElements = Array.from({ length: 10 }, (_, index) =>
        makeElement({
          id: index + 1,
          role: 'text',
          text: index % 2 === 0 ? `Product detail ${index}` : `$${100 + index}.00`,
          importance: 700 - index,
          inDominantGroup: index % 2 === 0,
          nearbyText: index % 2 === 0 ? undefined : 'Price',
        })
      );

      if (latestLimit >= 70) {
        return makeSnapshot([
          ...fillerElements,
          makeElement({
            id: 42,
            role: 'button',
            text: 'Add to Wishlist',
            importance: 300,
            clickable: true,
          }),
        ]);
      }

      return makeSnapshot(fillerElements);
    });

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitStep: 20,
          limitMax: 70,
          scrollAfterEscalation: false,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Add the product to the wishlist',
    });

    expect(result.success).toBe(true);
    expect(runtime.snapshotLimits.slice(0, 2)).toEqual([50, 70]);
    expect(runtime.clickCalls).toEqual([42]);
  });

  it('requires stronger intent overlap than a single shared token', () => {
    const snapshot = makeSnapshot([
      makeElement({
        id: 1,
        role: 'button',
        text: 'Add to Cart',
        importance: 900,
        clickable: true,
      }),
    ]);

    expect(fullSnapshotContainsIntent(snapshot, 'add to wishlist button')).toBe(false);
  });

  it('uses vision immediately when the snapshot explicitly requires it', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'continue button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(99)']);
    const runtime = new RuntimeStub('https://shop.test/canvas', () =>
      makeSnapshot(
        [
          makeElement({
            id: 99,
            role: 'button',
            text: 'Continue',
            importance: 900,
            clickable: true,
          }),
          makeElement({
            id: 100,
            role: 'button',
            text: 'Cancel',
            importance: 100,
            clickable: true,
          }),
          makeElement({ id: 101, role: 'link', text: 'Home', href: '/', clickable: true }),
        ],
        { status: 'require_vision' }
      )
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitMax: 50,
          limitStep: 10,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, { task: 'Continue through the canvas flow' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([99]);
    expect(executor.calls.map(call => call.mode)).toEqual(['vision']);
  });

  it('stops escalation early when element count is unchanged across iterations', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'login button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(5)']);
    const runtime = new RuntimeStub('https://example.com/login', rt => {
      return makeSnapshot([
        makeElement({ id: 1, role: 'textbox', text: 'Username', importance: 900 }),
        makeElement({ id: 2, role: 'textbox', text: 'Password', importance: 800 }),
        makeElement({ id: 3, role: 'button', text: 'Sign In', importance: 700, clickable: true }),
        makeElement({
          id: 4,
          role: 'link',
          text: 'Forgot Password',
          importance: 100,
          clickable: true,
        }),
        makeElement({ id: 5, role: 'button', text: 'Login', importance: 950, clickable: true }),
      ]);
    });

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitStep: 30,
          limitMax: 200,
          scrollAfterEscalation: false,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Click the login button',
    });

    expect(result.success).toBe(true);
    // Without the early-exit fix, Phase 1 would escalate: 50, 80, 110, 140, 170, 200
    // With the fix, each escalation call stops at limit=80 (element count unchanged from 50→80)
    // so limits 110, 140, 170 should never appear
    const escalationLimits = [110, 140, 170];
    for (const limit of escalationLimits) {
      expect(runtime.snapshotLimits).not.toContain(limit);
    }
  });

  it('skips scroll-after-escalation when page has fewer elements than limitBase', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'submit button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(3)']);
    const runtime = new RuntimeStub('https://example.com/form', rt => {
      return makeSnapshot([
        makeElement({ id: 1, role: 'textbox', text: 'Name', importance: 900 }),
        makeElement({ id: 2, role: 'textbox', text: 'Email', importance: 800 }),
        makeElement({ id: 3, role: 'button', text: 'Submit', importance: 700, clickable: true }),
      ]);
    });

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitStep: 30,
          limitMax: 200,
          scrollAfterEscalation: true,
          scrollDirections: ['down', 'up'] as const,
          scrollMaxAttempts: 3,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Click the submit button',
    });

    expect(result.success).toBe(true);
    expect(runtime.scrollByCalls).toBe(0);
    // Page has only 3 elements (< limitBase=50), so scroll-after-escalation is skipped entirely.
    // Escalation also stops early since element count doesn't grow beyond the base.
    const escalationLimits = [110, 140, 170];
    for (const limit of escalationLimits) {
      expect(runtime.snapshotLimits).not.toContain(limit);
    }
  });

  it('still escalates when element count increases across iterations', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'add to cart button',
        verify: [],
        required: true,
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(15)']);
    const runtime = new RuntimeStub('https://shop.test/product', rt => {
      const latestLimit = rt.snapshotLimits[rt.snapshotLimits.length - 1] ?? 0;
      if (latestLimit >= 80) {
        return makeSnapshot([
          ...Array.from({ length: 12 }, (_, i) =>
            makeElement({ id: i + 1, role: 'text', text: `Detail ${i}`, importance: 500 })
          ),
          makeElement({
            id: 15,
            role: 'button',
            text: 'Add to Cart',
            importance: 900,
            clickable: true,
          }),
        ]);
      }
      return makeSnapshot(
        Array.from({ length: 8 }, (_, i) =>
          makeElement({ id: i + 1, role: 'text', text: `Detail ${i}`, importance: 500 })
        )
      );
    });

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      intentHeuristics: {
        findElementForIntent: () => null,
        priorityOrder: () => [],
      },
      config: {
        stepwise: { maxSteps: 2 },
        snapshot: {
          limitBase: 50,
          limitStep: 30,
          limitMax: 200,
          scrollAfterEscalation: false,
        },
        retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 },
      },
    });

    const result = await agent.runStepwise(runtime, {
      task: 'Add the product to cart',
    });

    expect(result.success).toBe(true);
    expect(runtime.snapshotLimits.length).toBeGreaterThanOrEqual(2);
    expect(runtime.snapshotLimits[0]).toBe(50);
  });
});
