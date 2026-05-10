import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';
import { Tracer } from '../../../src/tracing/tracer';
import { TraceSink } from '../../../src/tracing/sink';
import type { TraceEvent } from '../../../src/tracing/types';

class MemoryTraceSink extends TraceSink {
  public readonly events: TraceEvent[] = [];

  emit(event: TraceEvent): void {
    this.events.push(event);
  }

  async close(): Promise<void> {}

  getSinkType(): string {
    return 'MemoryTraceSink';
  }
}

class ProviderStub extends LLMProvider {
  public readonly calls: Array<{ mode: 'text' | 'vision'; system: string; user: string }> = [];

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
    this.calls.push({ mode: 'text', system: systemPrompt, user: userPrompt });
    return {
      content: this.responses.length ? this.responses.shift()! : JSON.stringify({ action: 'DONE' }),
      modelName: this.modelName,
      promptTokens: 10,
      completionTokens: 4,
      totalTokens: 14,
    };
  }

  async generateWithImage(
    systemPrompt: string,
    userPrompt: string,
    _imageBase64: string
  ): Promise<LLMResponse> {
    this.calls.push({ mode: 'vision', system: systemPrompt, user: userPrompt });
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
  public gotoCalls: string[] = [];

  constructor(
    initialUrl: string,
    private readonly snapshotFactory: (runtime: RuntimeStub) => Snapshot | null,
    private readonly handlers: {
      onClick?: (elementId: number, runtime: RuntimeStub) => Promise<void> | void;
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

  async click(elementId: number): Promise<void> {
    this.clickCalls.push(elementId);
    await this.handlers.onClick?.(elementId, this);
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

function eventsOfType(events: TraceEvent[], type: string): TraceEvent[] {
  return events.filter(event => event.type === type);
}

describe('PlannerExecutorAgent tracing parity', () => {
  it('emits run start, planner action, step lifecycle, and checkout continuation decisions', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Add the item to cart',
        intent: 'add to cart button',
        verify: [],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'drawer continued' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    const traceSink = new MemoryTraceSink();
    const tracer = new Tracer('trace-checkout-continuation', traceSink);
    let stage: 'product' | 'drawer' | 'checkout' = 'product';

    const runtime = new RuntimeStub(
      'https://shop.test/product',
      () => {
        if (stage === 'drawer') {
          return makeSnapshot('https://shop.test/product', [
            { id: 1, role: 'button', text: 'Add to Cart', clickable: true, importance: 100 },
            { id: 9, role: 'button', text: 'Proceed to Checkout', clickable: true, importance: 90 },
            { id: 10, role: 'button', text: 'No Thanks', clickable: true, importance: 80 },
            { id: 11, role: 'text', text: 'Added to cart', importance: 30 },
            { id: 12, role: 'text', text: 'Subtotal', importance: 20 },
            { id: 13, role: 'text', text: 'Protection plan', importance: 15 },
            { id: 14, role: 'text', text: 'Drawer footer', importance: 10 },
          ]);
        }

        if (stage === 'checkout') {
          return makeSnapshot('https://shop.test/checkout', [
            { id: 20, role: 'heading', text: 'Checkout', importance: 100 },
          ]);
        }

        return makeSnapshot('https://shop.test/product', [
          { id: 1, role: 'button', text: 'Add to Cart', clickable: true, importance: 100 },
        ]);
      },
      {
        onClick: elementId => {
          if (elementId === 1) {
            stage = 'drawer';
          }
          if (elementId === 9) {
            stage = 'checkout';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor, tracer });
    await agent.runStepwise(runtime, { task: 'Add the item to cart and continue' });

    const runStart = eventsOfType(traceSink.events, 'run_start')[0];
    expect(runStart.data.agent).toBe('PlannerExecutorAgent');
    expect(runStart.data.config).toEqual(
      expect.objectContaining({ task: 'Add the item to cart and continue' })
    );

    const plannerActions = eventsOfType(traceSink.events, 'planner_action');
    expect(plannerActions[0].data.action).toBe('CLICK');
    expect(plannerActions[0].data.details).toEqual(
      expect.objectContaining({ intent: 'add to cart button', source: 'planner' })
    );
    expect(plannerActions[1].data.action).toBe('DONE');
    expect(plannerActions[1].step_id).toBeUndefined();

    const stepStart = eventsOfType(traceSink.events, 'step_start')[0];
    expect(stepStart.data.goal).toBe('Add the item to cart');

    const modalDecision = eventsOfType(traceSink.events, 'modal_action')[0];
    expect(modalDecision.data.action).toBe('continue_checkout');
    expect(modalDecision.data.element_id).toBe(9);

    const stepEnd = eventsOfType(traceSink.events, 'step_end')[0];
    expect(stepEnd.data.attempt).toBe(0);
    expect(stepEnd.data.llm).toEqual(
      expect.objectContaining({ response_text: 'CLICK(1)', model: 'provider' })
    );
    expect(stepEnd.data.exec?.success).toBe(true);
    expect(stepEnd.data.exec?.action).toBe('CLICK');
    expect(stepEnd.data.verify?.passed).toBe(true);

    const runEnd = eventsOfType(traceSink.events, 'run_end')[0];
    expect(runEnd.data.status).toBe('success');
  });

  it('emits recovery attempt and success events when a checkpoint restores the run', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'NAVIGATE',
        goal: 'Go to checkout',
        target: 'https://shop.test/checkout',
        verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        goal: 'Place the order',
        intent: 'place order button',
        verify: [{ predicate: 'url_contains', args: ['/done'] }],
        required: true,
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'recovered after checkpoint restore' }),
    ]);
    const executor = new ProviderStub(['CLICK(7)']);
    const traceSink = new MemoryTraceSink();
    const tracer = new Tracer('trace-recovery-success', traceSink);

    const runtime = new RuntimeStub(
      'https://shop.test/home',
      rt =>
        makeSnapshot(rt.currentUrl, [
          { id: 7, role: 'button', text: 'Place Order', clickable: true, importance: 100 },
        ]),
      {
        onClick: () => {
          runtime.currentUrl = 'https://shop.test/interstitial';
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      tracer,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 0 } },
    });

    await agent.runStepwise(runtime, { task: 'Recover from a failed place-order click' });

    const recoveryEvents = eventsOfType(traceSink.events, 'recovery');
    expect(recoveryEvents).toHaveLength(2);
    expect(recoveryEvents[0].data.details).toEqual(
      expect.objectContaining({ phase: 'attempt', checkpoint_url: 'https://shop.test/checkout' })
    );
    expect(recoveryEvents[1].data.success).toBe(true);
    expect(recoveryEvents[1].data.details).toEqual(
      expect.objectContaining({ phase: 'result', checkpoint_url: 'https://shop.test/checkout' })
    );
  });

  it('emits recovery failure and replan events when rollback cannot restore the step', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'NAVIGATE',
        goal: 'Open the cart',
        target: 'https://shop.test/cart',
        verify: [{ predicate: 'url_contains', args: ['/cart'] }],
      }),
      JSON.stringify({
        action: 'CLICK',
        goal: 'Go to checkout',
        intent: 'checkout button',
        verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
        required: true,
      }),
      JSON.stringify({
        mode: 'patch',
        replace_steps: [
          {
            id: 2,
            step: {
              id: 2,
              goal: 'Navigate directly to checkout',
              action: 'NAVIGATE',
              target: 'https://shop.test/checkout',
              verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
              required: true,
            },
          },
        ],
      }),
      JSON.stringify({ action: 'DONE', reasoning: 'replanned after recovery failed' }),
    ]);
    const executor = new ProviderStub(['CLICK(7)', 'CLICK(7)']);
    const traceSink = new MemoryTraceSink();
    const tracer = new Tracer('trace-replan', traceSink);
    let recoveryBroken = false;

    const runtime = new RuntimeStub(
      'https://shop.test/home',
      rt => {
        if (recoveryBroken && rt.currentUrl.includes('/cart')) {
          return makeSnapshot('https://shop.test/login', [
            { id: 50, role: 'heading', text: 'Sign in', importance: 100 },
          ]);
        }

        return makeSnapshot(rt.currentUrl, [
          { id: 7, role: 'button', text: 'Checkout', clickable: true, importance: 100 },
        ]);
      },
      {
        onClick: () => {
          runtime.currentUrl = 'https://shop.test/interstitial';
          recoveryBroken = true;
        },
      }
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      tracer,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 1 } },
    });

    await agent.runStepwise(runtime, { task: 'Replan after recovery fails' });

    const recoveryResult = eventsOfType(traceSink.events, 'recovery')[1];
    expect(recoveryResult.data.success).toBe(false);

    const replanEvents = eventsOfType(traceSink.events, 'replan');
    expect(replanEvents).toHaveLength(2);
    expect(replanEvents[0].data.details).toEqual(expect.objectContaining({ phase: 'start' }));
    expect(replanEvents[1].data.details).toEqual(
      expect.objectContaining({ phase: 'result', replacement_step_count: 1 })
    );
  });

  it('emits a terminal replan failure event when repair generation fails', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Go to checkout',
        intent: 'checkout button',
        verify: [{ predicate: 'url_contains', args: ['/checkout'] }],
        required: true,
      }),
      '{"mode":"patch","replace_steps":[]}',
    ]);
    const executor = new ProviderStub(['NONE']);
    const traceSink = new MemoryTraceSink();
    const tracer = new Tracer('trace-replan-failure', traceSink);
    const runtime = new RuntimeStub('https://shop.test/cart', rt =>
      makeSnapshot(rt.currentUrl, [
        { id: 7, role: 'button', text: 'Checkout', clickable: true, importance: 100 },
      ])
    );

    const agent = new PlannerExecutorAgent({
      planner,
      executor,
      tracer,
      config: { retry: { verifyTimeoutMs: 20, verifyPollMs: 1, maxReplans: 1 } },
    });

    await agent.runStepwise(runtime, { task: 'Replan after malformed repair output' });

    const replanEvents = eventsOfType(traceSink.events, 'replan');
    expect(replanEvents).toHaveLength(2);
    expect(replanEvents[0].data.details).toEqual(expect.objectContaining({ phase: 'start' }));
    expect(replanEvents[1].data.success).toBe(false);
    expect(replanEvents[1].data.details).toEqual(
      expect.objectContaining({
        phase: 'result',
        error: 'Repair planner returned no replacement steps',
      })
    );
  });

  it('emits vision fallback decisions when the snapshot requires vision', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Continue through the canvas gate',
        intent: 'continue button',
        verify: [],
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(99)'], { vision: true });
    const traceSink = new MemoryTraceSink();
    const tracer = new Tracer('trace-vision', traceSink);
    const runtime = new RuntimeStub('https://shop.test/canvas', () =>
      makeSnapshot(
        'https://shop.test/canvas',
        [{ id: 99, role: 'button', text: 'Continue', clickable: true, importance: 100 }],
        { status: 'require_vision', screenshot: 'ZmFrZQ==' }
      )
    );

    const agent = new PlannerExecutorAgent({ planner, executor, tracer });
    await agent.runStepwise(runtime, { task: 'Continue through the canvas flow' });

    const visionDecision = eventsOfType(traceSink.events, 'vision_decision')[0];
    expect(visionDecision.data.details).toEqual(
      expect.objectContaining({ use_vision: true, reason: 'page_requires_vision' })
    );
  });

  it('emits modal dismissal decisions for non-checkout overlays', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        goal: 'Open the promo modal',
        intent: 'open promo modal',
        verify: [],
      }),
      JSON.stringify({ action: 'DONE' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    const traceSink = new MemoryTraceSink();
    const tracer = new Tracer('trace-modal-dismissal', traceSink);
    let stage: 'base' | 'modal' | 'dismissed' = 'base';

    const runtime = new RuntimeStub(
      'https://shop.test/product',
      () => {
        if (stage === 'modal') {
          return makeSnapshot('https://shop.test/product', [
            { id: 1, role: 'button', text: 'Learn More', clickable: true, importance: 100 },
            { id: 10, role: 'button', text: 'No Thanks', clickable: true, importance: 90 },
            { id: 11, role: 'text', text: 'Add warranty?', importance: 70 },
            { id: 12, role: 'text', text: 'Recommended', importance: 60 },
            { id: 13, role: 'text', text: 'Overlay footer', importance: 50 },
            { id: 14, role: 'text', text: 'Extra coverage', importance: 40 },
          ]);
        }

        return makeSnapshot('https://shop.test/product', [
          { id: 1, role: 'button', text: 'Learn More', clickable: true, importance: 100 },
          {
            id: 2,
            role: 'text',
            text: stage === 'dismissed' ? 'Modal gone' : 'Product page',
            importance: 20,
          },
        ]);
      },
      {
        onClick: elementId => {
          if (elementId === 1) {
            stage = 'modal';
          }
          if (elementId === 10) {
            stage = 'dismissed';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor, tracer });
    await agent.runStepwise(runtime, { task: 'Clear the promo modal' });

    const modalDecision = eventsOfType(traceSink.events, 'modal_action')[0];
    expect(modalDecision.data.action).toBe('dismiss');
    expect(modalDecision.data.element_id).toBe(10);
  });
});
