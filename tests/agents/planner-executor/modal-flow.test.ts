import { LLMProvider, type LLMResponse } from '../../../src/llm-provider';
import {
  PlannerExecutorAgent,
  type AgentRuntime,
  type Snapshot,
} from '../../../src/agents/planner-executor';

class ProviderStub extends LLMProvider {
  private responses: string[];
  public generateCalls = 0;

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
    this.generateCalls += 1;
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

  constructor(
    initialUrl: string,
    private readonly snapshotFactory: (runtime: RuntimeStub) => Snapshot | null,
    private readonly handlers: {
      onClick?: (elementId: number, runtime: RuntimeStub) => Promise<void> | void;
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

describe('PlannerExecutorAgent modal flow parity', () => {
  it('dismisses a normal post-click modal before continuing', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'open promo modal', verify: [] }),
      JSON.stringify({ action: 'DONE', reasoning: 'modal handled' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
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
            { id: 13, role: 'text', text: 'Extra coverage', importance: 60 },
            { id: 14, role: 'text', text: 'Overlay footer', importance: 50 },
          ]);
        }
        return makeSnapshot('https://shop.test/product', [
          { id: 1, role: 'button', text: 'Learn More', clickable: true, importance: 100 },
          {
            id: 2,
            role: 'text',
            text: stage === 'dismissed' ? 'Modal gone' : 'Product page',
            importance: 50,
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

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Clear the promo modal' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1, 10]);
  });

  it('continues through a checkout drawer after an add-to-cart click', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'add to cart button', verify: [] }),
      JSON.stringify({ action: 'DONE', reasoning: 'drawer continued' }),
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
        ]);
      },
      {
        onClick: elementId => {
          if (elementId === 1) {
            stage = 'drawer';
          }
          if (elementId === 9) {
            stage = 'checkout';
            runtime.currentUrl = 'https://shop.test/checkout';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Add the item to cart and continue' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1, 9]);
    expect(runtime.currentUrl).toContain('/checkout');
  });

  it('finishes an add-to-cart task when the cart count confirms success', async () => {
    const planner = new ProviderStub([
      JSON.stringify({
        action: 'CLICK',
        intent: 'add_to_cart',
        input: 'Add to Cart',
        verify: [],
        required: true,
      }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    let stage: 'product' | 'cart-confirmed' = 'product';
    const runtime = new RuntimeStub(
      'https://shop.test/product',
      () => {
        if (stage === 'cart-confirmed') {
          return makeSnapshot('https://shop.test/product', [
            { id: 1, role: 'button', text: 'Add to Cart', clickable: true, importance: 100 },
            {
              id: 9,
              role: 'button',
              text: 'Cart contains 1 item Total $59.99',
              clickable: true,
              importance: 110,
            },
            { id: 10, role: 'text', text: 'Added to cart', importance: 90 },
          ]);
        }
        return makeSnapshot('https://shop.test/product', [
          { id: 1, role: 'button', text: 'Add to Cart', clickable: true, importance: 100 },
        ]);
      },
      {
        onClick: elementId => {
          if (elementId === 1) {
            stage = 'cart-confirmed';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, {
      task: 'Search for running shoes and add the item to cart',
    });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1]);
    expect(planner.generateCalls).toBe(1);
  });

  it('does not dismiss or auto-continue drawers with checkout or cart controls for unrelated clicks', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'open shipping info', verify: [] }),
      JSON.stringify({ action: 'DONE', reasoning: 'leave drawer alone' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    let stage: 'base' | 'drawer' = 'base';
    const runtime = new RuntimeStub(
      'https://shop.test/product',
      () => {
        if (stage === 'drawer') {
          return makeSnapshot('https://shop.test/product', [
            { id: 1, role: 'button', text: 'Shipping details', clickable: true, importance: 100 },
            { id: 9, role: 'button', text: 'View Cart', clickable: true, importance: 90 },
            { id: 10, role: 'button', text: 'No Thanks', clickable: true, importance: 80 },
            { id: 11, role: 'text', text: 'Drawer content', importance: 40 },
            { id: 12, role: 'text', text: 'Upsell option', importance: 30 },
            { id: 13, role: 'text', text: 'Footer copy', importance: 20 },
            { id: 14, role: 'text', text: 'Offer copy', importance: 20 },
          ]);
        }
        return makeSnapshot('https://shop.test/product', [
          { id: 1, role: 'button', text: 'Shipping details', clickable: true, importance: 100 },
        ]);
      },
      {
        onClick: elementId => {
          if (elementId === 1) {
            stage = 'drawer';
          }
        },
      }
    );

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Open shipping details' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1]);
  });

  it('does not auto-continue through a persistent cart control during an unrelated cart-help modal flow', async () => {
    const planner = new ProviderStub([
      JSON.stringify({ action: 'CLICK', intent: 'open cart help modal', verify: [] }),
      JSON.stringify({ action: 'DONE', reasoning: 'cart help dismissed' }),
    ]);
    const executor = new ProviderStub(['CLICK(1)']);
    let stage: 'base' | 'modal' | 'dismissed' = 'base';
    const runtime = new RuntimeStub(
      'https://shop.test/product',
      () => {
        if (stage === 'modal') {
          return makeSnapshot('https://shop.test/product', [
            { id: 1, role: 'button', text: 'Cart Help', clickable: true, importance: 100 },
            { id: 9, role: 'button', text: 'View Cart', clickable: true, importance: 95 },
            { id: 10, role: 'button', text: 'Close', clickable: true, importance: 90 },
            { id: 11, role: 'text', text: 'Cart help overlay', importance: 80 },
            { id: 12, role: 'text', text: 'Shipping info', importance: 70 },
            { id: 13, role: 'text', text: 'Overlay footer', importance: 60 },
            { id: 14, role: 'text', text: 'Support details', importance: 50 },
          ]);
        }
        return makeSnapshot('https://shop.test/product', [
          { id: 1, role: 'button', text: 'Cart Help', clickable: true, importance: 100 },
          { id: 9, role: 'button', text: 'View Cart', clickable: true, importance: 95 },
          {
            id: 2,
            role: 'text',
            text: stage === 'dismissed' ? 'Overlay closed' : 'Product page',
            importance: 40,
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

    const agent = new PlannerExecutorAgent({ planner, executor });
    const result = await agent.runStepwise(runtime, { task: 'Open cart help and close it' });

    expect(result.success).toBe(true);
    expect(runtime.clickCalls).toEqual([1, 10]);
  });
});
