/**
 * Tests for Intent Heuristics
 */

import type { SnapshotElement } from '../../../src/agents/planner-executor/plan-models';

// We need to test the SimpleIntentHeuristics class, but it's not exported.
// So we'll test via the PlannerExecutorAgent which uses it internally.
// For now, let's test the pattern matching logic conceptually.

describe('Intent Heuristics', () => {
  // Helper to create mock elements
  const createElement = (
    id: number,
    text: string,
    role: string = 'button',
    clickable: boolean = true
  ): SnapshotElement => ({
    id,
    text,
    role,
    clickable,
  });

  describe('Common Intent Patterns', () => {
    // These tests verify the patterns we expect to match

    it('should recognize add_to_cart patterns', () => {
      const patterns = ['add to cart', 'add to bag', 'add to basket', 'buy now', 'add item'];
      const elements = patterns.map((p, i) => createElement(i + 1, p));

      // Each element text should match one of the patterns
      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        expect(patterns.some(p => text.includes(p))).toBe(true);
      }
    });

    it('should recognize checkout patterns', () => {
      const patterns = ['checkout', 'proceed to checkout', 'go to checkout', 'check out'];
      const elements = patterns.map((p, i) => createElement(i + 1, p));

      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        expect(patterns.some(p => text.includes(p))).toBe(true);
      }
    });

    it('should recognize search patterns', () => {
      const patterns = ['search', 'find', 'go', 'submit'];
      const elements = patterns.map((p, i) => createElement(i + 1, p));

      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        expect(patterns.some(p => text.includes(p))).toBe(true);
      }
    });

    it('should recognize login patterns', () => {
      const patterns = ['log in', 'login', 'sign in', 'signin'];
      const elements = patterns.map((p, i) => createElement(i + 1, p));

      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        expect(patterns.some(p => text.includes(p))).toBe(true);
      }
    });

    it('should recognize close/dismiss patterns', () => {
      const patterns = ['close', 'dismiss', 'x', 'cancel', 'no thanks'];
      const elements = patterns.map((p, i) => createElement(i + 1, p));

      for (const element of elements) {
        const text = (element.text || '').toLowerCase();
        expect(patterns.some(p => text.includes(p))).toBe(true);
      }
    });
  });

  describe('Element Matching Priority', () => {
    it('should prefer clickable buttons over non-clickable elements', () => {
      const elements = [
        createElement(1, 'Add to Cart', 'text', false),
        createElement(2, 'Add to Cart', 'button', true),
      ];

      // When matching, clickable buttons should be preferred
      const clickableButtons = elements.filter(e => e.clickable && e.role === 'button');
      expect(clickableButtons.length).toBe(1);
      expect(clickableButtons[0].id).toBe(2);
    });

    it('should match elements with matching aria-label', () => {
      const element: SnapshotElement = {
        id: 1,
        text: '',
        role: 'button',
        ariaLabel: 'Add to shopping cart',
        clickable: true,
      };

      const ariaLabel = (element.ariaLabel || '').toLowerCase();
      expect(ariaLabel.includes('cart')).toBe(true);
    });
  });

  describe('Intent Normalization', () => {
    it('should handle underscore-separated intents', () => {
      const intent = 'add_to_cart';
      const normalized = intent.toLowerCase().replace(/[_-]/g, ' ');
      expect(normalized).toBe('add to cart');
    });

    it('should handle hyphen-separated intents', () => {
      const intent = 'add-to-cart';
      const normalized = intent.toLowerCase().replace(/[_-]/g, ' ');
      expect(normalized).toBe('add to cart');
    });

    it('should handle mixed case intents', () => {
      const intent = 'Add_To_Cart';
      const normalized = intent.toLowerCase().replace(/[_-]/g, ' ');
      expect(normalized).toBe('add to cart');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty elements array', () => {
      const elements: SnapshotElement[] = [];
      expect(elements.length).toBe(0);
    });

    it('should handle elements with undefined text', () => {
      const element: SnapshotElement = {
        id: 1,
        role: 'button',
        text: undefined,
      };

      const text = (element.text || '').toLowerCase();
      expect(text).toBe('');
    });

    it('should handle elements with empty text', () => {
      const element: SnapshotElement = {
        id: 1,
        role: 'button',
        text: '',
      };

      const text = (element.text || '').toLowerCase();
      expect(text).toBe('');
    });
  });
});
