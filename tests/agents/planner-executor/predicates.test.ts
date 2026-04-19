/**
 * Tests for Predicate System
 */

import {
  urlContains,
  urlMatches,
  exists,
  notExists,
  elementCount,
  anyOf,
  allOf,
  buildPredicate,
  evaluatePredicates,
  type Predicate,
} from '../../../src/agents/planner-executor/predicates';
import type { Snapshot, SnapshotElement } from '../../../src/agents/planner-executor/plan-models';

describe('Predicates', () => {
  // Helper to create a mock snapshot
  const createSnapshot = (url: string, elements: Partial<SnapshotElement>[] = []): Snapshot => ({
    url,
    title: 'Test Page',
    elements: elements.map((el, i) => ({
      id: i + 1,
      role: el.role || 'button',
      text: el.text || '',
      ariaLabel: el.ariaLabel,
      ...el,
    })) as SnapshotElement[],
  });

  describe('urlContains', () => {
    it('should return true when URL contains substring', () => {
      const pred = urlContains('/cart');
      const snapshot = createSnapshot('https://example.com/cart');

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should return false when URL does not contain substring', () => {
      const pred = urlContains('/checkout');
      const snapshot = createSnapshot('https://example.com/cart');

      expect(pred.evaluate(snapshot)).toBe(false);
    });

    it('should be case-insensitive', () => {
      const pred = urlContains('/CART');
      const snapshot = createSnapshot('https://example.com/cart');

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should handle empty URL', () => {
      const pred = urlContains('/cart');
      const snapshot = createSnapshot('');

      expect(pred.evaluate(snapshot)).toBe(false);
    });
  });

  describe('urlMatches', () => {
    it('should match regex pattern', () => {
      const pred = urlMatches('/dp/[A-Z0-9]+');
      const snapshot = createSnapshot('https://amazon.com/dp/B08N5WRWNW');

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should return false for non-matching URL', () => {
      const pred = urlMatches('/dp/[A-Z0-9]+');
      const snapshot = createSnapshot('https://amazon.com/s?k=laptop');

      expect(pred.evaluate(snapshot)).toBe(false);
    });

    it('should fall back to substring match for invalid regex', () => {
      const pred = urlMatches('[invalid(regex');
      const snapshot = createSnapshot('https://example.com/[invalid(regex');

      expect(pred.evaluate(snapshot)).toBe(true);
    });
  });

  describe('exists', () => {
    it('should return true when element with text exists', () => {
      const pred = exists('Add to Cart');
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Add to Cart', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should return false when element does not exist', () => {
      const pred = exists('Checkout');
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Add to Cart', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(false);
    });

    it('should match partial text', () => {
      const pred = exists('cart');
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Add to Cart', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should match aria-label', () => {
      const pred = exists('search');
      const snapshot = createSnapshot('https://example.com', [
        { text: '', ariaLabel: 'Search products', role: 'textbox' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should match by role', () => {
      const pred = exists('button');
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Click me', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(true);
    });
  });

  describe('notExists', () => {
    it('should return true when element does not exist', () => {
      const pred = notExists('Error message');
      const snapshot = createSnapshot('https://example.com', [{ text: 'Success', role: 'alert' }]);

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should return false when element exists', () => {
      const pred = notExists('Error');
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Error: Something went wrong', role: 'alert' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(false);
    });
  });

  describe('elementCount', () => {
    it('should return true when count is within range', () => {
      const pred = elementCount('button', 2, 5);
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Button 1', role: 'button' },
        { text: 'Button 2', role: 'button' },
        { text: 'Button 3', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should return false when count is below minimum', () => {
      const pred = elementCount('button', 5);
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Button 1', role: 'button' },
        { text: 'Button 2', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(false);
    });

    it('should return false when count exceeds maximum', () => {
      const pred = elementCount('button', 0, 2);
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Button 1', role: 'button' },
        { text: 'Button 2', role: 'button' },
        { text: 'Button 3', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(false);
    });

    it('should work with only minimum specified', () => {
      const pred = elementCount('button', 1);
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Button 1', role: 'button' },
        { text: 'Button 2', role: 'button' },
      ]);

      expect(pred.evaluate(snapshot)).toBe(true);
    });
  });

  describe('anyOf', () => {
    it('should return true when any predicate passes', () => {
      const pred = anyOf(urlContains('/cart'), urlContains('/checkout'));
      const snapshot = createSnapshot('https://example.com/checkout');

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should return false when no predicate passes', () => {
      const pred = anyOf(urlContains('/cart'), urlContains('/checkout'));
      const snapshot = createSnapshot('https://example.com/home');

      expect(pred.evaluate(snapshot)).toBe(false);
    });

    it('should short-circuit on first true', () => {
      const pred = anyOf(urlContains('/cart'), urlContains('/cart'));
      const snapshot = createSnapshot('https://example.com/cart');

      expect(pred.evaluate(snapshot)).toBe(true);
    });
  });

  describe('allOf', () => {
    it('should return true when all predicates pass', () => {
      const pred = allOf(urlContains('amazon'), urlContains('/dp/'));
      const snapshot = createSnapshot('https://amazon.com/dp/B123456');

      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should return false when any predicate fails', () => {
      const pred = allOf(urlContains('amazon'), urlContains('/checkout'));
      const snapshot = createSnapshot('https://amazon.com/dp/B123456');

      expect(pred.evaluate(snapshot)).toBe(false);
    });
  });

  describe('buildPredicate', () => {
    it('should build url_contains predicate', () => {
      const pred = buildPredicate({ predicate: 'url_contains', args: ['/cart'] });
      const snapshot = createSnapshot('https://example.com/cart');

      expect(pred.name).toBe('url_contains');
      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should build url_matches predicate', () => {
      const pred = buildPredicate({ predicate: 'url_matches', args: ['/dp/.*'] });
      const snapshot = createSnapshot('https://example.com/dp/123');

      expect(pred.name).toBe('url_matches');
      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should build exists predicate', () => {
      const pred = buildPredicate({ predicate: 'exists', args: ['Add to Cart'] });
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Add to Cart', role: 'button' },
      ]);

      expect(pred.name).toBe('exists');
      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should build not_exists predicate', () => {
      const pred = buildPredicate({ predicate: 'not_exists', args: ['Error'] });
      const snapshot = createSnapshot('https://example.com', []);

      expect(pred.name).toBe('not_exists');
      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should build element_count predicate', () => {
      const pred = buildPredicate({ predicate: 'element_count', args: ['button', 1, 5] });
      const snapshot = createSnapshot('https://example.com', [
        { text: 'Button 1', role: 'button' },
        { text: 'Button 2', role: 'button' },
      ]);

      expect(pred.name).toBe('element_count');
      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should build any_of predicate', () => {
      const pred = buildPredicate({
        predicate: 'any_of',
        args: [
          { predicate: 'url_contains', args: ['/cart'] },
          { predicate: 'url_contains', args: ['/checkout'] },
        ],
      });
      const snapshot = createSnapshot('https://example.com/cart');

      expect(pred.name).toBe('any_of');
      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should build all_of predicate', () => {
      const pred = buildPredicate({
        predicate: 'all_of',
        args: [
          { predicate: 'url_contains', args: ['amazon'] },
          { predicate: 'url_contains', args: ['/dp/'] },
        ],
      });
      const snapshot = createSnapshot('https://amazon.com/dp/123');

      expect(pred.name).toBe('all_of');
      expect(pred.evaluate(snapshot)).toBe(true);
    });

    it('should handle unknown predicate gracefully', () => {
      const pred = buildPredicate({ predicate: 'unknown_predicate', args: ['foo'] });

      expect(pred.name).toBe('unknown:unknown_predicate');
      expect(pred.evaluate(createSnapshot(''))).toBe(true); // Always passes
    });
  });

  describe('evaluatePredicates', () => {
    it('should return true when all predicates pass', () => {
      const predicates = [
        { predicate: 'url_contains', args: ['/cart'] },
        { predicate: 'exists', args: ['Checkout'] },
      ];
      const snapshot = createSnapshot('https://example.com/cart', [
        { text: 'Proceed to Checkout', role: 'button' },
      ]);

      expect(evaluatePredicates(predicates, snapshot)).toBe(true);
    });

    it('should return false when any predicate fails', () => {
      const predicates = [
        { predicate: 'url_contains', args: ['/cart'] },
        { predicate: 'exists', args: ['Login'] },
      ];
      const snapshot = createSnapshot('https://example.com/cart', [
        { text: 'Checkout', role: 'button' },
      ]);

      expect(evaluatePredicates(predicates, snapshot)).toBe(false);
    });

    it('should return true for empty predicates array', () => {
      const snapshot = createSnapshot('https://example.com');

      expect(evaluatePredicates([], snapshot)).toBe(true);
    });

    it('should handle errors gracefully', () => {
      // Create an intentionally broken predicate by passing a circular reference
      const circular: any = { predicate: 'any_of', args: [] };
      circular.args.push(circular); // Create circular reference

      const predicates = [circular];
      const snapshot = createSnapshot('https://example.com');

      // Should not throw - the buildPredicate catches errors
      expect(() => evaluatePredicates(predicates, snapshot)).not.toThrow();
    });
  });
});
