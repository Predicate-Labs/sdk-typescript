/**
 * Tests for modal/overlay dismissal logic.
 */

import {
  findDismissalTarget,
  detectModalAppearance,
  detectModalDismissed,
  DEFAULT_MODAL_CONFIG,
  type ModalDismissalConfig,
} from '../../../src/agents/planner-executor/modal-dismissal';
import type { SnapshotElement } from '../../../src/agents/planner-executor/plan-models';

describe('modal-dismissal', () => {
  describe('DEFAULT_MODAL_CONFIG', () => {
    it('should have expected default values', () => {
      expect(DEFAULT_MODAL_CONFIG.enabled).toBe(true);
      expect(DEFAULT_MODAL_CONFIG.maxAttempts).toBe(2);
      expect(DEFAULT_MODAL_CONFIG.minNewElements).toBe(5);
      expect(DEFAULT_MODAL_CONFIG.roleFilter).toContain('button');
      expect(DEFAULT_MODAL_CONFIG.roleFilter).toContain('link');
      expect(DEFAULT_MODAL_CONFIG.dismissPatterns).toContain('no thanks');
      expect(DEFAULT_MODAL_CONFIG.dismissPatterns).toContain('close');
      expect(DEFAULT_MODAL_CONFIG.iconPatterns).toContain('x');
      expect(DEFAULT_MODAL_CONFIG.iconPatterns).toContain('×');
      expect(DEFAULT_MODAL_CONFIG.checkoutPatterns).toContain('checkout');
    });
  });

  describe('findDismissalTarget', () => {
    const createButton = (id: number, text: string, role = 'button'): SnapshotElement => ({
      id,
      role,
      text,
    });

    it('should find "No thanks" button', () => {
      const elements: SnapshotElement[] = [
        createButton(1, 'Add Protection'),
        createButton(2, 'No thanks'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('no thanks');
      expect(result.hasCheckoutButton).toBe(false);
    });

    it('should find "Close" button', () => {
      const elements: SnapshotElement[] = [createButton(1, 'Subscribe'), createButton(2, 'Close')];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('close');
    });

    it('should find "X" icon button (exact match)', () => {
      const elements: SnapshotElement[] = [createButton(1, 'Accept'), createButton(2, 'x')];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('x');
    });

    it('should find multiplication sign icon', () => {
      const elements: SnapshotElement[] = [createButton(1, 'Subscribe'), createButton(2, '×')];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('×');
    });

    it('should not match "x" within words like "mexico"', () => {
      const elements: SnapshotElement[] = [
        createButton(1, 'Ship to Mexico'),
        createButton(2, 'Continue'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('continue');
    });

    it('should not match "close" within "enclosed"', () => {
      const elements: SnapshotElement[] = [
        createButton(1, 'Enclosed package'),
        createButton(2, 'Dismiss'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.matchedPattern).toBe('dismiss');
    });

    it('should prioritize icon patterns over text patterns', () => {
      const elements: SnapshotElement[] = [createButton(1, 'No thanks'), createButton(2, 'x')];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('x');
    });

    it('should prioritize earlier dismiss patterns', () => {
      // "no thanks" comes before "close" in the pattern list
      const elements: SnapshotElement[] = [createButton(1, 'Close'), createButton(2, 'No thanks')];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('no thanks');
    });

    it('should skip global nav cart links', () => {
      const elements: SnapshotElement[] = [
        { id: 1, role: 'link', text: 'Cart', href: '/cart' },
        createButton(2, 'Close'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
    });

    it('should skip cart count indicators', () => {
      const elements: SnapshotElement[] = [
        { id: 1, role: 'button', text: '3' },
        createButton(2, 'Dismiss'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
    });

    it('should not dismiss when checkout button found', () => {
      const elements: SnapshotElement[] = [
        createButton(1, 'Proceed to Checkout'),
        createButton(2, 'Close'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(false);
      expect(result.elementId).toBeNull();
      expect(result.hasCheckoutButton).toBe(true);
    });

    it('should not dismiss when view cart button found', () => {
      const elements: SnapshotElement[] = [
        createButton(1, 'View Cart'),
        createButton(2, 'No thanks'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(false);
      expect(result.hasCheckoutButton).toBe(true);
    });

    it('should not dismiss when cart/checkout link found', () => {
      const elements: SnapshotElement[] = [
        { id: 1, role: 'link', text: 'Review Order', href: '/checkout/review' },
        createButton(2, 'Close'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(false);
      expect(result.hasCheckoutButton).toBe(true);
    });

    it('should only consider buttons and links', () => {
      const elements: SnapshotElement[] = [
        { id: 1, role: 'text', text: 'Close' },
        { id: 2, role: 'heading', text: 'No thanks' },
        createButton(3, 'OK'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(3);
    });

    it('should use aria-label for matching', () => {
      const elements: SnapshotElement[] = [
        { id: 1, role: 'button', text: '', ariaLabel: 'Close dialog' },
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(1);
      expect(result.matchedPattern).toBe('close');
    });

    it('should return not found for empty elements', () => {
      const result = findDismissalTarget([]);
      expect(result.found).toBe(false);
      expect(result.elementId).toBeNull();
      expect(result.hasCheckoutButton).toBe(false);
    });

    it('should respect disabled config', () => {
      const config: ModalDismissalConfig = {
        ...DEFAULT_MODAL_CONFIG,
        enabled: false,
      };
      const elements: SnapshotElement[] = [createButton(1, 'No thanks')];

      const result = findDismissalTarget(elements, config);
      expect(result.found).toBe(false);
    });

    it('should use custom dismiss patterns', () => {
      const config: ModalDismissalConfig = {
        ...DEFAULT_MODAL_CONFIG,
        dismissPatterns: ['custom dismiss'],
      };
      const elements: SnapshotElement[] = [
        createButton(1, 'No thanks'),
        createButton(2, 'Custom Dismiss'),
      ];

      const result = findDismissalTarget(elements, config);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
      expect(result.matchedPattern).toBe('custom dismiss');
    });

    it('should skip elements without id', () => {
      const elements: SnapshotElement[] = [
        { role: 'button', text: 'No thanks' } as SnapshotElement,
        createButton(2, 'Close'),
      ];

      const result = findDismissalTarget(elements);
      expect(result.found).toBe(true);
      expect(result.elementId).toBe(2);
    });
  });

  describe('detectModalAppearance', () => {
    it('should detect modal appearance with many new elements', () => {
      const preElements = new Set([1, 2, 3]);
      const postElements = new Set([1, 2, 3, 4, 5, 6, 7, 8]);

      expect(detectModalAppearance(preElements, postElements)).toBe(true);
    });

    it('should not detect modal with few new elements', () => {
      const preElements = new Set([1, 2, 3]);
      const postElements = new Set([1, 2, 3, 4, 5]);

      expect(detectModalAppearance(preElements, postElements)).toBe(false);
    });

    it('should detect modal with exactly minNewElements', () => {
      const preElements = new Set([1, 2]);
      const postElements = new Set([1, 2, 3, 4, 5, 6, 7]);

      expect(detectModalAppearance(preElements, postElements, 5)).toBe(true);
    });

    it('should not detect modal when elements removed', () => {
      const preElements = new Set([1, 2, 3, 4, 5, 6, 7, 8]);
      const postElements = new Set([1, 2]);

      expect(detectModalAppearance(preElements, postElements)).toBe(false);
    });

    it('should handle empty pre-elements', () => {
      const preElements = new Set<number>();
      const postElements = new Set([1, 2, 3, 4, 5, 6]);

      expect(detectModalAppearance(preElements, postElements)).toBe(true);
    });

    it('should handle empty post-elements', () => {
      const preElements = new Set([1, 2, 3]);
      const postElements = new Set<number>();

      expect(detectModalAppearance(preElements, postElements)).toBe(false);
    });

    it('should use custom minNewElements', () => {
      const preElements = new Set([1, 2]);
      const postElements = new Set([1, 2, 3, 4, 5]);

      expect(detectModalAppearance(preElements, postElements, 3)).toBe(true);
      expect(detectModalAppearance(preElements, postElements, 5)).toBe(false);
    });
  });

  describe('detectModalDismissed', () => {
    it('should detect modal dismissed with many removed elements', () => {
      const preElements = new Set([1, 2, 3, 4, 5, 6, 7, 8]);
      const postElements = new Set([1, 2, 3]);

      expect(detectModalDismissed(preElements, postElements)).toBe(true);
    });

    it('should not detect dismissal with few removed elements', () => {
      const preElements = new Set([1, 2, 3, 4, 5]);
      const postElements = new Set([1, 2, 3, 4]);

      expect(detectModalDismissed(preElements, postElements)).toBe(false);
    });

    it('should detect dismissal with exactly minRemovedElements', () => {
      const preElements = new Set([1, 2, 3, 4, 5]);
      const postElements = new Set([1, 2]);

      expect(detectModalDismissed(preElements, postElements, 3)).toBe(true);
    });

    it('should not detect dismissal when elements added', () => {
      const preElements = new Set([1, 2]);
      const postElements = new Set([1, 2, 3, 4, 5, 6, 7, 8]);

      expect(detectModalDismissed(preElements, postElements)).toBe(false);
    });

    it('should handle empty pre-elements', () => {
      const preElements = new Set<number>();
      const postElements = new Set([1, 2, 3]);

      expect(detectModalDismissed(preElements, postElements)).toBe(false);
    });

    it('should handle empty post-elements', () => {
      const preElements = new Set([1, 2, 3, 4, 5]);
      const postElements = new Set<number>();

      expect(detectModalDismissed(preElements, postElements)).toBe(true);
    });

    it('should use custom minRemovedElements', () => {
      const preElements = new Set([1, 2, 3, 4, 5]);
      const postElements = new Set([1, 2, 3]);

      expect(detectModalDismissed(preElements, postElements, 2)).toBe(true);
      expect(detectModalDismissed(preElements, postElements, 5)).toBe(false);
    });
  });
});
