/**
 * Tests for vision fallback detection.
 */

import {
  detectSnapshotFailure,
  shouldUseVision,
  type VisionFallbackResult,
} from '../../../src/agents/planner-executor/vision-fallback';
import type { Snapshot } from '../../../src/agents/planner-executor/plan-models';

describe('vision-fallback', () => {
  describe('detectSnapshotFailure', () => {
    it('should require vision for null snapshot', () => {
      const result = detectSnapshotFailure(null);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('snapshot_null');
    });

    it('should not require vision for snapshot with 10+ elements', () => {
      const snapshot: Snapshot = {
        url: 'https://example.com',
        title: 'Test Page',
        elements: Array.from({ length: 15 }, (_, i) => ({
          id: i,
          role: 'button',
          text: `Button ${i}`,
        })),
      };
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(false);
      expect(result.reason).toBeNull();
    });

    it('should require vision when status is require_vision', () => {
      const snapshot: Snapshot = {
        url: 'https://example.com',
        title: 'Test Page',
        elements: Array.from({ length: 5 }, (_, i) => ({
          id: i,
          role: 'button',
          text: `Button ${i}`,
        })),
        status: 'require_vision',
      };
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('require_vision');
    });

    it('should require vision when status is error', () => {
      const snapshot: Snapshot = {
        url: 'https://example.com',
        title: 'Test Page',
        elements: Array.from({ length: 5 }, (_, i) => ({
          id: i,
          role: 'button',
          text: `Button ${i}`,
        })),
        status: 'error',
      };
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('snapshot_error');
    });

    it('should require vision for too few elements (< 3)', () => {
      const snapshot: Snapshot = {
        url: 'https://example.com',
        title: 'Test Page',
        elements: [
          { id: 0, role: 'button', text: 'Button 0' },
          { id: 1, role: 'button', text: 'Button 1' },
        ],
      };
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('too_few_elements');
    });

    it('should not require vision with 3-9 elements and success status', () => {
      const snapshot: Snapshot = {
        url: 'https://example.com',
        title: 'Test Page',
        elements: Array.from({ length: 5 }, (_, i) => ({
          id: i,
          role: 'button',
          text: `Button ${i}`,
        })),
        status: 'success',
      };
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(false);
      expect(result.reason).toBeNull();
    });

    it('should require vision for low confidence diagnostics', () => {
      const snapshot = {
        url: 'https://example.com',
        elements: Array.from({ length: 5 }, (_, i) => ({
          id: i,
          role: 'button',
          text: `Button ${i}`,
        })),
        diagnostics: {
          confidence: 0.2,
        },
      } as unknown as Snapshot;
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('low_confidence');
    });

    it('should require vision for canvas page with few elements', () => {
      const snapshot = {
        url: 'https://example.com',
        elements: Array.from({ length: 4 }, (_, i) => ({
          id: i,
          role: 'button',
          text: `Button ${i}`,
        })),
        diagnostics: {
          hasCanvas: true,
        },
      } as unknown as Snapshot;
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('canvas_page');
    });

    it('should not require vision for canvas page with many elements', () => {
      const snapshot = {
        url: 'https://example.com',
        elements: Array.from({ length: 15 }, (_, i) => ({
          id: i,
          role: 'button',
          text: `Button ${i}`,
        })),
        diagnostics: {
          hasCanvas: true,
        },
      } as unknown as Snapshot;
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(false);
      expect(result.reason).toBeNull();
    });

    it('should handle empty elements array', () => {
      const snapshot: Snapshot = {
        url: 'https://example.com',
        title: 'Test Page',
        elements: [],
      };
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('too_few_elements');
    });

    it('should handle undefined elements', () => {
      const snapshot = {
        url: 'https://example.com',
      } as Snapshot;
      const result = detectSnapshotFailure(snapshot);
      expect(result.shouldUseVision).toBe(true);
      expect(result.reason).toBe('too_few_elements');
    });
  });

  describe('shouldUseVision', () => {
    it('should return true if snapshot failed', () => {
      expect(shouldUseVision(false, false)).toBe(true);
    });

    it('should return true if requiresVision is true', () => {
      expect(shouldUseVision(true, true)).toBe(true);
    });

    it('should return false if snapshot succeeded and vision not required', () => {
      expect(shouldUseVision(true, false)).toBe(false);
    });

    it('should return true if both conditions fail', () => {
      expect(shouldUseVision(false, true)).toBe(true);
    });
  });
});
