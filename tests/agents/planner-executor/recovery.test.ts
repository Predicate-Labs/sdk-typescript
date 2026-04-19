/**
 * Tests for recovery navigation state management.
 */

import {
  RecoveryState,
  DEFAULT_RECOVERY_CONFIG,
  type RecoveryCheckpoint,
  type RecoveryNavigationConfig,
} from '../../../src/agents/planner-executor/recovery';

describe('recovery', () => {
  describe('DEFAULT_RECOVERY_CONFIG', () => {
    it('should have expected default values', () => {
      expect(DEFAULT_RECOVERY_CONFIG.enabled).toBe(true);
      expect(DEFAULT_RECOVERY_CONFIG.maxRecoveryAttempts).toBe(2);
      expect(DEFAULT_RECOVERY_CONFIG.trackSuccessfulUrls).toBe(true);
      expect(DEFAULT_RECOVERY_CONFIG.maxCheckpoints).toBe(10);
    });
  });

  describe('RecoveryState', () => {
    let state: RecoveryState;

    beforeEach(() => {
      state = new RecoveryState();
    });

    describe('initial state', () => {
      it('should start with no checkpoints', () => {
        expect(state.length).toBe(0);
        expect(state.lastSuccessfulUrl).toBeNull();
        expect(state.lastSuccessfulStep).toBeNull();
      });

      it('should have zero attempts used', () => {
        expect(state.attemptsUsed).toBe(0);
      });

      it('should not be able to recover with no checkpoints', () => {
        expect(state.canRecover()).toBe(false);
      });

      it('should return null for getRecoveryTarget with no checkpoints', () => {
        expect(state.getRecoveryTarget()).toBeNull();
      });
    });

    describe('recordCheckpoint', () => {
      it('should record a checkpoint', () => {
        const checkpoint = state.recordCheckpoint({
          url: 'https://example.com/page1',
          stepIndex: 0,
          snapshotDigest: 'abc123',
          predicatesPassed: ['url_contains("/page1")'],
        });

        expect(checkpoint.url).toBe('https://example.com/page1');
        expect(checkpoint.stepIndex).toBe(0);
        expect(checkpoint.snapshotDigest).toBe('abc123');
        expect(checkpoint.predicatesPassed).toEqual(['url_contains("/page1")']);
        expect(checkpoint.timestamp).toBeInstanceOf(Date);
      });

      it('should update lastSuccessfulUrl and lastSuccessfulStep', () => {
        state.recordCheckpoint({
          url: 'https://example.com/page1',
          stepIndex: 0,
          snapshotDigest: 'abc123',
          predicatesPassed: [],
        });

        expect(state.lastSuccessfulUrl).toBe('https://example.com/page1');
        expect(state.lastSuccessfulStep).toBe(0);

        state.recordCheckpoint({
          url: 'https://example.com/page2',
          stepIndex: 1,
          snapshotDigest: 'def456',
          predicatesPassed: [],
        });

        expect(state.lastSuccessfulUrl).toBe('https://example.com/page2');
        expect(state.lastSuccessfulStep).toBe(1);
      });

      it('should increment length', () => {
        expect(state.length).toBe(0);

        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        expect(state.length).toBe(1);

        state.recordCheckpoint({
          url: 'https://example.com/page2',
          stepIndex: 1,
          snapshotDigest: 'def',
          predicatesPassed: [],
        });

        expect(state.length).toBe(2);
      });

      it('should limit checkpoints to maxCheckpoints', () => {
        const customState = new RecoveryState({ maxCheckpoints: 3 });

        for (let i = 0; i < 5; i++) {
          customState.recordCheckpoint({
            url: `https://example.com/page${i}`,
            stepIndex: i,
            snapshotDigest: `digest${i}`,
            predicatesPassed: [],
          });
        }

        expect(customState.length).toBe(3);
        expect(customState.lastSuccessfulStep).toBe(4);
        expect(customState.lastSuccessfulUrl).toBe('https://example.com/page4');
      });

      it('should handle missing predicatesPassed', () => {
        const checkpoint = state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
        } as Omit<RecoveryCheckpoint, 'timestamp'>);

        expect(checkpoint.predicatesPassed).toEqual([]);
      });
    });

    describe('getRecoveryTarget', () => {
      it('should return most recent checkpoint', () => {
        state.recordCheckpoint({
          url: 'https://example.com/page1',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        state.recordCheckpoint({
          url: 'https://example.com/page2',
          stepIndex: 1,
          snapshotDigest: 'def',
          predicatesPassed: [],
        });

        const target = state.getRecoveryTarget();
        expect(target?.url).toBe('https://example.com/page2');
        expect(target?.stepIndex).toBe(1);
      });
    });

    describe('getCheckpointAtStep', () => {
      beforeEach(() => {
        state.recordCheckpoint({
          url: 'https://example.com/page0',
          stepIndex: 0,
          snapshotDigest: 'digest0',
          predicatesPassed: [],
        });
        state.recordCheckpoint({
          url: 'https://example.com/page1',
          stepIndex: 1,
          snapshotDigest: 'digest1',
          predicatesPassed: [],
        });
        state.recordCheckpoint({
          url: 'https://example.com/page2',
          stepIndex: 2,
          snapshotDigest: 'digest2',
          predicatesPassed: [],
        });
      });

      it('should return checkpoint at specific step', () => {
        const checkpoint = state.getCheckpointAtStep(1);
        expect(checkpoint?.url).toBe('https://example.com/page1');
        expect(checkpoint?.stepIndex).toBe(1);
      });

      it('should return null for non-existent step', () => {
        expect(state.getCheckpointAtStep(5)).toBeNull();
      });
    });

    describe('getCheckpointBeforeStep', () => {
      beforeEach(() => {
        state.recordCheckpoint({
          url: 'https://example.com/page0',
          stepIndex: 0,
          snapshotDigest: 'digest0',
          predicatesPassed: [],
        });
        state.recordCheckpoint({
          url: 'https://example.com/page2',
          stepIndex: 2,
          snapshotDigest: 'digest2',
          predicatesPassed: [],
        });
        state.recordCheckpoint({
          url: 'https://example.com/page4',
          stepIndex: 4,
          snapshotDigest: 'digest4',
          predicatesPassed: [],
        });
      });

      it('should return most recent checkpoint before step', () => {
        const checkpoint = state.getCheckpointBeforeStep(3);
        expect(checkpoint?.stepIndex).toBe(2);
      });

      it('should return null for step 0', () => {
        expect(state.getCheckpointBeforeStep(0)).toBeNull();
      });

      it('should skip steps and find earlier checkpoint', () => {
        const checkpoint = state.getCheckpointBeforeStep(2);
        expect(checkpoint?.stepIndex).toBe(0);
      });
    });

    describe('canRecover', () => {
      it('should return true with checkpoints and attempts remaining', () => {
        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        expect(state.canRecover()).toBe(true);
      });

      it('should return false when all attempts used', () => {
        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        // Use all attempts
        state.consumeRecoveryAttempt();
        state.consumeRecoveryAttempt();

        expect(state.canRecover()).toBe(false);
      });

      it('should return false with no checkpoints', () => {
        expect(state.canRecover()).toBe(false);
      });
    });

    describe('consumeRecoveryAttempt', () => {
      it('should return checkpoint and increment attempts', () => {
        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        const checkpoint = state.consumeRecoveryAttempt();
        expect(checkpoint?.url).toBe('https://example.com');
        expect(state.attemptsUsed).toBe(1);
      });

      it('should set currentRecoveryTarget', () => {
        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        state.consumeRecoveryAttempt();
        expect(state.currentRecoveryTarget?.url).toBe('https://example.com');
      });

      it('should return null when recovery not possible', () => {
        expect(state.consumeRecoveryAttempt()).toBeNull();
      });

      it('should return null when attempts exhausted', () => {
        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        state.consumeRecoveryAttempt();
        state.consumeRecoveryAttempt();

        expect(state.consumeRecoveryAttempt()).toBeNull();
      });
    });

    describe('clearRecoveryTarget', () => {
      it('should clear current recovery target', () => {
        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        state.consumeRecoveryAttempt();
        expect(state.currentRecoveryTarget).not.toBeNull();

        state.clearRecoveryTarget();
        expect(state.currentRecoveryTarget).toBeNull();
      });
    });

    describe('reset', () => {
      it('should reset all state', () => {
        state.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });
        state.consumeRecoveryAttempt();

        state.reset();

        expect(state.length).toBe(0);
        expect(state.attemptsUsed).toBe(0);
        expect(state.currentRecoveryTarget).toBeNull();
        expect(state.lastSuccessfulUrl).toBeNull();
        expect(state.lastSuccessfulStep).toBeNull();
      });
    });

    describe('popCheckpoint', () => {
      it('should remove and return most recent checkpoint', () => {
        state.recordCheckpoint({
          url: 'https://example.com/page1',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });
        state.recordCheckpoint({
          url: 'https://example.com/page2',
          stepIndex: 1,
          snapshotDigest: 'def',
          predicatesPassed: [],
        });

        const popped = state.popCheckpoint();
        expect(popped?.url).toBe('https://example.com/page2');
        expect(state.length).toBe(1);
        expect(state.lastSuccessfulUrl).toBe('https://example.com/page1');
      });

      it('should return null for empty state', () => {
        expect(state.popCheckpoint()).toBeNull();
      });
    });

    describe('custom config', () => {
      it('should respect custom maxRecoveryAttempts', () => {
        const customState = new RecoveryState({ maxRecoveryAttempts: 5 });
        customState.recordCheckpoint({
          url: 'https://example.com',
          stepIndex: 0,
          snapshotDigest: 'abc',
          predicatesPassed: [],
        });

        for (let i = 0; i < 5; i++) {
          expect(customState.canRecover()).toBe(true);
          customState.consumeRecoveryAttempt();
        }

        expect(customState.canRecover()).toBe(false);
      });
    });
  });
});
