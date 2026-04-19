/**
 * Tests for PlannerExecutorAgent Configuration
 */

import {
  DEFAULT_CONFIG,
  mergeConfig,
  getConfigPreset,
  ConfigPreset,
  type SnapshotEscalationConfig,
} from '../../../src/agents/planner-executor/config';

describe('PlannerExecutorConfig', () => {
  describe('DEFAULT_CONFIG', () => {
    it('should have correct snapshot escalation defaults', () => {
      const snapshot = DEFAULT_CONFIG.snapshot;

      expect(snapshot.enabled).toBe(true);
      // Same defaults as Python SDK - formatContext uses multi-strategy selection
      // to ensure product links are captured even with lower snapshot limits
      expect(snapshot.limitBase).toBe(60);
      expect(snapshot.limitStep).toBe(30);
      expect(snapshot.limitMax).toBe(200);
    });

    it('should have correct scroll-after-escalation defaults', () => {
      const snapshot = DEFAULT_CONFIG.snapshot;

      expect(snapshot.scrollAfterEscalation).toBe(true);
      expect(snapshot.scrollMaxAttempts).toBe(3);
      expect(snapshot.scrollDirections).toEqual(['down', 'up']);
      expect(snapshot.scrollViewportFraction).toBe(0.4);
      expect(snapshot.scrollStabilizeMs).toBe(300);
    });

    it('should have correct retry defaults', () => {
      const retry = DEFAULT_CONFIG.retry;

      expect(retry.verifyTimeoutMs).toBe(10000);
      expect(retry.verifyPollMs).toBe(500);
      expect(retry.verifyMaxAttempts).toBe(4);
      expect(retry.executorRepairAttempts).toBe(2);
      expect(retry.maxReplans).toBe(2);
    });

    it('should have correct stepwise planning defaults', () => {
      const stepwise = DEFAULT_CONFIG.stepwise;

      expect(stepwise.maxSteps).toBe(20);
      expect(stepwise.actionHistoryLimit).toBe(5);
      expect(stepwise.includePageContext).toBe(true);
    });

    it('should have correct LLM token defaults', () => {
      expect(DEFAULT_CONFIG.plannerMaxTokens).toBe(2048);
      expect(DEFAULT_CONFIG.plannerTemperature).toBe(0.0);
      expect(DEFAULT_CONFIG.executorMaxTokens).toBe(96);
      expect(DEFAULT_CONFIG.executorTemperature).toBe(0.0);
    });

    it('should have preStepVerification enabled by default', () => {
      expect(DEFAULT_CONFIG.preStepVerification).toBe(true);
    });
  });

  describe('mergeConfig', () => {
    it('should merge partial config with defaults', () => {
      const partial = {
        verbose: true,
        plannerMaxTokens: 4096,
      };

      const merged = mergeConfig(partial);

      expect(merged.verbose).toBe(true);
      expect(merged.plannerMaxTokens).toBe(4096);
      // Other values should be defaults
      expect(merged.executorMaxTokens).toBe(DEFAULT_CONFIG.executorMaxTokens);
      expect(merged.snapshot.limitBase).toBe(DEFAULT_CONFIG.snapshot.limitBase);
    });

    it('should deep merge nested snapshot config', () => {
      const partial = {
        snapshot: {
          limitBase: 100,
          scrollMaxAttempts: 5,
        },
      };

      const merged = mergeConfig(partial);

      expect(merged.snapshot.limitBase).toBe(100);
      expect(merged.snapshot.scrollMaxAttempts).toBe(5);
      // Other snapshot values should be defaults
      expect(merged.snapshot.limitStep).toBe(DEFAULT_CONFIG.snapshot.limitStep);
      expect(merged.snapshot.scrollAfterEscalation).toBe(
        DEFAULT_CONFIG.snapshot.scrollAfterEscalation
      );
    });

    it('should deep merge nested retry config', () => {
      const partial = {
        retry: {
          verifyTimeoutMs: 20000,
        },
      };

      const merged = mergeConfig(partial);

      expect(merged.retry.verifyTimeoutMs).toBe(20000);
      expect(merged.retry.verifyPollMs).toBe(DEFAULT_CONFIG.retry.verifyPollMs);
    });

    it('should handle empty partial config', () => {
      const merged = mergeConfig({});

      expect(merged).toEqual(DEFAULT_CONFIG);
    });

    it('should allow disabling scroll-after-escalation', () => {
      const partial = {
        snapshot: {
          scrollAfterEscalation: false,
        },
      };

      const merged = mergeConfig(partial);

      expect(merged.snapshot.scrollAfterEscalation).toBe(false);
      expect(merged.snapshot.limitBase).toBe(DEFAULT_CONFIG.snapshot.limitBase);
    });
  });

  describe('getConfigPreset', () => {
    it('should return LOCAL_SMALL_MODEL preset with high token limits', () => {
      const config = getConfigPreset(ConfigPreset.LOCAL_SMALL_MODEL);

      expect(config.plannerMaxTokens).toBe(8192);
      expect(config.executorMaxTokens).toBe(4096);
      expect(config.verbose).toBe(true);
      // Should inherit scroll settings from DEFAULT_CONFIG
      expect(config.snapshot.scrollAfterEscalation).toBe(true);
    });

    it('should return CLOUD_HIGH_QUALITY preset', () => {
      const config = getConfigPreset(ConfigPreset.CLOUD_HIGH_QUALITY);

      expect(config.plannerMaxTokens).toBe(2048);
      expect(config.executorMaxTokens).toBe(128);
      expect(config.verbose).toBe(false);
    });

    it('should return FAST_ITERATION preset with minimal retries', () => {
      const config = getConfigPreset(ConfigPreset.FAST_ITERATION);

      expect(config.retry.verifyMaxAttempts).toBe(2);
      expect(config.retry.executorRepairAttempts).toBe(1);
      expect(config.plannerMaxTokens).toBe(1024);
    });

    it('should return PRODUCTION preset with more retries', () => {
      const config = getConfigPreset(ConfigPreset.PRODUCTION);

      expect(config.retry.verifyMaxAttempts).toBe(8);
      expect(config.retry.executorRepairAttempts).toBe(3);
      expect(config.retry.verifyTimeoutMs).toBe(20000);
    });

    it('should return DEFAULT preset', () => {
      const config = getConfigPreset(ConfigPreset.DEFAULT);

      expect(config).toEqual(DEFAULT_CONFIG);
    });

    it('should handle string preset names', () => {
      const config = getConfigPreset('local_small');

      expect(config.plannerMaxTokens).toBe(8192);
    });

    it('should return default for unknown preset', () => {
      const config = getConfigPreset('unknown_preset');

      expect(config).toEqual(DEFAULT_CONFIG);
    });
  });

  describe('SnapshotEscalationConfig scroll parameters', () => {
    it('should support custom scroll directions', () => {
      const partial = {
        snapshot: {
          scrollDirections: ['up'] as Array<'up' | 'down'>,
        },
      };

      const merged = mergeConfig(partial);

      expect(merged.snapshot.scrollDirections).toEqual(['up']);
    });

    it('should support custom viewport fraction', () => {
      const partial = {
        snapshot: {
          scrollViewportFraction: 0.5,
        },
      };

      const merged = mergeConfig(partial);

      expect(merged.snapshot.scrollViewportFraction).toBe(0.5);
    });

    it('should support custom stabilize delay', () => {
      const partial = {
        snapshot: {
          scrollStabilizeMs: 500,
        },
      };

      const merged = mergeConfig(partial);

      expect(merged.snapshot.scrollStabilizeMs).toBe(500);
    });
  });
});
