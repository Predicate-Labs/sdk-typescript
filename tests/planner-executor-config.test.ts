import {
  ConfigPreset,
  getConfigPreset,
  mergeConfig,
  DEFAULT_CONFIG,
  detectProvider,
  createProvider,
  resolveConfig,
  createPlannerExecutorAgentProviders,
} from '../src/agents/planner-executor';
import { OllamaProvider, OpenAIProvider, AnthropicProvider } from '../src/llm-provider';

describe('ConfigPreset', () => {
  it('should have all expected preset values', () => {
    expect(ConfigPreset.DEFAULT).toBe('default');
    expect(ConfigPreset.LOCAL_SMALL_MODEL).toBe('local_small');
    expect(ConfigPreset.CLOUD_HIGH_QUALITY).toBe('cloud_high');
    expect(ConfigPreset.FAST_ITERATION).toBe('fast');
    expect(ConfigPreset.PRODUCTION).toBe('production');
  });
});

describe('getConfigPreset', () => {
  it('should return default config for DEFAULT preset', () => {
    const config = getConfigPreset(ConfigPreset.DEFAULT);
    expect(config).toEqual(DEFAULT_CONFIG);
  });

  it('should return optimized config for LOCAL_SMALL_MODEL', () => {
    const config = getConfigPreset(ConfigPreset.LOCAL_SMALL_MODEL);
    // HIGH token limits for local models like Qwen3 that include reasoning in output
    expect(config.plannerMaxTokens).toBe(8192);
    expect(config.executorMaxTokens).toBe(4096);
    expect(config.retry.verifyTimeoutMs).toBe(15000);
    expect(config.retry.verifyMaxAttempts).toBe(6);
    expect(config.verbose).toBe(true);
  });

  it('should return optimized config for CLOUD_HIGH_QUALITY', () => {
    const config = getConfigPreset(ConfigPreset.CLOUD_HIGH_QUALITY);
    expect(config.plannerMaxTokens).toBe(2048);
    expect(config.executorMaxTokens).toBe(128);
    expect(config.retry.verifyTimeoutMs).toBe(10000);
    expect(config.verbose).toBe(false);
  });

  it('should return fast iteration config', () => {
    const config = getConfigPreset(ConfigPreset.FAST_ITERATION);
    expect(config.retry.verifyMaxAttempts).toBe(2);
    expect(config.retry.maxReplans).toBe(1);
    expect(config.verbose).toBe(true);
  });

  it('should return production config', () => {
    const config = getConfigPreset(ConfigPreset.PRODUCTION);
    expect(config.retry.verifyMaxAttempts).toBe(8);
    expect(config.retry.verifyTimeoutMs).toBe(20000);
    expect(config.verbose).toBe(false);
  });

  it('should accept string preset names', () => {
    const config = getConfigPreset('local_small');
    expect(config.plannerMaxTokens).toBe(8192);
  });
});

describe('mergeConfig', () => {
  it('should merge partial config with defaults', () => {
    const config = mergeConfig({ verbose: true });
    expect(config.verbose).toBe(true);
    expect(config.plannerMaxTokens).toBe(DEFAULT_CONFIG.plannerMaxTokens);
  });

  it('should merge nested config objects', () => {
    const config = mergeConfig({
      retry: { verifyTimeoutMs: 5000 },
    });
    expect(config.retry.verifyTimeoutMs).toBe(5000);
    expect(config.retry.verifyMaxAttempts).toBe(DEFAULT_CONFIG.retry.verifyMaxAttempts);
  });
});

describe('detectProvider', () => {
  it('should detect OpenAI for GPT models', () => {
    expect(detectProvider('gpt-4o')).toBe('openai');
    expect(detectProvider('gpt-4-turbo')).toBe('openai');
    expect(detectProvider('gpt-4o-mini')).toBe('openai');
    expect(detectProvider('GPT-4o')).toBe('openai');
  });

  it('should detect OpenAI for o1/o3/o4 models', () => {
    expect(detectProvider('o1-preview')).toBe('openai');
    expect(detectProvider('o1-mini')).toBe('openai');
    expect(detectProvider('o3-mini')).toBe('openai');
  });

  it('should detect Anthropic for Claude models', () => {
    expect(detectProvider('claude-3-opus-20240229')).toBe('anthropic');
    expect(detectProvider('claude-3-5-sonnet-20241022')).toBe('anthropic');
    expect(detectProvider('Claude-3-Opus')).toBe('anthropic');
  });

  it('should detect Ollama for common local models', () => {
    expect(detectProvider('qwen3:8b')).toBe('ollama');
    expect(detectProvider('llama3:8b')).toBe('ollama');
    expect(detectProvider('phi3:mini')).toBe('ollama');
    expect(detectProvider('mistral:7b')).toBe('ollama');
    expect(detectProvider('gemma:2b')).toBe('ollama');
    expect(detectProvider('deepseek:6.7b')).toBe('ollama');
    expect(detectProvider('codellama:7b')).toBe('ollama');
  });

  it('should detect Ollama for model:tag format', () => {
    expect(detectProvider('custom-model:latest')).toBe('ollama');
    expect(detectProvider('my-finetuned:v2')).toBe('ollama');
  });

  it('should default to Ollama for unknown models', () => {
    expect(detectProvider('some-unknown-model')).toBe('ollama');
  });
});

describe('createProvider', () => {
  it('should create OllamaProvider for ollama', () => {
    const provider = createProvider('qwen3:8b', 'ollama', {});
    expect(provider).toBeInstanceOf(OllamaProvider);
    expect(provider.modelName).toBe('qwen3:8b');
  });

  it('should create OllamaProvider with custom base URL', () => {
    const provider = createProvider('llama3:8b', 'ollama', {
      ollamaBaseUrl: 'http://192.168.1.100:11434',
    }) as OllamaProvider;
    expect(provider.ollamaBaseUrl).toBe('http://192.168.1.100:11434');
  });

  it('should create OpenAIProvider for openai', () => {
    const provider = createProvider('gpt-4o', 'openai', {
      openaiApiKey: 'test-key',
    });
    expect(provider).toBeInstanceOf(OpenAIProvider);
    expect(provider.modelName).toBe('gpt-4o');
  });

  it('should create AnthropicProvider for anthropic', () => {
    const provider = createProvider('claude-3-opus-20240229', 'anthropic', {
      anthropicApiKey: 'test-key',
    });
    expect(provider).toBeInstanceOf(AnthropicProvider);
    expect(provider.modelName).toBe('claude-3-opus-20240229');
  });

  it('should auto-detect provider', () => {
    const provider = createProvider('qwen3:8b', 'auto', {});
    expect(provider).toBeInstanceOf(OllamaProvider);
  });

  it('should throw for unknown provider', () => {
    expect(() => {
      createProvider('test', 'invalid' as any, {});
    }).toThrow(/Unknown provider/);
  });
});

describe('resolveConfig', () => {
  it('should return default config when undefined', () => {
    const config = resolveConfig();
    expect(config).toEqual(DEFAULT_CONFIG);
  });

  it('should resolve string preset', () => {
    const config = resolveConfig('local_small');
    expect(config.plannerMaxTokens).toBe(8192);
  });

  it('should resolve ConfigPreset enum', () => {
    const config = resolveConfig(ConfigPreset.PRODUCTION);
    expect(config.retry.verifyMaxAttempts).toBe(8);
  });

  it('should merge partial config', () => {
    const config = resolveConfig({ verbose: true });
    expect(config.verbose).toBe(true);
    expect(config.plannerMaxTokens).toBe(DEFAULT_CONFIG.plannerMaxTokens);
  });
});

describe('createPlannerExecutorAgentProviders', () => {
  it('should create providers with minimal config', async () => {
    const result = await createPlannerExecutorAgentProviders({
      plannerModel: 'qwen3:8b',
      executorModel: 'qwen3:4b',
    });

    expect(result.planner).toBeInstanceOf(OllamaProvider);
    expect(result.executor).toBeInstanceOf(OllamaProvider);
    expect(result.planner.modelName).toBe('qwen3:8b');
    expect(result.executor.modelName).toBe('qwen3:4b');
    expect(result.config).toEqual(DEFAULT_CONFIG);
    expect(result.tracer).toBeUndefined();
  });

  it('should use config preset', async () => {
    const result = await createPlannerExecutorAgentProviders({
      plannerModel: 'qwen3:8b',
      executorModel: 'qwen3:4b',
      config: ConfigPreset.LOCAL_SMALL_MODEL,
    });

    expect(result.config.plannerMaxTokens).toBe(8192);
    expect(result.config.verbose).toBe(true);
  });

  it('should support mixed providers', async () => {
    const result = await createPlannerExecutorAgentProviders({
      plannerModel: 'gpt-4o',
      plannerProvider: 'openai',
      executorModel: 'qwen3:4b',
      executorProvider: 'ollama',
      openaiApiKey: 'test-key',
    });

    expect(result.planner).toBeInstanceOf(OpenAIProvider);
    expect(result.executor).toBeInstanceOf(OllamaProvider);
  });
});
