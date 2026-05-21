import type { ActionRecord, RepairFailureCategory, RepairHistoryEntry } from './plan-models';
import type { StepwisePlannerResponse } from './prompts';

function formatActionHistory(actionHistory: ActionRecord[]): string {
  if (actionHistory.length === 0) {
    return 'Recent action history:\n- none\n';
  }

  const lines = actionHistory.map(record => {
    const targetStr = record.target ? `(${record.target})` : '';
    const urlStr = record.urlAfter ? ` [URL: ${record.urlAfter}]` : '';
    return `${record.stepNum}. ${record.action}${targetStr} -> ${record.result}${urlStr}`;
  });

  return `Recent action history:\n${lines.join('\n')}\n`;
}

function formatRepairHistory(history: RepairHistoryEntry[]): string {
  if (history.length === 0) {
    return 'Previous repair attempts:\n- none\n';
  }

  const lines = history.map(entry => {
    return `${entry.attempt}. ${entry.failedAction} (${entry.failureCategory}) -> ${entry.reason}`;
  });

  return `Previous repair attempts:\n${lines.join('\n')}\n`;
}

export function buildRepairPlannerPrompt(args: {
  task: string;
  currentUrl: string;
  failedStep: StepwisePlannerResponse;
  failureReason: string;
  failureCategory: RepairFailureCategory;
  actionHistory: ActionRecord[];
  repairHistory: RepairHistoryEntry[];
}): [string, string] {
  const {
    task,
    currentUrl,
    failedStep,
    failureReason,
    failureCategory,
    actionHistory,
    repairHistory,
  } = args;

  const system = `You are a browser automation repair planner.
Return ONLY JSON.
Repair the failed step with a deterministic JSON patch.
Use mode="patch" and a replace_steps array.
Each replace_steps entry must contain:
- "id": the failed step id
- "step": a replacement step object

Prefer a materially different recovery:
- element-not-found: navigate, search, scroll, or choose a different control
- verification-failed: add a wait, stronger verification, or an alternate path
- parse-failure: simplify the action
- auth-or-recovery: recover safely without repeating the same blocked action`;

  const failedStepJson = JSON.stringify(
    {
      id: failedStep.id ?? 1,
      goal: failedStep.goal || failedStep.intent || failedStep.action,
      action: failedStep.action,
      target: failedStep.target,
      intent: failedStep.intent,
      input: failedStep.input,
      fields: failedStep.fields,
      submitText: failedStep.submitText,
      verify: failedStep.verify || [],
      required: failedStep.required !== false,
      optional_substeps: failedStep.optionalSubsteps || [],
    },
    null,
    2
  );

  const user = `/no_think
Task: ${task}
Current URL: ${currentUrl}
Failure classification: ${failureCategory}
Failure reason: ${failureReason}

Failed step:
${failedStepJson}

${formatActionHistory(actionHistory)}
${formatRepairHistory(repairHistory)}
Return JSON patch:
{"mode":"patch","replace_steps":[{"id":${failedStep.id ?? 1},"step":{"id":${failedStep.id ?? 1},"goal":"describe the repair","action":"NAVIGATE","target":"https://example.com","verify":[],"required":true}}]}`;

  return [system, user];
}
