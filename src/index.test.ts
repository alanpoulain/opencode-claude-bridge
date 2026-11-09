/**
 * Unit tests for opencode-claude-bridge plugin logic.
 * Run with: npm test (builds then runs node --test).
 *
 * These tests exercise the actual production modules — the SSE processor
 * is imported from ./stream, the argument translator from ./claude-tools.
 * No local re-implementations.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  translateToolArgsJsonString,
  translateArgsOpencodeToClaude,
  AGENT_TYPE_CLAUDE_TO_OPENCODE,
  AGENT_TYPE_OPENCODE_TO_CLAUDE,
  shouldUseClaudeToolSchemas,
} from "./claude-tools.js";
import { extractOAuthErrorDetail, parseTokenResponse } from "./oauth.js";
import {
  buildKeychainWriteCommand,
  selectClaudeCredentials,
  updateCredentialBlob,
  writeBackCredentialsFile,
} from "./keychain.js";
import type { ClaudeCliTokens, CredentialSource } from "./keychain.js";
import {
  default as pluginModule,
  applyThinkingConfig,
  deriveModelDisplayName,
  getClaudeToolsForActiveOpenCodeTools,
  getInboundToolNameMapForActiveOpenCodeTools,
  mapOutboundToolName,
  injectClaudePrefix,
  mergeSystemBlocks,
  refreshCredentialLayered,
  resolveOAuthAccessToken,
  rewriteSystemBlocksForModel,
  shouldInjectClaudeTools,
  stripAssistantPrefillForClaude,
  stripSystemCacheControl,
} from "./index.js";
import { createSseProcessor, parseSseEvent, buildSseEvent } from "./stream.js";
import { fetchCliVersion, parseDistTagsVersion } from "./cli-version.js";
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CLAUDE_PREFIX = "You are a Claude agent, built on Anthropic's Claude Agent SDK.";

// ── Helpers ───────────────────────────────────────────────────────────────────

function transformBody(bodyStr: string): Record<string, unknown> {
  const parsed = JSON.parse(bodyStr);
  applyThinkingConfig(parsed);
  return parsed;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("thinking injection", () => {
  it("does NOT inject thinking for claude-sonnet-4-5", () => {
    const out = transformBody(JSON.stringify({ model: "claude-sonnet-4-5", messages: [] }));
    assert.equal(out.thinking, undefined, "sonnet-4-5 must not get thinking injected");
  });

  it("does NOT inject thinking for claude-haiku-4-5-20251001", () => {
    const out = transformBody(JSON.stringify({ model: "claude-haiku-4-5-20251001", messages: [] }));
    assert.equal(out.thinking, undefined, "haiku must not get thinking injected");
  });

  it("does NOT inject thinking for an unknown model", () => {
    const out = transformBody(JSON.stringify({ model: "gpt-4o", messages: [] }));
    assert.equal(out.thinking, undefined, "unknown model must not get thinking injected");
  });

  for (const model of [
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-20250514",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-haiku-5-5",
  ]) {
    it(`injects adaptive thinking with summarized display for ${model}`, () => {
      const out = transformBody(JSON.stringify({ model, messages: [] }));
      assert.deepEqual(out.thinking, { type: "adaptive", display: "summarized" });
    });
  }

  it("does not overwrite an existing thinking block", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-sonnet-4-6", messages: [], thinking: { type: "disabled" } }),
    );
    assert.deepEqual(out.thinking, { type: "disabled" });
  });

  it("adds summarized display to incoming thinking without display", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-opus-5-5", messages: [], thinking: { type: "adaptive", block_binding: "x" } }),
    );
    assert.deepEqual(out.thinking, { type: "adaptive", block_binding: "x", display: "summarized" });
  });

  it("keeps an explicit display value", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-opus-5-5", messages: [], thinking: { type: "adaptive", display: "omitted" } }),
    );
    assert.deepEqual(out.thinking, { type: "adaptive", display: "omitted" });
  });
});

describe("temperature coercion", () => {
  it("forces temperature to 1 when adaptive thinking is injected", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-sonnet-4-6", messages: [], temperature: 0.7 }),
    );
    assert.equal(out.temperature, 1);
  });

  it("forces temperature to 1 when thinking:enabled is already set", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-sonnet-4-6", messages: [], thinking: { type: "enabled" }, temperature: 0 }),
    );
    assert.equal(out.temperature, 1);
  });

  it("does not touch temperature when thinking is not injected (sonnet-4-5)", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-sonnet-4-5", messages: [], temperature: 0.7 }),
    );
    assert.equal(out.temperature, 0.7);
  });

  it("does not touch temperature when it is already 1", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-sonnet-4-6", messages: [], temperature: 1 }),
    );
    assert.equal(out.temperature, 1);
  });

  it("does not touch temperature when it is absent", () => {
    const out = transformBody(
      JSON.stringify({ model: "claude-sonnet-4-6", messages: [] }),
    );
    assert.equal(out.temperature, undefined);
  });
});

describe("OAuth error parsing", () => {
  it("extracts nested error messages from token responses", () => {
    const detail = extractOAuthErrorDetail(JSON.stringify({
      error: {
        type: "invalid_request_error",
        message: "invalid_grant: refresh token not found or invalid",
      },
    }));
    assert.equal(detail, "invalid_grant: refresh token not found or invalid");
  });

  it("falls back to raw response bodies when parsing fails", () => {
    const detail = extractOAuthErrorDetail("plain failure body");
    assert.equal(detail, "plain failure body");
  });
});

describe("Claude assistant prefill stripping", () => {
  it("strips the synthetic continue prefill so Anthropic requests end with a user message", () => {
    const out = stripAssistantPrefillForClaude([
      { role: "user", content: [{ type: "text", text: "hello" }] },
      { role: "assistant", content: "Continue with your tasks." },
    ]);
    assert.deepEqual(out, [
      { role: "user", content: [{ type: "text", text: "hello" }] },
    ]);
  });

  it("keeps non-synthetic assistant messages intact", () => {
    const input = [
      { role: "user", content: [{ type: "text", text: "hello" }] },
      { role: "assistant", content: "Real assistant content" },
    ];
    assert.deepEqual(stripAssistantPrefillForClaude(input), input);
  });
});

describe("system cache control", () => {
  it("strips cache_control from system blocks", () => {
    assert.deepEqual(stripSystemCacheControl([
      { type: "text", text: "a", cache_control: { type: "ephemeral" } },
      { type: "text", text: "b" },
    ]), [
      { type: "text", text: "a" },
      { type: "text", text: "b" },
    ]);
  });
});

describe("tool schema selection", () => {
  it("uses Claude schemas for native Anthropic requests", () => {
    assert.equal(shouldUseClaudeToolSchemas({
      model: "claude-sonnet-4-6",
      requestUrl: "https://api.anthropic.com/v1/messages",
    }), true);
  });

  it("uses Claude schemas for Claude-family models on OpenRouter", () => {
    assert.equal(shouldUseClaudeToolSchemas({
      model: "anthropic/claude-3.7-sonnet",
      requestUrl: "https://openrouter.ai/api/v1/chat/completions",
    }), true);
  });

  it("keeps default schemas for non-Claude models on OpenRouter", () => {
    assert.equal(shouldUseClaudeToolSchemas({
      model: "x-ai/grok-4-fast",
      requestUrl: "https://openrouter.ai/api/v1/chat/completions",
    }), false);
  });

  it("keeps default schemas for non-Claude targets", () => {
    assert.equal(shouldUseClaudeToolSchemas({
      model: "gpt-4o-mini",
      requestUrl: "https://api.openai.com/v1/responses",
    }), false);
  });

  it("injects Claude tool schemas only when active tools are present", () => {
    assert.equal(shouldInjectClaudeTools({
      model: "claude-sonnet-4-6",
      requestUrl: "https://api.anthropic.com/v1/messages",
      tools: [{ name: "bash" }],
    }), true);
  });

  it("does not inject Claude tool schemas for tool-less summary or compaction requests", () => {
    assert.equal(shouldInjectClaudeTools({
      model: "claude-sonnet-4-6",
      requestUrl: "https://api.anthropic.com/v1/messages",
      tools: [],
    }), false);
    assert.equal(shouldInjectClaudeTools({
      model: "claude-sonnet-4-6",
      requestUrl: "https://api.anthropic.com/v1/messages",
      tools: undefined,
    }), false);
  });

  it("filters Claude tool schemas to tools active in the OpenCode request", () => {
    const out = getClaudeToolsForActiveOpenCodeTools([
      { name: "bash" },
      { name: "webfetch" },
      { name: "question" },
    ]).map((tool) => tool.name).sort();
    assert.deepEqual(out, ["AskUserQuestion", "Bash", "WebFetch"]);
  });

  it("does not advertise WebSearch when only a custom websearch_cited tool is active", () => {
    const out = getClaudeToolsForActiveOpenCodeTools([
      { name: "webfetch" },
      {
        name: "websearch_cited",
        description: "Search the web with citations",
        input_schema: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"],
        },
      },
    ]).map((tool) => tool.name).sort();
    assert.deepEqual(out, ["WebFetch", "websearch_cited"]);
  });

  it("preserves active MCP and custom OpenCode tools", () => {
    const customTool = {
      name: "custom_mcp_query_record",
      description: "Get a record by ID",
      input_schema: {
        type: "object",
        properties: { recordId: { type: "string" } },
        required: ["recordId"],
      },
    };
    const out = getClaudeToolsForActiveOpenCodeTools([
      { name: "bash", description: "OpenCode bash", input_schema: { type: "object" } },
      customTool,
    ]);

    const bashTool = out.find((tool) => tool.name === "Bash");
    const preservedTool = out.find((tool) => tool.name === customTool.name);

    assert.ok(bashTool);
    assert.notEqual(bashTool.description, "OpenCode bash");
    assert.deepEqual(preservedTool, customTool);
  });

  it("preserves MCP tools that have no description", () => {
    const out = getClaudeToolsForActiveOpenCodeTools([
      {
        name: "custom_mcp_get_timestamp",
        input_schema: { type: "object", properties: {} },
      },
    ]);

    assert.deepEqual(out, [
      {
        name: "custom_mcp_get_timestamp",
        description: "",
        input_schema: { type: "object", properties: {} },
      },
    ]);
  });

  it("preserves MCP tools whose names look like prefixed core tools", () => {
    const mcpTool = {
      name: "mcp_bash",
      description: "Run a command through an MCP server",
      input_schema: {
        type: "object",
        properties: { command: { type: "string" } },
        required: ["command"],
      },
    };
    const out = getClaudeToolsForActiveOpenCodeTools([mcpTool]);

    assert.deepEqual(out, [mcpTool]);
  });

  it("preserves custom tools whose names collide with Claude core tool names", () => {
    const customTool = {
      name: "Bash",
      description: "Custom Bash-like MCP tool",
      input_schema: {
        type: "object",
        properties: { script: { type: "string" } },
        required: ["script"],
      },
    };
    const out = getClaudeToolsForActiveOpenCodeTools([customTool]);

    assert.deepEqual(out, [customTool]);
  });

  it("maps inbound Claude core names only for active OpenCode core tools", () => {
    assert.deepEqual(getInboundToolNameMapForActiveOpenCodeTools([
      { name: "bash" },
      { name: "mcp_bash", input_schema: { type: "object" } },
    ]), { Bash: "bash" });
    assert.deepEqual(getInboundToolNameMapForActiveOpenCodeTools([
      { name: "Bash", input_schema: { type: "object" } },
    ]), {});
  });

  it("maps outbound OpenCode core history but preserves MCP-prefixed names", () => {
    assert.equal(mapOutboundToolName("bash"), "Bash");
    assert.equal(mapOutboundToolName("mcp_bash"), "mcp_bash");
  });

  it("does not advertise AskUserQuestion when OpenCode did not enable question", () => {
    const out = getClaudeToolsForActiveOpenCodeTools([
      { name: "bash" },
      { name: "read" },
    ]).map((tool) => tool.name).sort();
    assert.deepEqual(out, ["Bash", "Read"]);
  });
});

// ── translateToolArgsJsonString: argument translation on parsed JSON ─────────

describe("translateToolArgsJsonString", () => {
  function translated(toolName: string, args: Record<string, unknown>): any {
    return JSON.parse(translateToolArgsJsonString(JSON.stringify(args), toolName));
  }

  it("renames file_path → path for Read", () => {
    const out = translated("Read", { file_path: "/tmp/x.txt" });
    assert.deepEqual(out, { path: "/tmp/x.txt" });
  });

  it("renames all Edit params", () => {
    const out = translated("Edit", {
      file_path: "/f.ts",
      old_string: "foo",
      new_string: "bar",
      replace_all: true,
    });
    assert.deepEqual(out, {
      path: "/f.ts",
      oldString: "foo",
      newString: "bar",
      replaceAll: true,
    });
  });

  it("renames glob → include for Grep and preserves other keys", () => {
    const out = translated("Grep", { pattern: "hello", glob: "*.ts", path: "/src" });
    assert.deepEqual(out, { pattern: "hello", include: "*.ts", path: "/src" });
  });

  it("renames skill → id for Skill and strips args (OpenCode skill has no args param)", () => {
    const out = translated("Skill", { skill: "commit", args: "-m fix" });
    assert.deepEqual(out, { id: "commit" });
  });

  it("drops activeForm INSIDE TodoWrite todos[]", () => {
    const out = translated("TodoWrite", {
      todos: [
        { content: "fix bug", status: "pending", activeForm: "Running tests" },
        { content: "ship it", status: "completed", activeForm: "Shipping" },
      ],
    });
    assert.deepEqual(out, {
      todos: [
        { content: "fix bug", status: "pending" },
        { content: "ship it", status: "completed" },
      ],
    });
  });

  it("does NOT rewrite activeForm when it appears only inside a string value (Linus case)", () => {
    // A TodoWrite whose content text happens to contain the literal token
    // "activeForm" must not have that substring corrupted.
    const content = 'Update the activeForm handler in foo.ts';
    const out = translated("TodoWrite", {
      todos: [{ content, status: "pending", activeForm: "Editing foo.ts" }],
    });
    // content preserved verbatim
    assert.equal(out.todos[0].content, content);
    // activeForm key dropped
    assert.equal(out.todos[0].activeForm, undefined);
  });

  it("does NOT rewrite file_path when it appears inside a Bash command string", () => {
    // Bash commands have no translation — the string value must survive untouched.
    const command = 'cat heredoc <<EOF\n{"file_path": "inside/string"}\nEOF';
    const out = translated("Bash", { command });
    assert.deepEqual(out, { command });
  });

  it("maps Agent subagent_type values", () => {
    assert.equal(translated("Agent", { subagent_type: "general-purpose" }).subagent_type, "general");
    assert.equal(translated("Agent", { subagent_type: "Explore" }).subagent_type, "explore");
    assert.equal(translated("Agent", { subagent_type: "Plan" }).subagent_type, "plan");
    assert.equal(translated("Agent", { subagent_type: "statusline-setup" }).subagent_type, "build");
  });

  it("leaves an unknown Agent subagent_type untouched", () => {
    assert.equal(translated("Agent", { subagent_type: "custom-agent" }).subagent_type, "custom-agent");
  });

  it("strips prompt param for WebFetch and injects default format", () => {
    // Claude's WebFetch has no "format" field; OpenCode's webfetch needs one.
    // We default to "markdown" so the OpenCode tool receives a valid payload.
    const out = translated("WebFetch", { url: "https://example.com", prompt: "Extract title" });
    assert.deepEqual(out, { url: "https://example.com", format: "markdown" });
  });

  it("respects explicit WebFetch format if already set", () => {
    const out = translated("WebFetch", { url: "https://example.com", format: "text" });
    assert.deepEqual(out, { url: "https://example.com", format: "text" });
  });

  it("returns input unchanged for malformed JSON", () => {
    const malformed = "{not valid json";
    assert.equal(translateToolArgsJsonString(malformed, "Read"), malformed);
  });

  it("returns input unchanged for non-object JSON (array or primitive)", () => {
    assert.equal(translateToolArgsJsonString("[1,2,3]", "Read"), "[1,2,3]");
    assert.equal(translateToolArgsJsonString('"hello"', "Read"), '"hello"');
    assert.equal(translateToolArgsJsonString("null", "Read"), "null");
  });

  it("passes through unknown tool names without modification", () => {
    const out = translated("TotallyMadeUpTool", { foo: "bar", file_path: "/x" });
    assert.deepEqual(out, { foo: "bar", file_path: "/x" });
  });

  // Field stripping — Claude sends fields OpenCode's Zod schemas don't accept
  it("strips Claude-only Agent fields (model, run_in_background, isolation)", () => {
    const out = translated("Agent", {
      description: "d",
      prompt: "p",
      subagent_type: "general-purpose",
      model: "opus",
      run_in_background: true,
      isolation: "worktree",
    });
    assert.deepEqual(out, {
      description: "d",
      prompt: "p",
      subagent_type: "general",
    });
  });

  it("defaults Agent subagent_type to 'general' when missing (OpenCode requires it)", () => {
    const out = translated("Agent", { description: "d", prompt: "p" });
    assert.equal(out.subagent_type, "general");
  });

  it("strips Claude-only Bash fields (run_in_background, dangerouslyDisableSandbox)", () => {
    const out = translated("Bash", {
      command: "ls",
      run_in_background: true,
      dangerouslyDisableSandbox: true,
    });
    assert.deepEqual(out, { command: "ls" });
  });

  it("clamps Read offset 0 → 1 (OpenCode offsets are 1-based)", () => {
    const out = translated("Read", { file_path: "/x.txt", offset: 0, limit: 10 });
    assert.deepEqual(out, { path: "/x.txt", offset: 1, limit: 10 });
  });

  it("strips Claude-only Read field (pages)", () => {
    const out = translated("Read", { file_path: "/x.pdf", pages: "1-5" });
    assert.deepEqual(out, { path: "/x.pdf" });
  });

  it("strips Claude-only Grep fields", () => {
    const out = translated("Grep", {
      pattern: "foo",
      glob: "*.ts",
      output_mode: "content",
      "-B": 2,
      "-A": 3,
      "-C": 5,
      context: 1,
      "-n": true,
      "-i": true,
      type: "js",
      head_limit: 10,
      offset: 5,
      multiline: false,
    });
    assert.deepEqual(out, { pattern: "foo", include: "*.ts" });
  });

  it("translates AskUserQuestion multiSelect → multiple per question", () => {
    const out = translated("AskUserQuestion", {
      questions: [
        { question: "Which?", options: ["a", "b"], multiSelect: true },
        { question: "Either?", options: ["y", "n"], multiSelect: false },
      ],
    });
    assert.deepEqual(out, {
      questions: [
        { question: "Which?", options: ["a", "b"], multiple: true },
        { question: "Either?", options: ["y", "n"], multiple: false },
      ],
    });
  });

  it("parses stringified AskUserQuestion questions and translates multiSelect", () => {
    const out = translated("AskUserQuestion", {
      questions: JSON.stringify([
        {
          header: "Slack secret fix",
          question: "SLACK_WEBHOOK_URL is missing. How do you want to handle it?",
          options: [
            { label: "Set manually now", description: "Paste the webhook URL now." },
            { label: "IaC it properly now", description: "Wire the secret through IaC." },
          ],
          multiSelect: false,
        },
      ]),
    });
    assert.deepEqual(out, {
      questions: [
        {
          header: "Slack secret fix",
          question: "SLACK_WEBHOOK_URL is missing. How do you want to handle it?",
          options: [
            { label: "Set manually now", description: "Paste the webhook URL now." },
            { label: "IaC it properly now", description: "Wire the secret through IaC." },
          ],
          multiple: false,
        },
      ],
    });
  });
});

// ── translateArgsOpencodeToClaude: outbound translation on parsed objects ───

describe("translateArgsOpencodeToClaude", () => {
  it("renames camelCase keys to snake_case for Edit", () => {
    const out = translateArgsOpencodeToClaude("Edit", {
      path: "/f.ts",
      oldString: "a",
      newString: "b",
      replaceAll: true,
    });
    assert.deepEqual(out, {
      file_path: "/f.ts",
      old_string: "a",
      new_string: "b",
      replace_all: true,
    });
  });

  it("renames path → file_path for Read and Write", () => {
    assert.deepEqual(translateArgsOpencodeToClaude("Read", { path: "/r.ts", offset: 5 }), {
      file_path: "/r.ts",
      offset: 5,
    });
    assert.deepEqual(translateArgsOpencodeToClaude("Write", { path: "/w.ts", content: "x" }), {
      file_path: "/w.ts",
      content: "x",
    });
  });

  it("renames id → skill for Skill", () => {
    assert.deepEqual(translateArgsOpencodeToClaude("Skill", { id: "commit" }), { skill: "commit" });
  });

  it("keeps path unchanged for Grep", () => {
    assert.deepEqual(translateArgsOpencodeToClaude("Grep", { pattern: "x", path: "/src" }), {
      pattern: "x",
      path: "/src",
    });
  });

  it("maps OpenCode subagent_type values back to Claude", () => {
    assert.equal(translateArgsOpencodeToClaude("Agent", { subagent_type: "general" }).subagent_type, "general-purpose");
    assert.equal(translateArgsOpencodeToClaude("Agent", { subagent_type: "build" }).subagent_type, "general-purpose");
    assert.equal(translateArgsOpencodeToClaude("Agent", { subagent_type: "explore" }).subagent_type, "Explore");
    assert.equal(translateArgsOpencodeToClaude("Agent", { subagent_type: "plan" }).subagent_type, "Plan");
  });

  it("strips Agent OpenCode-only fields (task_id, command)", () => {
    const out = translateArgsOpencodeToClaude("Agent", {
      description: "d",
      prompt: "p",
      subagent_type: "general",
      task_id: "t_1",
      command: "go",
    });
    assert.deepEqual(out, {
      description: "d",
      prompt: "p",
      subagent_type: "general-purpose",
    });
  });

  it("translates AskUserQuestion multiple → multiSelect per question", () => {
    const out = translateArgsOpencodeToClaude("AskUserQuestion", {
      questions: [
        { question: "Which?", options: ["a", "b"], multiple: true },
        { question: "Either?", options: ["y", "n"], multiple: false },
      ],
    });
    assert.deepEqual(out, {
      questions: [
        { question: "Which?", options: ["a", "b"], multiSelect: true },
        { question: "Either?", options: ["y", "n"], multiSelect: false },
      ],
    });
  });

  it("synthesizes a WebFetch prompt from format (markdown)", () => {
    const out = translateArgsOpencodeToClaude("WebFetch", {
      url: "https://example.com",
      format: "markdown",
    });
    assert.deepEqual(out, {
      url: "https://example.com",
      prompt: "Fetch this URL and return the content as markdown.",
    });
  });

  it("synthesizes a WebFetch prompt from format (text)", () => {
    const out = translateArgsOpencodeToClaude("WebFetch", {
      url: "https://example.com",
      format: "text",
    });
    assert.equal(out.prompt, "Fetch this URL and return the content as plain text.");
  });

  it("synthesizes a WebFetch prompt from format (html)", () => {
    const out = translateArgsOpencodeToClaude("WebFetch", {
      url: "https://example.com",
      format: "html",
    });
    assert.equal(out.prompt, "Fetch this URL and return the raw HTML.");
  });

  it("strips WebFetch timeout (OpenCode-only)", () => {
    const out = translateArgsOpencodeToClaude("WebFetch", {
      url: "https://example.com",
      format: "markdown",
      timeout: 30,
    });
    assert.equal(out.timeout, undefined);
  });

  it("synthesizes activeForm from content in TodoWrite and collapses cancelled → completed", () => {
    const out = translateArgsOpencodeToClaude("TodoWrite", {
      todos: [
        { content: "a", status: "pending" },
        { content: "b", status: "cancelled" },
      ],
    });
    assert.deepEqual(out, {
      todos: [
        { content: "a", status: "pending", activeForm: "a" },
        { content: "b", status: "completed", activeForm: "b" },
      ],
    });
  });

  it("maps plan_enter → EnterPlanMode with empty args (outbound)", () => {
    const out = translateArgsOpencodeToClaude("EnterPlanMode", {});
    assert.deepEqual(out, {});
  });

  it("maps plan_exit → ExitPlanMode preserving allowedPrompts (outbound)", () => {
    const out = translateArgsOpencodeToClaude("ExitPlanMode", {
      allowedPrompts: [{ tool: "Bash", prompt: "run tests" }],
    });
    assert.deepEqual(out, { allowedPrompts: [{ tool: "Bash", prompt: "run tests" }] });
  });

  it("is the inverse of translateToolArgsJsonString for the Edit round trip", () => {
    // Round-trip: Claude inbound → OpenCode → Claude outbound should be identity
    // for keys that have a bidirectional mapping.
    const claudeArgs = { file_path: "/f.ts", old_string: "a", new_string: "b" };
    const opencodeArgs = JSON.parse(translateToolArgsJsonString(JSON.stringify(claudeArgs), "Edit"));
    const roundTripped = translateArgsOpencodeToClaude("Edit", opencodeArgs);
    assert.deepEqual(roundTripped, claudeArgs);
  });
});

// ── Agent type map integrity ─────────────────────────────────────────────────

describe("Agent type maps", () => {
  it("CLAUDE_TO_OPENCODE covers Claude's subagent_type values", () => {
    assert.equal(AGENT_TYPE_CLAUDE_TO_OPENCODE["general-purpose"], "general");
    assert.equal(AGENT_TYPE_CLAUDE_TO_OPENCODE["Explore"], "explore");
    assert.equal(AGENT_TYPE_CLAUDE_TO_OPENCODE["Plan"], "plan");
    assert.equal(AGENT_TYPE_CLAUDE_TO_OPENCODE["statusline-setup"], "build");
  });

  it("OPENCODE_TO_CLAUDE covers OpenCode's built-in agents", () => {
    assert.equal(AGENT_TYPE_OPENCODE_TO_CLAUDE["general"], "general-purpose");
    assert.equal(AGENT_TYPE_OPENCODE_TO_CLAUDE["build"], "general-purpose");
    assert.equal(AGENT_TYPE_OPENCODE_TO_CLAUDE["explore"], "Explore");
    assert.equal(AGENT_TYPE_OPENCODE_TO_CLAUDE["plan"], "Plan");
  });
});

// ── SSE processor: test the actual createSseProcessor from ./stream ─────────

const INBOUND_TOOL_NAME_MAP: Record<string, string> = {
  Bash: "bash",
  Read: "read",
  Glob: "glob",
  Grep: "grep",
  Edit: "edit",
  Write: "write",
  Agent: "task",
  WebFetch: "webfetch",
  TodoWrite: "todowrite",
  Skill: "skill",
  AskUserQuestion: "question",
  EnterPlanMode: "plan_enter",
  ExitPlanMode: "plan_exit",
};

function makeProcessor(opts: { debug?: (msg: string) => void } = {}) {
  return createSseProcessor({
    inboundToolNameMap: INBOUND_TOOL_NAME_MAP,
    translateToolArgs: translateToolArgsJsonString,
    debug: opts.debug,
  });
}

function sseEvent(eventName: string, data: object): string {
  return `event: ${eventName}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * Walk a concatenated SSE stream and return every parsed event, so tests
 * can assert on object shape rather than substring matching.
 */
function parseAllEvents(sse: string): Array<{ event: string | null; data: any }> {
  const events: Array<{ event: string | null; data: any }> = [];
  const frames = sse.split("\n\n").filter((f) => f.length > 0).map((f) => f + "\n\n");
  for (const frame of frames) {
    const parsed = parseSseEvent(frame);
    if (!parsed) continue;
    events.push({ event: parsed.event, data: JSON.parse(parsed.data) });
  }
  return events;
}

/**
 * Extract the tool-use input that will reach OpenCode for a given block
 * index. Returns the parsed args object the consumer's SDK will see after
 * concatenating all partial_json fragments for that block.
 */
function finalToolArgs(sse: string, blockIndex: number): any {
  const events = parseAllEvents(sse);
  const fragments: string[] = [];
  for (const e of events) {
    if (
      e.data?.type === "content_block_delta" &&
      e.data.index === blockIndex &&
      e.data.delta?.type === "input_json_delta"
    ) {
      fragments.push(String(e.data.delta.partial_json ?? ""));
    }
  }
  if (fragments.length === 0) return null;
  return JSON.parse(fragments.join(""));
}

function toolUseStartName(sse: string, blockIndex: number): string | null {
  const events = parseAllEvents(sse);
  for (const e of events) {
    if (
      e.data?.type === "content_block_start" &&
      e.data.index === blockIndex &&
      e.data.content_block?.type === "tool_use"
    ) {
      return e.data.content_block.name;
    }
  }
  return null;
}

// Helper that runs a tool_use through the processor: start, one or more
// delta fragments, stop. Returns the concatenated SSE output.
function runToolUse(toolName: string, fragments: string[], opts: { index?: number } = {}): string {
  const idx = opts.index ?? 1;
  const proc = makeProcessor();
  let out = "";
  out += proc.feedChunk(sseEvent("content_block_start", {
    type: "content_block_start",
    index: idx,
    content_block: { type: "tool_use", id: `tu_${idx}`, name: toolName, input: {} },
  }));
  for (const frag of fragments) {
    out += proc.feedChunk(sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: idx,
      delta: { type: "input_json_delta", partial_json: frag },
    }));
  }
  out += proc.feedChunk(sseEvent("content_block_stop", {
    type: "content_block_stop",
    index: idx,
  }));
  return out;
}

describe("SSE processor: tool name mapping", () => {
  for (const [claude, opencode] of Object.entries(INBOUND_TOOL_NAME_MAP)) {
    it(`maps ${claude} → ${opencode} on content_block_start`, () => {
      const out = runToolUse(claude, ['{}']);
      assert.equal(toolUseStartName(out, 1), opencode);
    });
  }

  it("passes through an unknown tool name without modification", () => {
    const out = runToolUse("CustomUnmappedTool", ['{}']);
    assert.equal(toolUseStartName(out, 1), "CustomUnmappedTool");
  });
});

describe("SSE processor: argument translation", () => {
  it("translates file_path → path for Read", () => {
    const out = runToolUse("Read", ['{"file_path": "/tmp/test.txt"}']);
    assert.deepEqual(finalToolArgs(out, 1), { path: "/tmp/test.txt" });
  });

  it("translates all Edit params", () => {
    const out = runToolUse("Edit", [
      '{"file_path": "/f.ts", "old_string": "foo", "new_string": "bar", "replace_all": true}',
    ]);
    assert.deepEqual(finalToolArgs(out, 1), {
      path: "/f.ts",
      oldString: "foo",
      newString: "bar",
      replaceAll: true,
    });
  });

  it("translates glob → include for Grep, preserves pattern", () => {
    const out = runToolUse("Grep", ['{"pattern": "hello", "glob": "*.ts"}']);
    assert.deepEqual(finalToolArgs(out, 1), { pattern: "hello", include: "*.ts" });
  });

  it("drops activeForm per todo item in TodoWrite", () => {
    const out = runToolUse("TodoWrite", [
      '{"todos": [{"content": "fix", "status": "pending", "activeForm": "Fixing"}]}',
    ]);
    assert.deepEqual(finalToolArgs(out, 1), {
      todos: [{ content: "fix", status: "pending" }],
    });
  });

  it("translates Agent subagent_type values", () => {
    const out = runToolUse("Agent", [
      '{"description": "d", "prompt": "p", "subagent_type": "general-purpose"}',
    ]);
    assert.deepEqual(finalToolArgs(out, 1), {
      description: "d",
      prompt: "p",
      subagent_type: "general",
    });
  });

  it("strips prompt and injects default format for WebFetch", () => {
    const out = runToolUse("WebFetch", [
      '{"url": "https://example.com", "prompt": "Extract the title"}',
    ]);
    assert.deepEqual(finalToolArgs(out, 1), {
      url: "https://example.com",
      format: "markdown",
    });
  });

  it("translates skill → name and strips args for Skill", () => {
    const out = runToolUse("Skill", ['{"skill": "commit", "args": "-m fix"}']);
    assert.deepEqual(finalToolArgs(out, 1), { id: "commit" });
  });

  it("leaves Bash args untouched", () => {
    const out = runToolUse("Bash", ['{"command": "echo hello"}']);
    assert.deepEqual(finalToolArgs(out, 1), { command: "echo hello" });
  });

  it("maps EnterPlanMode → plan_enter with empty args", () => {
    const out = runToolUse("EnterPlanMode", ['{}']);
    assert.equal(toolUseStartName(out, 1), "plan_enter");
    assert.deepEqual(finalToolArgs(out, 1), {});
  });

  it("maps ExitPlanMode → plan_exit preserving allowedPrompts", () => {
    const out = runToolUse("ExitPlanMode", [
      '{"allowedPrompts":[{"tool":"Bash","prompt":"run tests"}]}',
    ]);
    assert.equal(toolUseStartName(out, 1), "plan_exit");
    assert.deepEqual(finalToolArgs(out, 1), {
      allowedPrompts: [{ tool: "Bash", prompt: "run tests" }],
    });
  });
});

describe("SSE processor: chunk boundary handling", () => {
  it("handles args split across many small fragments", () => {
    const out = runToolUse("Read", ['{"fi', 'le_', 'pa', 'th":', ' "/', 'tmp/', 'x.t', 'xt"}']);
    assert.deepEqual(finalToolArgs(out, 1), { path: "/tmp/x.txt" });
  });

  it("handles multiple SSE events concatenated into one chunk", () => {
    const proc = makeProcessor();
    const combined =
      sseEvent("content_block_start", {
        type: "content_block_start",
        index: 1,
        content_block: { type: "tool_use", id: "tu_1", name: "Read", input: {} },
      }) +
      sseEvent("content_block_delta", {
        type: "content_block_delta",
        index: 1,
        delta: { type: "input_json_delta", partial_json: '{"file_path": "/a.txt"}' },
      }) +
      sseEvent("content_block_stop", {
        type: "content_block_stop",
        index: 1,
      });
    const out = proc.feedChunk(combined);
    assert.deepEqual(finalToolArgs(out, 1), { path: "/a.txt" });
    assert.ok(parseAllEvents(out).some((e) => e.data.type === "content_block_stop"));
  });

  it("handles an SSE event split across two chunks", () => {
    const proc = makeProcessor();
    const fullEvent = sseEvent("content_block_start", {
      type: "content_block_start",
      index: 1,
      content_block: { type: "tool_use", id: "tu_1", name: "Write", input: {} },
    });
    const mid = Math.floor(fullEvent.length / 2);
    const first = proc.feedChunk(fullEvent.slice(0, mid));
    assert.equal(first, "", "No complete event yet — should not emit");
    const second = proc.feedChunk(fullEvent.slice(mid));
    assert.equal(toolUseStartName(second, 1), "write");
  });

  it("passes through text deltas unchanged", () => {
    const proc = makeProcessor();
    const out = proc.feedChunk(sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "Hello world" },
    }));
    const events = parseAllEvents(out);
    assert.equal(events.length, 1);
    assert.equal(events[0].data.delta.text, "Hello world");
  });

  it("does NOT translate tool args inside a text_delta", () => {
    // Proves text content isn't mutated: a text_delta containing the
    // literal string "file_path" survives intact.
    const proc = makeProcessor();
    const out = proc.feedChunk(sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "use file_path to specify path" },
    }));
    const events = parseAllEvents(out);
    assert.equal(events[0].data.delta.text, "use file_path to specify path");
  });
});

describe("SSE processor: interleaved and concurrent tool_use blocks", () => {
  it("keeps per-block state isolated across interleaved deltas", () => {
    const proc = makeProcessor();
    let out = "";

    // Open block 1 (Read)
    out += proc.feedChunk(sseEvent("content_block_start", {
      type: "content_block_start",
      index: 1,
      content_block: { type: "tool_use", id: "tu_1", name: "Read", input: {} },
    }));
    // Open block 2 (Write) — shouldn't be valid per Anthropic's ordering but
    // the processor must still keep state by index, not a single "current".
    out += proc.feedChunk(sseEvent("content_block_start", {
      type: "content_block_start",
      index: 2,
      content_block: { type: "tool_use", id: "tu_2", name: "Write", input: {} },
    }));

    // Interleave deltas for both blocks
    out += proc.feedChunk(sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: '{"file_path": "/read.txt"}' },
    }));
    out += proc.feedChunk(sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: 2,
      delta: { type: "input_json_delta", partial_json: '{"file_path": "/write.txt", "content": "hi"}' },
    }));

    out += proc.feedChunk(sseEvent("content_block_stop", { type: "content_block_stop", index: 1 }));
    out += proc.feedChunk(sseEvent("content_block_stop", { type: "content_block_stop", index: 2 }));

    assert.deepEqual(finalToolArgs(out, 1), { path: "/read.txt" });
    assert.deepEqual(finalToolArgs(out, 2), { path: "/write.txt", content: "hi" });
  });
});

describe("SSE processor: error handling", () => {
  it("calls debug callback on malformed JSON in an SSE data frame", () => {
    const messages: string[] = [];
    const proc = createSseProcessor({
      inboundToolNameMap: INBOUND_TOOL_NAME_MAP,
      translateToolArgs: translateToolArgsJsonString,
      debug: (m) => messages.push(m),
    });
    const malformed = "event: content_block_start\ndata: {not valid\n\n";
    const out = proc.feedChunk(malformed);
    // Passed through as-is
    assert.equal(out, malformed);
    // And we told the operator about it (behind their debug flag)
    assert.ok(messages.some((m) => m.includes("JSON.parse failed")), `Expected debug log, got: ${JSON.stringify(messages)}`);
  });

  it("calls debug callback when translateToolArgs throws", () => {
    const messages: string[] = [];
    const proc = createSseProcessor({
      inboundToolNameMap: INBOUND_TOOL_NAME_MAP,
      translateToolArgs: () => { throw new Error("translator blew up"); },
      debug: (m) => messages.push(m),
    });
    let out = "";
    out += proc.feedChunk(sseEvent("content_block_start", {
      type: "content_block_start",
      index: 1,
      content_block: { type: "tool_use", id: "tu_1", name: "Read", input: {} },
    }));
    out += proc.feedChunk(sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: '{"file_path": "/x"}' },
    }));
    out += proc.feedChunk(sseEvent("content_block_stop", { type: "content_block_stop", index: 1 }));
    assert.ok(messages.some((m) => m.includes("translator blew up")));
    // And we fall back to emitting the raw (untranslated) JSON so downstream doesn't hang
    assert.deepEqual(finalToolArgs(out, 1), { file_path: "/x" });
  });
});

describe("SSE processor: flush", () => {
  it("flush returns any trailing buffered bytes", () => {
    const proc = makeProcessor();
    const partial = "event: content_block_start\ndata: {\"t"; // incomplete frame
    assert.equal(proc.feedChunk(partial), "");
    assert.equal(proc.flush(), partial);
    // And state is cleared after flush
    assert.equal(proc.flush(), "");
  });

  it("logs a debug warning when a tool_use block is abandoned mid-stream", () => {
    // Simulates upstream disconnect after start + some deltas but before stop.
    const messages: string[] = [];
    const proc = createSseProcessor({
      inboundToolNameMap: INBOUND_TOOL_NAME_MAP,
      translateToolArgs: translateToolArgsJsonString,
      debug: (m) => messages.push(m),
    });
    proc.feedChunk(sseEvent("content_block_start", {
      type: "content_block_start",
      index: 1,
      content_block: { type: "tool_use", id: "tu_1", name: "Read", input: {} },
    }));
    proc.feedChunk(sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: 1,
      delta: { type: "input_json_delta", partial_json: '{"file_path"' },
    }));
    // ...and then the stream ends without a content_block_stop.
    proc.flush();
    assert.ok(
      messages.some((m) => m.includes("abandoned") && m.includes("index=1") && m.includes("tool=Read")),
      `Expected abandoned-block warning, got: ${JSON.stringify(messages)}`,
    );
  });
});

describe("SSE processor: pass-through optimization", () => {
  // Pass-through events (message_start, ping, message_delta, text
  // content_block_delta, etc.) should not be round-tripped through
  // JSON.parse + JSON.stringify — the output bytes should match the input
  // exactly when no translation is needed.
  it("emits the exact input bytes for ping events (no reserialize)", () => {
    const proc = makeProcessor();
    const pingFrame = sseEvent("ping", { type: "ping" });
    const out = proc.feedChunk(pingFrame);
    assert.equal(out, pingFrame);
  });

  it("emits the exact input bytes for message_start events", () => {
    const proc = makeProcessor();
    // Anthropic message_start payloads include nested objects and arrays —
    // passthrough must preserve byte-for-byte equality, including any
    // particular key order the API happens to emit.
    const frame =
      "event: message_start\n" +
      'data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1}}}' +
      "\n\n";
    const out = proc.feedChunk(frame);
    assert.equal(out, frame);
  });

  it("emits the exact input bytes for text_delta events", () => {
    const proc = makeProcessor();
    const frame = sseEvent("content_block_delta", {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "hello world" },
    });
    const out = proc.feedChunk(frame);
    assert.equal(out, frame);
  });

  it("still transforms tool_use content_block_start (optimization does not skip interesting events)", () => {
    const proc = makeProcessor();
    const input = sseEvent("content_block_start", {
      type: "content_block_start",
      index: 1,
      content_block: { type: "tool_use", id: "tu_1", name: "Read", input: {} },
    });
    const out = proc.feedChunk(input);
    assert.notEqual(out, input, "tool_use start should be transformed (name mapped)");
    assert.equal(toolUseStartName(out, 1), "read");
  });
});

describe("parseSseEvent / buildSseEvent round trip", () => {
  it("round-trips a simple event", () => {
    const original = buildSseEvent("ping", '{"type":"ping"}');
    const parsed = parseSseEvent(original);
    assert.deepEqual(parsed, { event: "ping", data: '{"type":"ping"}' });
  });

  it("returns null for a frame with no data line", () => {
    assert.equal(parseSseEvent("event: foo\n\n"), null);
  });

  it("handles a data-only frame (no event line)", () => {
    const parsed = parseSseEvent('data: {"ok":true}\n\n');
    assert.deepEqual(parsed, { event: null, data: '{"ok":true}' });
  });
});

describe("deriveModelDisplayName", () => {
  it("parses claude-opus-4-7 → Opus 4.7", () => {
    assert.equal(deriveModelDisplayName("claude-opus-4-7"), "Opus 4.7");
  });

  it("parses claude-sonnet-4-6 → Sonnet 4.6", () => {
    assert.equal(deriveModelDisplayName("claude-sonnet-4-6"), "Sonnet 4.6");
  });

  it("ignores a trailing date suffix (claude-haiku-4-5-20251001 → Haiku 4.5)", () => {
    assert.equal(deriveModelDisplayName("claude-haiku-4-5-20251001"), "Haiku 4.5");
  });

  it("falls back to the raw id when the convention doesn't match", () => {
    assert.equal(deriveModelDisplayName("weird-custom-model"), "weird-custom-model");
    assert.equal(deriveModelDisplayName("claude-3-5-sonnet-20241022"), "claude-3-5-sonnet-20241022");
  });
});

describe("rewriteSystemBlocksForModel", () => {
  const sourceBlock = {
    type: "text",
    text:
      "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n" +
      " - You are powered by the model named Sonnet 4.6. The exact model ID is claude-sonnet-4-6.\n" +
      " - Assistant knowledge cutoff is August 2025.",
  };

  it("rewrites the identity line to match the requested model", () => {
    const [out] = rewriteSystemBlocksForModel([sourceBlock], "claude-opus-4-7");
    assert.ok(out.text?.includes("named Opus 4.7"));
    assert.ok(out.text?.includes("claude-opus-4-7"));
    assert.ok(!out.text?.includes("Sonnet 4.6"));
    assert.ok(!out.text?.includes("claude-sonnet-4-6"));
  });

  it("leaves blocks untouched when no model id is provided", () => {
    const [out] = rewriteSystemBlocksForModel([sourceBlock], undefined);
    assert.equal(out, sourceBlock);
  });

  it("handles a dated variant without re-including the date suffix in the display name", () => {
    const [out] = rewriteSystemBlocksForModel([sourceBlock], "claude-haiku-4-5-20251001");
    assert.ok(out.text?.includes("named Haiku 4.5"));
    assert.ok(out.text?.includes("The exact model ID is claude-haiku-4-5-20251001."));
  });

  it("preserves non-text blocks and non-matching text", () => {
    const nonText = { type: "other", text: "unchanged" };
    const noMatch = { type: "text", text: "no identity line here" };
    const blocks = [nonText, noMatch];
    const out = rewriteSystemBlocksForModel(blocks, "claude-opus-4-7");
    assert.deepEqual(out[0], nonText);
    assert.equal(out[1].text, "no identity line here");
  });
});

describe("mergeSystemBlocks", () => {
  const billing = "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli; cch=00000;";
  const claude = [{ type: "text", text: "You are Claude Code." }];

  it("keeps OpenCode's system blocks (e.g. AGENTS.md) after the Claude Code prompt", () => {
    const opencode = [{ type: "text", text: "Instructions from: AGENTS.md\nBe terse." }];
    const out = mergeSystemBlocks(billing, claude, opencode);
    assert.deepEqual(out.map((b) => b.text), [
      billing,
      "You are Claude Code.",
      "Instructions from: AGENTS.md\nBe terse.",
    ]);
  });

  it("emits the billing header exactly once", () => {
    const opencode = [
      { type: "text", text: billing },
      { type: "text", text: "opencode prompt" },
    ];
    const out = mergeSystemBlocks(billing, claude, opencode);
    const billingBlocks = out.filter((b) => b.text?.startsWith("x-anthropic-billing-header:"));
    assert.equal(billingBlocks.length, 1);
    assert.equal(out[0].text, billing);
    assert.equal(out[out.length - 1].text, "opencode prompt");
  });

  it("strips the context-hook CLAUDE_PREFIX from OpenCode's blocks when the Claude prompt has it", () => {
    const prefix = "You are a Claude agent, built on Anthropic's Claude Agent SDK.";
    const claudeWithPrefix = [{ type: "text", text: `${prefix}\n\nClaude Code instructions.` }];
    const opencode = [{ type: "text", text: `${prefix}\n\nInstructions from: AGENTS.md` }];
    const out = mergeSystemBlocks(billing, claudeWithPrefix, opencode);
    const occurrences = out.filter((b) => b.text?.includes(prefix)).length;
    assert.equal(occurrences, 1);
    assert.equal(out[out.length - 1].text, "Instructions from: AGENTS.md");
  });

  it("drops a bare CLAUDE_PREFIX block when the Claude prompt has it", () => {
    const prefix = "You are a Claude agent, built on Anthropic's Claude Agent SDK.";
    const claudeWithPrefix = [{ type: "text", text: prefix }];
    const out = mergeSystemBlocks(billing, claudeWithPrefix, [{ type: "text", text: prefix }]);
    assert.equal(out.length, 2);
  });

  it("keeps the prefix on OpenCode's blocks when the Claude prompt lacks it", () => {
    const prefix = "You are a Claude agent, built on Anthropic's Claude Agent SDK.";
    const opencode = [{ type: "text", text: `${prefix}\n\nInstructions from: AGENTS.md` }];
    const out = mergeSystemBlocks(billing, claude, opencode);
    assert.equal(out[out.length - 1].text, `${prefix}\n\nInstructions from: AGENTS.md`);
  });
});

describe("plugin module shape", () => {
  it("default-exports { id, setup } so opencode's loader recognises it", () => {
    // A bare function default export sends opencode down its legacy path, where
    // it calls every named export of this module as a plugin factory — which
    // crashes on the helpers above and takes the whole plugin down with it.
    assert.equal(typeof pluginModule, "object");
    assert.equal(typeof pluginModule.setup, "function");
  });

  it("carries an id, which opencode requires for path-referenced plugins", () => {
    assert.equal(pluginModule.id, "opencode-claude-bridge");
  });

  it("v2 setup registers the anthropic oauth method override and provider-scoped hooks", async () => {
    const registrations: string[] = [];
    const methodUpdate = { capture: null as unknown };
    const hooks: Array<{ name: string; options?: { providerID?: string } }> = [];
    const ctx = {
      integration: {
        transform: async (editor: (editor: {
          method: {
            update(registration: Record<string, unknown>): void;
          };
        }) => void) => {
          editor({
            method: {
              update(registration: Record<string, unknown>) {
                methodUpdate.capture = registration;
                registrations.push("integration.transform");
              },
            },
          });
        },
        connection: {
          active: async () => undefined,
          resolve: async () => undefined,
        },
      },
      session: {
        hook: async (name: string, _cb: unknown, options?: { providerID?: string }) => {
          hooks.push({ name, options });
          registrations.push(`session.hook(${name})`);
        },
      },
    };
    // setup() kicks off a background CLI-version refresh; keep it off the network.
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      throw new Error("network disabled in tests");
    }) as typeof globalThis.fetch;
    let cleanup: unknown;
    try {
      cleanup = await (pluginModule.setup as (c: unknown) => Promise<unknown>)(ctx);
    } finally {
      globalThis.fetch = realFetch;
    }

    const registration = methodUpdate.capture as Record<string, unknown>;
    assert.equal(registration.integrationID, "anthropic");
    const method = registration.method as Record<string, unknown>;
    assert.equal(method.id, "oauth");
    assert.equal(method.type, "oauth");
    assert.equal(typeof registration.authorize, "function");
    assert.equal(typeof registration.refresh, "function");
    // The refresh callback must be async — opencode calls .then() on its result
    // (a plain-object return crashes the credential resolver).
    assert.equal(
      (registration.refresh as (...args: unknown[]) => unknown).constructor.name,
      "AsyncFunction",
      "refresh must be async so opencode can await it",
    );

    assert.deepEqual(
      hooks.map((h) => `${h.name}:${h.options?.providerID ?? ""}`).sort(),
      [
        "compaction:anthropic",
        "context:anthropic",
        "generate:anthropic",
        "http.request:anthropic",
        "http.response:anthropic",
        "title:anthropic",
      ],
    );
    assert.equal(typeof cleanup, "function");
  });

  it("v2 setup prefixes the system prompt of title and compaction requests", async () => {
    const callbacks = new Map<string, (event: any) => unknown>();
    const ctx = {
      integration: {
        transform: async (editor: (editor: { method: { update(r: unknown): void } }) => void) => {
          editor({ method: { update() {} } });
        },
        connection: { active: async () => undefined, resolve: async () => undefined },
      },
      session: {
        hook: async (name: string, cb: (event: any) => unknown) => {
          callbacks.set(name, cb);
        },
      },
    };
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      throw new Error("network disabled in tests");
    }) as typeof globalThis.fetch;
    try {
      await (pluginModule.setup as (c: unknown) => Promise<unknown>)(ctx);
    } finally {
      globalThis.fetch = realFetch;
    }

    for (const kind of ["title", "compaction"]) {
      const event = {
        model: { providerID: "anthropic" },
        system: [{ type: "text", text: "Generate a title." }],
      };
      callbacks.get(kind)!(event);
      assert.ok(event.system[0].text.startsWith(CLAUDE_PREFIX), `${kind} must be prefixed`);

      const other = {
        model: { providerID: "openai" },
        system: [{ type: "text", text: "Generate a title." }],
      };
      callbacks.get(kind)!(other);
      assert.equal(other.system[0].text, "Generate a title.", `${kind} must skip other providers`);
    }
  });
});

describe("http hooks: per-request OAuth tracking", () => {
  const url = "https://api.anthropic.com/v1/messages";
  const body = JSON.stringify({
    model: "claude-opus-4-7",
    messages: [{ role: "user", content: "hi" }],
    tools: [{ name: "read", description: "Read a file", input_schema: { type: "object", properties: {} } }],
  });
  const sse = () =>
    new Response(
      'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"Read","input":{}}}\n\n' +
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
      { headers: { "content-type": "text/event-stream" } },
    );

  async function setupHooks() {
    const callbacks = new Map<string, (event: any) => unknown>();
    const ctx = {
      integration: {
        transform: async (editor: (editor: { method: { update(r: unknown): void } }) => void) => {
          editor({ method: { update() {} } });
        },
        connection: { active: async () => undefined, resolve: async () => undefined },
      },
      session: {
        hook: async (name: string, cb: (event: any) => unknown) => {
          callbacks.set(name, cb);
        },
      },
    };
    await (pluginModule.setup as (c: unknown) => Promise<unknown>)(ctx);
    return {
      request: async (request: Request) => {
        const event = { request };
        await callbacks.get("http.request")!(event);
        return event.request;
      },
      response: async (request: Request, response: Response) => {
        const event = { request, response };
        await callbacks.get("http.response")!(event);
        return event.response;
      },
    };
  }

  async function withNetworkDisabled(run: () => Promise<void>) {
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      throw new Error("network disabled in tests");
    }) as typeof globalThis.fetch;
    try {
      await run();
    } finally {
      globalThis.fetch = realFetch;
    }
  }

  it("translates an OAuth response even after an API-key request passed through", async () => {
    await withNetworkDisabled(async () => {
      const hooks = await setupHooks();
      const oauth = await hooks.request(
        new Request(url, { method: "POST", headers: { authorization: "Bearer tok" }, body }),
      );
      const apiKeyOriginal = new Request(url, { method: "POST", headers: { "x-api-key": "sk" }, body });
      const apiKey = await hooks.request(apiKeyOriginal);
      assert.equal(apiKey, apiKeyOriginal, "API-key request must pass through");

      const text = await (await hooks.response(oauth, sse())).text();
      assert.match(text, /"name":"read"/);
    });
  });

  it("leaves the response of a passthrough request untouched", async () => {
    await withNetworkDisabled(async () => {
      const hooks = await setupHooks();
      await hooks.request(new Request(url, { method: "POST", headers: { authorization: "Bearer tok" }, body }));
      const apiKey = await hooks.request(
        new Request(url, { method: "POST", headers: { "x-api-key": "sk" }, body }),
      );
      const original = sse();
      assert.equal(await hooks.response(apiKey, original), original);
    });
  });

  it("falls back to the session header when the rewritten request was replaced", async () => {
    await withNetworkDisabled(async () => {
      const hooks = await setupHooks();
      const oauth = await hooks.request(
        new Request(url, { method: "POST", headers: { authorization: "Bearer tok" }, body }),
      );
      const replaced = new Request(oauth.url, {
        method: "POST",
        headers: { "x-claude-code-session-id": oauth.headers.get("x-claude-code-session-id")! },
      });
      const text = await (await hooks.response(replaced, sse())).text();
      assert.match(text, /"name":"read"/);
    });
  });

  it("ignores a request it never rewrote", async () => {
    await withNetworkDisabled(async () => {
      const hooks = await setupHooks();
      const original = sse();
      assert.equal(await hooks.response(new Request(url, { method: "POST" }), original), original);
    });
  });
});

describe("refreshCredentialLayered", () => {
  const HOUR = 3_600_000;
  const fileSource: CredentialSource = { kind: "file", path: "/tmp/creds.json" };
  const cliTokens = (refresh: string, expires: number): ClaudeCliTokens => ({
    access: `access-${refresh}`,
    refresh,
    expires,
    source: fileSource,
  });
  const fresh = (from: string) => ({ access: `new-a-${from}`, refresh: `new-r-${from}`, expires: Date.now() + HOUR });

  function harness(opts: {
    cli?: Array<ClaudeCliTokens | null>;
    refresh?: (token: string) => { access: string; refresh: string; expires: number };
    writeBack?: boolean;
  }) {
    const reads = [...(opts.cli ?? [null])];
    const refreshed: string[] = [];
    const writes: Array<{ spent: string; refresh: string }> = [];
    const warnings: string[] = [];
    const deps = {
      readCli: () => (reads.length > 1 ? reads.shift()! : reads[0]),
      refresh: (token: string) => {
        refreshed.push(token);
        return (opts.refresh ?? fresh)(token);
      },
      writeBack: (_source: CredentialSource, spent: string, tokens: { refresh: string }) => {
        writes.push({ spent, refresh: tokens.refresh });
        return opts.writeBack ?? true;
      },
      warn: (m: string) => warnings.push(m),
    };
    return { deps, refreshed, writes, warnings };
  }

  it("adopts a valid CLI access token without refreshing", () => {
    const h = harness({ cli: [cliTokens("cli-r", Date.now() + HOUR)] });
    const result = refreshCredentialLayered({ refresh: "stored-r" }, h.deps);
    assert.equal(result?.access, "access-cli-r");
    assert.deepEqual(h.refreshed, []);
  });

  it("writes back after spending a refresh token shared with the CLI", () => {
    const h = harness({ cli: [cliTokens("shared-r", Date.now() - 1)] });
    const result = refreshCredentialLayered({ refresh: "shared-r" }, h.deps);
    assert.equal(result?.refresh, "new-r-shared-r");
    assert.deepEqual(h.refreshed, ["shared-r"]);
    assert.deepEqual(h.writes, [{ spent: "shared-r", refresh: "new-r-shared-r" }]);
  });

  it("does not write back when the stored token is not the CLI's", () => {
    const h = harness({ cli: [cliTokens("cli-r", Date.now() - 1)] });
    const result = refreshCredentialLayered({ refresh: "browser-r" }, h.deps);
    assert.equal(result?.refresh, "new-r-browser-r");
    assert.deepEqual(h.writes, []);
  });

  it("falls back to the CLI token and writes it back", () => {
    const h = harness({
      cli: [cliTokens("cli-r", Date.now() - 1)],
      refresh: (token) => {
        if (token === "stored-r") throw new Error("invalid_grant");
        return fresh(token);
      },
    });
    const result = refreshCredentialLayered({ refresh: "stored-r" }, h.deps);
    assert.equal(result?.refresh, "new-r-cli-r");
    assert.deepEqual(h.refreshed, ["stored-r", "cli-r"]);
    assert.deepEqual(h.writes, [{ spent: "cli-r", refresh: "new-r-cli-r" }]);
  });

  it("adopts tokens another refresher just rotated", () => {
    const h = harness({
      cli: [cliTokens("shared-r", Date.now() - 1), cliTokens("rotated-r", Date.now() + HOUR)],
      refresh: () => {
        throw new Error("invalid_grant");
      },
    });
    const result = refreshCredentialLayered({ refresh: "shared-r" }, h.deps);
    assert.equal(result?.refresh, "rotated-r");
    assert.deepEqual(h.refreshed, ["shared-r"]);
    assert.deepEqual(h.writes, []);
  });

  it("throws the last error when every refresh fails", () => {
    const h = harness({
      cli: [cliTokens("cli-r", Date.now() - 1)],
      refresh: (token) => {
        throw new Error(`failed ${token}`);
      },
    });
    assert.throws(() => refreshCredentialLayered({ refresh: "stored-r" }, h.deps), /failed cli-r/);
  });

  it("returns null when there is nothing to refresh with", () => {
    const h = harness({});
    assert.equal(refreshCredentialLayered({}, h.deps), null);
  });

  it("still returns fresh tokens when the write-back fails", () => {
    const h = harness({ cli: [cliTokens("shared-r", Date.now() - 1)], writeBack: false });
    const result = refreshCredentialLayered({ refresh: "shared-r" }, h.deps);
    assert.equal(result?.refresh, "new-r-shared-r");
    assert.equal(h.warnings.length, 1);
  });
});

describe("Claude CLI credential write-back", () => {
  const blob = (refreshToken: string) =>
    JSON.stringify({
      claudeAiOauth: {
        accessToken: "old-a",
        refreshToken,
        expiresAt: 1,
        scopes: ["user:inference"],
        subscriptionType: "max",
      },
      other: { keep: true },
    });
  const tokens = { access: "new-a", refresh: "new-r", expires: 42 };

  it("updateCredentialBlob swaps tokens and keeps other fields", () => {
    const updated = JSON.parse(updateCredentialBlob(blob("spent-r"), "spent-r", tokens)!);
    assert.deepEqual(updated, {
      claudeAiOauth: {
        accessToken: "new-a",
        refreshToken: "new-r",
        expiresAt: 42,
        scopes: ["user:inference"],
        subscriptionType: "max",
      },
      other: { keep: true },
    });
  });

  it("updateCredentialBlob refuses a blob rotated by someone else", () => {
    assert.equal(updateCredentialBlob(blob("other-r"), "spent-r", tokens), null);
    assert.equal(updateCredentialBlob("{not json", "spent-r", tokens), null);
  });

  const withTempDir = (run: (dir: string) => void) => {
    const dir = mkdtempSync(join(tmpdir(), "ocb-creds-"));
    try {
      run(dir);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  it("writes the file atomically with mode 0600 when the token matches", () => {
    withTempDir((dir) => {
      const path = join(dir, ".credentials.json");
      writeFileSync(path, blob("spent-r"), { mode: 0o644 });
      assert.equal(writeBackCredentialsFile(path, "spent-r", tokens), true);
      assert.equal(JSON.parse(readFileSync(path, "utf8")).claudeAiOauth.refreshToken, "new-r");
      assert.equal(statSync(path).mode & 0o777, 0o600);
      assert.deepEqual(readdirSync(dir), [".credentials.json"]);
    });
  });

  it("leaves the file alone when the token no longer matches", () => {
    withTempDir((dir) => {
      const path = join(dir, ".credentials.json");
      writeFileSync(path, blob("other-r"));
      assert.equal(writeBackCredentialsFile(path, "spent-r", tokens), false);
      assert.equal(readFileSync(path, "utf8"), blob("other-r"));
      assert.deepEqual(readdirSync(dir), [".credentials.json"]);
    });
  });

  it("builds a security -i command with the secret hex-encoded", () => {
    const secret = blob("new-r");
    const command = buildKeychainWriteCommand("Claude Code-credentials", "alan", secret)!;
    assert.equal(
      command,
      `add-generic-password -U -s "Claude Code-credentials" -a "alan" -X ${Buffer.from(secret).toString("hex")}\n`,
    );
    assert.ok(!command.includes("new-r"), "token must not appear in clear text");
  });

  it("refuses names it cannot quote safely", () => {
    assert.equal(buildKeychainWriteCommand("svc", 'a"b', "{}"), null);
    assert.equal(buildKeychainWriteCommand("svc", "a\nb", "{}"), null);
  });
});

describe("parseTokenResponse", () => {
  const ok = (data: Record<string, unknown>) => JSON.stringify(data);

  it("keeps the current refresh token when the endpoint doesn't rotate it", () => {
    const tokens = parseTokenResponse(200, ok({ access_token: "a", expires_in: 3600 }), "Token refresh", "old-r");
    assert.equal(tokens.refresh, "old-r");
  });

  it("prefers a rotated refresh token", () => {
    const tokens = parseTokenResponse(
      200,
      ok({ access_token: "a", refresh_token: "new-r", expires_in: 3600 }),
      "Token refresh",
      "old-r",
    );
    assert.equal(tokens.refresh, "new-r");
  });

  it("returns an integer expiry", () => {
    const tokens = parseTokenResponse(200, ok({ access_token: "a", refresh_token: "r", expires_in: 0.0005 }), "x");
    assert.ok(Number.isInteger(tokens.expires));
  });

  it("rejects a response without an access token", () => {
    assert.throws(
      () => parseTokenResponse(200, ok({ refresh_token: "r", expires_in: 3600 }), "Token refresh"),
      /no access_token/,
    );
  });

  it("rejects a response without any refresh token", () => {
    assert.throws(
      () => parseTokenResponse(200, ok({ access_token: "a", expires_in: 3600 }), "Token exchange"),
      /no refresh_token/,
    );
  });
});

describe("injectClaudePrefix", () => {
  it("prepends the prefix to the first block", () => {
    const system = [{ type: "text", text: "Body" }, { type: "text", text: "Env" }];
    injectClaudePrefix(system);
    assert.equal(system[0].text, `${CLAUDE_PREFIX}\n\nBody`);
    assert.equal(system[1].text, "Env");
  });

  it("adds a block when the system prompt is empty", () => {
    const system: Array<{ type?: string; text?: string }> = [];
    injectClaudePrefix(system);
    assert.deepEqual(system, [{ type: "text", text: CLAUDE_PREFIX }]);
  });

  it("leaves the prompt alone when a later block already carries the prefix", () => {
    const system = [{ type: "text", text: "Body" }, { type: "text", text: CLAUDE_PREFIX }];
    injectClaudePrefix(system);
    assert.equal(system[0].text, "Body");
  });

  it("is idempotent", () => {
    const system = [{ type: "text", text: "Body" }];
    injectClaudePrefix(system);
    injectClaudePrefix(system);
    assert.equal(system[0].text, `${CLAUDE_PREFIX}\n\nBody`);
  });
});

describe("resolveOAuthAccessToken", () => {
  const future = () => Date.now() + 3_600_000;
  const past = () => Date.now() - 1_000;
  const request = (headers: Record<string, string> = {}) =>
    new Request("https://api.anthropic.com/v1/messages", { method: "POST", headers });
  const cli = () => ({ access: "cli-token", expires: future() });

  it("uses the bearer token already on the request without any lookup", async () => {
    const token = await resolveOAuthAccessToken(request({ authorization: "Bearer req-token" }), {
      getActiveCredential: async () => {
        throw new Error("must not be called");
      },
      getCliTokens: () => {
        throw new Error("must not be called");
      },
    });
    assert.equal(token, "req-token");
  });

  it("passes API-key requests through even when the Claude CLI is logged in", async () => {
    const token = await resolveOAuthAccessToken(request({ "x-api-key": "sk-ant-api" }), {
      getActiveCredential: async () => undefined,
      getCliTokens: cli,
    });
    assert.equal(token, null);
  });

  it("passes through when the active connection is not OAuth", async () => {
    const token = await resolveOAuthAccessToken(request(), {
      getActiveCredential: async () => ({ type: "api" }),
      getCliTokens: cli,
    });
    assert.equal(token, null);
  });

  it("uses a valid OAuth connection", async () => {
    const token = await resolveOAuthAccessToken(request(), {
      getActiveCredential: async () => ({ type: "oauth", access: "stored", expires: future() }),
      getCliTokens: cli,
    });
    assert.equal(token, "stored");
  });

  it("falls back to the CLI token when the OAuth connection is expired", async () => {
    const token = await resolveOAuthAccessToken(request(), {
      getActiveCredential: async () => ({ type: "oauth", access: "stale", expires: past() }),
      getCliTokens: cli,
    });
    assert.equal(token, "cli-token");
  });

  it("falls back to the CLI token when there is no connection", async () => {
    const token = await resolveOAuthAccessToken(request(), {
      getActiveCredential: async () => undefined,
      getCliTokens: cli,
    });
    assert.equal(token, "cli-token");
  });

  it("falls back to the CLI token when the connection lookup throws", async () => {
    const token = await resolveOAuthAccessToken(request(), {
      getActiveCredential: async () => {
        throw new Error("store unavailable");
      },
      getCliTokens: cli,
    });
    assert.equal(token, "cli-token");
  });

  it("returns null when the CLI token is expired too", async () => {
    const token = await resolveOAuthAccessToken(request(), {
      getActiveCredential: async () => undefined,
      getCliTokens: () => ({ access: "old", expires: past() }),
    });
    assert.equal(token, null);
  });
});

describe("CLI version lookup", () => {
  const DAY = 24 * 60 * 60 * 1000;
  const distTags = { stable: "2.1.311", latest: "2.1.319", next: "2.1.319" };
  const jsonFetch = (body: unknown, ok = true) => {
    let calls = 0;
    const fn = (async () => {
      calls++;
      return { ok, json: async () => body } as Response;
    }) as typeof globalThis.fetch;
    return { fn, calls: () => calls };
  };
  const withTempCache = async (run: (cachePath: string) => Promise<void>) => {
    const dir = mkdtempSync(join(tmpdir(), "ocb-version-"));
    try {
      await run(join(dir, "nested", "cli-version.json"));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };

  it("parseDistTagsVersion picks the requested tag and rejects junk", () => {
    assert.equal(parseDistTagsVersion(distTags, "stable"), "2.1.311");
    assert.equal(parseDistTagsVersion(distTags, "latest"), "2.1.319");
    assert.equal(parseDistTagsVersion({ stable: "not-a-version" }, "stable"), null);
    assert.equal(parseDistTagsVersion(null, "stable"), null);
  });

  it("fetches the stable tag and writes the cache", async () => {
    await withTempCache(async (cachePath) => {
      const f = jsonFetch(distTags);
      const version = await fetchCliVersion({ fetch: f.fn, cachePath, now: 1000, tag: "stable" });
      assert.equal(version, "2.1.311");
      assert.deepEqual(JSON.parse(readFileSync(cachePath, "utf8")), {
        version: "2.1.311",
        fetchedAt: 1000,
      });
    });
  });

  it("serves a fresh cache without touching the network", async () => {
    await withTempCache(async (cachePath) => {
      await fetchCliVersion({ fetch: jsonFetch(distTags).fn, cachePath, now: 0, tag: "stable" });
      const f = jsonFetch({ stable: "9.9.9" });
      const version = await fetchCliVersion({ fetch: f.fn, cachePath, now: DAY - 1, tag: "stable" });
      assert.equal(version, "2.1.311");
      assert.equal(f.calls(), 0);
    });
  });

  it("refetches once the cache is stale", async () => {
    await withTempCache(async (cachePath) => {
      await fetchCliVersion({ fetch: jsonFetch(distTags).fn, cachePath, now: 0, tag: "stable" });
      const f = jsonFetch({ stable: "2.1.400" });
      const version = await fetchCliVersion({ fetch: f.fn, cachePath, now: DAY + 1, tag: "stable" });
      assert.equal(version, "2.1.400");
      assert.equal(f.calls(), 1);
    });
  });

  it("falls back to a stale cache when the registry fails", async () => {
    await withTempCache(async (cachePath) => {
      await fetchCliVersion({ fetch: jsonFetch(distTags).fn, cachePath, now: 0, tag: "stable" });
      const failing = (async () => {
        throw new Error("offline");
      }) as typeof globalThis.fetch;
      assert.equal(
        await fetchCliVersion({ fetch: failing, cachePath, now: DAY * 2, tag: "stable" }),
        "2.1.311",
      );
      assert.equal(
        await fetchCliVersion({ fetch: jsonFetch({}, false).fn, cachePath, now: DAY * 2, tag: "stable" }),
        "2.1.311",
      );
    });
  });

  it("returns null with no cache and a failing registry", async () => {
    await withTempCache(async (cachePath) => {
      const version = await fetchCliVersion({
        fetch: jsonFetch({ stable: "garbage" }).fn,
        cachePath,
        now: 0,
        tag: "stable",
      });
      assert.equal(version, null);
    });
  });

  it("ignores a corrupt cache file", async () => {
    await withTempCache(async (cachePath) => {
      await fetchCliVersion({ fetch: jsonFetch(distTags).fn, cachePath, now: 0, tag: "stable" });
      writeFileSync(cachePath, "{not json");
      const version = await fetchCliVersion({ fetch: jsonFetch(distTags).fn, cachePath, now: 1, tag: "stable" });
      assert.equal(version, "2.1.311");
    });
  });
});

describe("selectClaudeCredentials", () => {
  const real = {
    claudeAiOauth: {
      accessToken: "a",
      refreshToken: "r",
      expiresAt: 1,
    },
  };
  // What attaching an MCP server writes under the same service name.
  const mcpOnly = { mcpOAuth: { some: "thing" } } as never;

  it("skips an item that has no claudeAiOauth", () => {
    assert.equal(selectClaudeCredentials([mcpOnly, real]), real);
  });

  it("prefers the first item that does carry claudeAiOauth", () => {
    const other = { claudeAiOauth: { accessToken: "b", refreshToken: "r", expiresAt: 2 } };
    assert.equal(selectClaudeCredentials([real, other]), real);
  });

  it("returns null when nothing carries claudeAiOauth", () => {
    assert.equal(selectClaudeCredentials([null, mcpOnly]), null);
  });

  it("tolerates a null candidate from a failed lookup", () => {
    assert.equal(selectClaudeCredentials([null, real]), real);
  });
});
