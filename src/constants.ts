import { homedir } from "node:os";
import { join } from "node:path";

export const FALLBACK_CLI_VERSION = "2.1.319";

export const CLI_VERSION_URL =
  process.env.ANTHROPIC_CLI_VERSION_URL ||
  "https://registry.npmjs.org/-/package/@anthropic-ai/claude-code/dist-tags";

// npm dist-tag to follow. "stable" trails "latest" by a few releases and is
// what most real Claude Code installs report.
export const CLI_VERSION_TAG = process.env.ANTHROPIC_CLI_VERSION_TAG || "stable";

export const CLI_VERSION_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
export const CLI_VERSION_FETCH_TIMEOUT_MS = 2000;

export const CLI_VERSION_CACHE_PATH = join(
  process.env.XDG_CACHE_HOME || join(homedir(), ".cache"),
  "opencode-claude-bridge",
  "cli-version.json",
);

/**
 * Identity opencode registers this plugin under. Required: opencode refuses to
 * load a plugin referenced by filesystem path unless its module exports an id.
 */
export const PLUGIN_ID = "opencode-claude-bridge";

export const CLIENT_ID =
  process.env.ANTHROPIC_CLIENT_ID || "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

export const PROFILE_URL =
  process.env.ANTHROPIC_PROFILE_URL ||
  "https://api.anthropic.com/api/oauth/profile";

export const TOKEN_URL =
  process.env.ANTHROPIC_TOKEN_URL ||
  "https://platform.claude.com/v1/oauth/token";

export const AUTHORIZE_URL =
  process.env.ANTHROPIC_AUTHORIZE_URL ||
  "https://claude.com/cai/oauth/authorize";

// Manual redirect URL — used when the local server can't receive the callback
// (e.g., user copies the code from the browser). This matches Claude CLI's
// MANUAL_REDIRECT_URL from the production OAuth config.
export const MANUAL_REDIRECT_URL =
  process.env.ANTHROPIC_MANUAL_REDIRECT_URL ||
  "https://platform.claude.com/oauth/code/callback";

// All OAuth scopes — union of Console + Claude.ai scopes, matching Claude CLI's
// ALL_OAUTH_SCOPES. Includes org:create_api_key for Console -> Claude.ai redirect.
export const SCOPES =
  process.env.ANTHROPIC_SCOPES ||
  "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload user:plugins";

export const CLI_BUILD_ID =
  process.env.ANTHROPIC_CLI_BUILD_ID || "835";

export const ENTRYPOINT =
  process.env.ANTHROPIC_ENTRYPOINT ||
  process.env.CLAUDE_CODE_ENTRYPOINT ||
  "sdk-cli";

// Beta flags matching Claude Code 2.1.x (confirmed via request interception).
// oauth-2025-04-20 is added separately only for OAuth auth.
//
// 2026-10-16: added thinking-binding-controls-2026-08-01 (present in Claude
// Code ≥2.1.2xx). It legalizes the `thinking.block_binding` request field,
// which OpenCode v2 sends natively for models flagged
// supportsThinkingBlockBinding (claude-haiku-5-5, opus/sonnet/fable 5.x).
// Without this flag those requests fail validation with:
//   "thinking.adaptive.block_binding: Extra inputs are not permitted"
// 2026-11-09: checked against Claude Code 2.1.319. Every flag still exists;
// the CLI sends advisor-tool, thinking-binding-controls and extended-cache-ttl
// only conditionally (advisor model set, block_binding sent, 1h cache TTL).
// To refresh this list against a newer Claude Code install:
//   grep -aoE '"[a-z]+(-[a-z]+)*-20[0-9]{2}-[0-9]{2}-[0-9]{2}"' \
//     "$(readlink -f "$(command -v claude)")" | sort -u
export const BETA_FLAGS =
  process.env.ANTHROPIC_BETA_FLAGS ||
  "interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,advisor-tool-2026-03-01,effort-2025-11-24,extended-cache-ttl-2025-04-11,thinking-binding-controls-2026-08-01";

export const OAUTH_BETA_FLAG = "oauth-2025-04-20";

// Exact anthropic-version header from Claude Code SDK
export const ANTHROPIC_VERSION = "2023-06-01";

// Stainless SDK headers (matches what Claude Code's bundled SDK sends)
export const STAINLESS_HEADERS: Record<string, string> = {
  "x-stainless-lang": "js",
  "x-stainless-package-version": process.env.ANTHROPIC_SDK_VERSION || "0.128.0",
  "x-stainless-os": process.platform === "darwin" ? "MacOS" : process.platform === "linux" ? "Linux" : "Windows",
  "x-stainless-arch": process.arch === "arm64" ? "arm64" : process.arch,
  "x-stainless-runtime": "node",
  "x-stainless-runtime-version": process.version,
};

export const DEFAULT_EFFORT =
  process.env.ANTHROPIC_DEFAULT_EFFORT || "medium";

// Newer models (Opus 4.7+, all 5.x) default thinking.display to "omitted",
// which returns thinking blocks with empty text, so OpenCode has nothing to
// show. "summarized" restores readable reasoning.
export const THINKING_DISPLAY =
  process.env.ANTHROPIC_THINKING_DISPLAY || "summarized";

export const CLEAR_THINKING_TYPE =
  process.env.ANTHROPIC_CLEAR_THINKING_TYPE || "clear_thinking_20251015";

export const BILLING_CCH =
  process.env.ANTHROPIC_BILLING_CCH || "";

export const SESSION_ID =
  process.env.ANTHROPIC_SESSION_ID || "";

export const TOOL_PREFIX = "mcp_";
