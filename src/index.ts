import { createHash, randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { hostname, userInfo } from "node:os";
import { join } from "node:path";
import {
  createAuthorizationRequest,
  exchangeCodeForTokens,
  parseAuthCode,
  refreshTokens,
  startOAuthCallbackServer,
} from "./oauth.js";
import type { OAuthTokens } from "./oauth.js";
import {
  getClaudeTokens,
  getLastKeychainError,
  readClaudeCliTokens,
  writeBackClaudeCredentials,
} from "./keychain.js";
import type { ClaudeCliTokens, CredentialSource } from "./keychain.js";
import {
  BETA_FLAGS,
  OAUTH_BETA_FLAG,
  ANTHROPIC_VERSION,
  BILLING_CCH,
  STAINLESS_HEADERS,
  PROFILE_URL,
  CLI_BUILD_ID,
  ENTRYPOINT,
  DEFAULT_EFFORT,
  CLEAR_THINKING_TYPE,
  THINKING_DISPLAY,
  SESSION_ID,
  PLUGIN_ID,
  MANUAL_REDIRECT_URL,
} from "./constants.js";
import { getCliVersion, getUserAgent, refreshCliVersion } from "./cli-version.js";
import {
  getClaudeTools,
  translateArgsOpencodeToClaude,
  translateToolArgsJsonString,
  computeFingerprint,
  extractFirstUserMessageText,
  shouldUseClaudeToolSchemas,
} from "./claude-tools.js";
import type { ToolDefinition } from "./claude-tools.js";
import { createSseProcessor } from "./stream.js";

// ── Types ──────────────────────────────────────────────────────────

type OAuthProfile = {
  account?: { uuid?: string };
};

type ClaudeSystemPromptCache = {
  system?: Array<{ type?: string; text?: string; cache_control?: unknown }>;
  billing?: {
    ccVersion?: string;
    ccVersionSuffix?: string;
    ccEntrypoint?: string;
  };
};

const TOOL_NAME_MAP: Record<string, string> = {
  bash: "Bash",
  read: "Read",
  glob: "Glob",
  grep: "Grep",
  edit: "Edit",
  write: "Write",
  task: "Agent",
  webfetch: "WebFetch",
  todowrite: "TodoWrite",
  skill: "Skill",
  question: "AskUserQuestion",
  plan_enter: "EnterPlanMode",
  plan_exit: "ExitPlanMode",
};

const INBOUND_TOOL_NAME_MAP: Record<string, string> = Object.fromEntries(
  Object.entries(TOOL_NAME_MAP).map(([opencode, claude]) => [claude, opencode]),
);

// ── Helpers ────────────────────────────────────────────────────────

const CLAUDE_PREFIX =
  "You are a Claude agent, built on Anthropic's Claude Agent SDK.";

/** Every identity line Claude Code may send as its own system block. */
const CLAUDE_IDENTITY_LINES = [
  "You are Claude Code, Anthropic's official CLI for Claude.",
  "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.",
  CLAUDE_PREFIX,
];

/**
 * Derive a human-readable display name from a Claude model ID.
 *
 * Matches `claude-{family}-{major}-{minor}[-{date}]` (the convention used
 * for every Claude model family to date) and renders e.g.
 *   claude-opus-4-7            -> "Opus 4.7"
 *   claude-haiku-4-5-20251001  -> "Haiku 4.5"
 *   claude-sonnet-4-6          -> "Sonnet 4.6"
 *
 * Falls back to the raw model ID if the convention doesn't match, so future
 * naming changes are surfaced truthfully instead of silently wrong.
 */
export function deriveModelDisplayName(modelId: string): string {
  const m = modelId.match(/^claude-([a-z]+)-(\d+)-(\d+)(?:-\d+)?$/i);
  if (!m) return modelId;
  const [, family, major, minor] = m;
  const capitalized = family.charAt(0).toUpperCase() + family.slice(1).toLowerCase();
  return `${capitalized} ${major}.${minor}`;
}

/**
 * Rewrite model-identity lines inside the cached Claude Code system prompt
 * so they reflect the model actually being requested. Without this, every
 * request (regardless of selected model) gets a system prompt claiming
 * "You are powered by the model named Sonnet 4.6", which can bias behavior.
 */
export function rewriteSystemBlocksForModel(
  blocks: Array<{ type?: string; text?: string }>,
  modelId: string | undefined,
): Array<{ type?: string; text?: string }> {
  if (!modelId) return blocks;
  const display = deriveModelDisplayName(modelId);

  return blocks.map((block) => {
    if (block?.type !== "text" || typeof block.text !== "string") return block;
    let text = block.text;

    text = text.replace(
      /You are powered by the model named [^\n]+? The exact model ID is [a-z0-9.-]+\./g,
      `You are powered by the model named ${display}. The exact model ID is ${modelId}.`,
    );

    return { ...block, text };
  });
}

/**
 * Build the outbound system prompt when a captured Claude Code prompt is
 * cached: billing header, then the Claude Code blocks, then OpenCode's own
 * system blocks (which carry AGENTS.md, environment info and skills). Any
 * billing block already present in OpenCode's system is dropped so the
 * header appears exactly once. When the Claude Code blocks already carry
 * CLAUDE_PREFIX, the copy the `context` hook prepended to OpenCode's system
 * is stripped so the identity line isn't sent twice.
 */
export function mergeSystemBlocks(
  billingHeader: string,
  claudeSystem: Array<{ type?: string; text?: string }>,
  opencodeSystem: Array<{ type?: string; text?: string }>,
): Array<{ type?: string; text?: string }> {
  const claudeHasPrefix = claudeSystem.some(
    (block) => typeof block?.text === "string" && block.text.includes(CLAUDE_PREFIX),
  );
  const opencode = opencodeSystem
    .filter((block) => !block?.text?.startsWith("x-anthropic-billing-header:"))
    .map((block) => {
      if (!claudeHasPrefix || typeof block?.text !== "string") return block;
      if (block.text.startsWith(`${CLAUDE_PREFIX}\n\n`)) {
        return { ...block, text: block.text.slice(`${CLAUDE_PREFIX}\n\n`.length) };
      }
      if (block.text === CLAUDE_PREFIX) return { ...block, text: "" };
      return block;
    })
    .filter((block) => block?.text !== "");
  return [
    { type: "text", text: billingHeader },
    ...claudeSystem,
    ...opencode,
  ];
}

/**
 * Lay the system prompt out the way Claude Code 2.x does: the billing header,
 * then the identity line as its own block, then everything else joined with
 * "\n" into one block. An identity line leading a larger block is split off.
 * Existing cache markers are dropped; applyCacheBreakpoints sets them on the
 * identity and final blocks, as Claude Code does.
 */
export function collapseSystemBlocks(
  system: Array<{ type?: string; text?: string; cache_control?: unknown }>,
): Array<{ type: string; text: string }> {
  let billing: string | undefined;
  let identity: string | undefined;
  const rest: string[] = [];
  for (const block of system) {
    let text = block?.text;
    if (typeof text !== "string" || text === "") continue;
    if (billing === undefined && text.startsWith("x-anthropic-billing-header:")) {
      billing = text;
      continue;
    }
    if (identity === undefined) {
      const line = CLAUDE_IDENTITY_LINES.find((l) => text === l || text!.startsWith(`${l}\n`));
      if (line) {
        identity = line;
        text = text.slice(line.length).replace(/^\n+/, "");
        if (text === "") continue;
      }
    }
    rest.push(text);
  }
  const out: Array<{ type: string; text: string }> = [];
  if (billing !== undefined) out.push({ type: "text", text: billing });
  if (identity !== undefined) out.push({ type: "text", text: identity });
  if (rest.length > 0) out.push({ type: "text", text: rest.join("\n") });
  return out;
}

/**
 * Session request kinds that reach the model. OpenCode dispatches a separate
 * hook per kind, so a prefix registered only on `context` would leave
 * compaction, generate and title requests without it.
 */
export const SYSTEM_PROMPT_HOOKS = ["context", "compaction", "generate", "title"] as const;

/** Prepend CLAUDE_PREFIX to the first system block, once. */
export function injectClaudePrefix(system: Array<{ type?: string; text?: string }>): void {
  if (system.some((s) => typeof s?.text === "string" && s.text.includes(CLAUDE_PREFIX))) return;
  if (system.length > 0) {
    system[0] = { ...system[0], text: `${CLAUDE_PREFIX}\n\n${system[0].text ?? ""}` };
  } else {
    system.push({ type: "text", text: CLAUDE_PREFIX });
  }
}

export function stripSystemCacheControl(
  system: Array<{ type?: string; text?: string; cache_control?: unknown }>,
): Array<{ type?: string; text?: string }> {
  return system.map((block) => {
    if (!("cache_control" in block)) return block;
    const { cache_control, ...rest } = block;
    return rest;
  });
}

/** Anthropic rejects requests with more than this many cache breakpoints. */
export const MAX_CACHE_BREAKPOINTS = 4;

/**
 * Cap prompt-cache breakpoints at MAX_CACHE_BREAKPOINTS instead of stripping
 * them, so caching keeps working:
 *
 * - The last non-empty system block always gets a breakpoint: it caches tools
 *   + the whole (stable) system prompt, the biggest invariant prefix.
 * - A Claude Code identity-line block gets one too, as Claude Code marks it.
 * - Over the limit, the earliest markers are dropped first. The system
 *   breakpoints above and the latest message markers (which carry the cache
 *   from one turn to the next) survive. A top-level `cache_control` uses one
 *   slot.
 * - Added breakpoints use ttl "1h" when any other marker does, since a 1h
 *   marker may not follow a 5m one.
 */
export function applyCacheBreakpoints(parsed: any): void {
  type Holder = Record<string, unknown>;
  const isObject = (v: unknown): v is Holder => !!v && typeof v === "object";
  const tools: Holder[] = Array.isArray(parsed.tools) ? parsed.tools.filter(isObject) : [];
  const system: Holder[] = Array.isArray(parsed.system) ? parsed.system.filter(isObject) : [];
  const messageBlocks: Holder[] = [];
  if (Array.isArray(parsed.messages)) {
    for (const msg of parsed.messages) {
      if (Array.isArray(msg?.content)) messageBlocks.push(...msg.content.filter(isObject));
    }
  }
  // Prefix order: tools → system → messages.
  const ordered = [...tools, ...system, ...messageBlocks];
  const marked = () => ordered.filter((b) => b.cache_control != null);

  const systemEnd = [...system]
    .reverse()
    .find((b) => b.type === "text" && typeof b.text === "string" && b.text.length > 0);
  const identity = system.find(
    (b) => b.type === "text" && CLAUDE_IDENTITY_LINES.includes(b.text as string),
  );
  const pinned = [identity, systemEnd].filter((b): b is Holder => !!b);
  const usesHourTtl = marked().some(
    (b) => isObject(b.cache_control) && b.cache_control.ttl === "1h",
  ) || (isObject(parsed.cache_control) && parsed.cache_control.ttl === "1h");
  for (const block of pinned) {
    if (block.cache_control != null) continue;
    block.cache_control = usesHourTtl ? { type: "ephemeral", ttl: "1h" } : { type: "ephemeral" };
  }

  const budget = MAX_CACHE_BREAKPOINTS - (parsed.cache_control != null ? 1 : 0);
  const current = marked();
  let excess = current.length - budget;
  for (const block of current) {
    if (excess <= 0) break;
    if (pinned.includes(block)) continue;
    delete block.cache_control;
    excess--;
  }
}

export function shouldInjectClaudeTools(input: {
  model?: string;
  requestUrl?: string;
  tools?: unknown;
}): boolean {
  if (!shouldUseClaudeToolSchemas({ model: input.model, requestUrl: input.requestUrl })) {
    return false;
  }
  return Array.isArray(input.tools) && input.tools.length > 0;
}

function getToolName(tool: unknown): string | undefined {
  if (!tool || typeof tool !== "object") return undefined;
  const name = (tool as { name?: unknown }).name;
  return typeof name === "string" ? name : undefined;
}

function getToolInputSchema(tool: unknown): Record<string, unknown> | undefined {
  if (!tool || typeof tool !== "object") return undefined;
  const inputSchema = (tool as { input_schema?: unknown }).input_schema;
  if (!inputSchema || typeof inputSchema !== "object" || Array.isArray(inputSchema)) return undefined;
  return inputSchema as Record<string, unknown>;
}

export function getClaudeToolsForActiveOpenCodeTools(
  tools: unknown,
): ToolDefinition[] {
  if (!Array.isArray(tools)) return [];
  const claudeToolsByName = new Map(getClaudeTools().map((tool) => [tool.name, tool]));
  const selectedToolsByName = new Map<string, ToolDefinition>();

  for (const tool of tools) {
    const name = getToolName(tool);
    if (!name) continue;

    const claudeName = TOOL_NAME_MAP[name];
    if (claudeName) {
      const claudeTool = claudeToolsByName.get(claudeName);
      if (!claudeTool) continue;
      selectedToolsByName.set(claudeName, claudeTool);
      continue;
    }

    const inputSchema = getToolInputSchema(tool);
    if (!inputSchema) {
      console.error(
        `[opencode-claude-bridge] Dropping active tool "${name}" — missing or invalid input_schema`,
      );
      continue;
    }

    const description = (tool as { description?: unknown }).description;
    selectedToolsByName.set(name, {
      name,
      description: typeof description === "string" ? description : "",
      input_schema: inputSchema,
    });
  }

  return Array.from(selectedToolsByName.values()).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
}

export function getInboundToolNameMapForActiveOpenCodeTools(
  tools: unknown,
): Record<string, string> {
  if (!Array.isArray(tools)) return {};
  const inboundToolNameMap: Record<string, string> = {};

  for (const tool of tools) {
    const name = getToolName(tool);
    if (!name) continue;
    const claudeName = TOOL_NAME_MAP[name];
    if (claudeName) inboundToolNameMap[claudeName] = name;
  }

  return inboundToolNameMap;
}

const oauthProfileCache = new Map<string, Promise<OAuthProfile | null>>();
const SYSTEM_PROMPT_CACHE_PATH = process.env.ANTHROPIC_SYSTEM_PROMPT_PATH
  || join(process.env.HOME || "", ".cache", "opencode-claude-bridge", "claude-system-prompt.json");

export type RefreshDeps = {
  /** Claude CLI tokens regardless of expiry, with their store. */
  readCli: () => ClaudeCliTokens | null;
  refresh: (refreshToken: string) => OAuthTokens;
  /** Compare-and-swap write-back; true when the store holds the fresh tokens. */
  writeBack: (source: CredentialSource, spentRefreshToken: string, fresh: OAuthTokens) => boolean;
  warn?: (message: string) => void;
};

/**
 * Layered token refresh: valid Claude CLI access token → stored refresh token
 * → CLI refresh token. Returns the fresh token set, or null when no layer was
 * available; throws the last error when every attempted refresh failed.
 *
 * Claude refresh tokens are single-use, and a credential imported from the
 * CLI shares its refresh token with it. Whenever the token spent is the CLI's,
 * the rotation is written back to the CLI store (compare-and-swap on the spent
 * token) so the CLI isn't left holding a dead token. OpenCode persists the
 * returned tokens itself.
 */
export function refreshCredentialLayered(
  auth: { refresh?: string },
  deps: RefreshDeps,
): OAuthTokens | null {
  const strip = (t: ClaudeCliTokens): OAuthTokens => ({
    access: t.access,
    refresh: t.refresh,
    expires: t.expires,
  });
  const readCli = (): ClaudeCliTokens | null => {
    try {
      return deps.readCli();
    } catch {
      return null;
    }
  };
  // Not a type guard: narrowing on it would turn the invalid branch into `never`.
  const isValid = (t: OAuthTokens | null): boolean => !!t && t.expires > Date.now() + 60_000;

  // Layer 1: the CLI already holds a valid access token — adopt it, no network.
  const cli = readCli();
  if (cli && isValid(cli)) return strip(cli);

  // Layers 2 and 3: stored refresh token, then the CLI's if it differs.
  const candidates = [...new Set([auth.refresh, cli?.refresh].filter((t): t is string => !!t))];
  const errors: string[] = [];
  for (const token of candidates) {
    let fresh: OAuthTokens;
    try {
      fresh = deps.refresh(token);
    } catch (err) {
      errors.push(String(err));
      // The CLI or another OpenCode may have just rotated this token; adopt
      // its result rather than failing.
      const again = readCli();
      if (again && isValid(again) && again.refresh !== token) return strip(again);
      continue;
    }
    if (cli && token === cli.refresh && !deps.writeBack(cli.source, token, fresh)) {
      deps.warn?.(
        "[opencode-claude-bridge] Refreshed the Claude CLI's token but could not write it back; the Claude CLI may need `claude login`.",
      );
    }
    return fresh;
  }

  if (errors.length > 0) throw new Error(errors[errors.length - 1]);
  return null;
}

/** Best-effort browser open — tries multiple methods on each platform. */
function openBrowser(url: string): void {
  (async () => {
    const { execFileSync } = await import("node:child_process");

    // [command, args, timeoutMs] — tried in order until one succeeds.
    const attempts: Array<[string, string[], number]> =
      process.platform === "darwin"
        ? [
            ["/bin/launchctl", ["asuser", String(process.getuid?.() ?? ""), "/usr/bin/open", url], 5000],
            ["/usr/bin/open", [url], 3000],
            ["/usr/bin/osascript", ["-e", `open location "${url}"`], 3000],
          ]
        : process.platform === "win32"
          ? [["cmd", ["/c", "start", "", url], 3000]]
          : [
              ["/usr/bin/xdg-open", [url], 3000],
              ["/usr/bin/open", [url], 3000],
            ];

    for (const [command, args, timeout] of attempts) {
      try { execFileSync(command, args, { timeout }); break; }
      catch {}
    }
  })().catch(() => {});
}

/** Build deduplicated beta flags string. */
function buildBetaFlags(existing: string): string {
  const required = BETA_FLAGS.split(",").map((b) => b.trim()).filter(Boolean);
  const withoutSpecials = required.filter(
    (beta) => beta !== "claude-code-20250219" && beta !== OAUTH_BETA_FLAG,
  );
  return ["claude-code-20250219", OAUTH_BETA_FLAG, ...withoutSpecials].join(",");
}

/** Deduplicate repeated Claude Code prefix in a text block. */
function deduplicatePrefix(text: string): string {
  const doubled = `${CLAUDE_PREFIX}\n\n${CLAUDE_PREFIX}`;
  while (text.includes(doubled)) {
    text = text.replace(doubled, CLAUDE_PREFIX);
  }
  return text;
}

export function mapOutboundToolName(name: string | undefined): string | undefined {
  if (!name) return name;
  return TOOL_NAME_MAP[name] || name;
}

function maybeUnquoteText(text: string): string {
  const trimmed = text.trim();
  if (!trimmed.startsWith("\"")) return text;
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed === "string") return parsed;
  } catch {
    // Expected for non-JSON text — don't log.
  }
  return text;
}

export function stripAssistantPrefillForClaude(
  messages: Array<{ role?: string; content?: unknown }>,
): Array<{ role?: string; content?: unknown }> {
  if (messages.length === 0) return messages;
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") return messages;
  if (typeof last.content !== "string") return messages;
  if (last.content.trim() !== "Continue with your tasks.") return messages;
  return messages.slice(0, -1);
}

function normalizeSystemBlocks(
  system: Array<{ type?: string; text?: string }>,
): Array<{ type?: string; text?: string }> {
  return system.map((item) => {
    if (item.type === "text" && item.text) {
      return {
        ...item,
        text: deduplicatePrefix(
          item.text
            .replace(/OpenCode/g, "Claude Code")
            .replace(/\/\.config\/opencode/g, "/.claude")
            .replace(/opencode/gi, "claude"),
        ),
      };
    }
    return item;
  });
}

function readCachedClaudePromptCache(): ClaudeSystemPromptCache | null {
  try {
    const raw = readFileSync(SYSTEM_PROMPT_CACHE_PATH, "utf8");
    const parsed = JSON.parse(raw) as ClaudeSystemPromptCache;
    if (!Array.isArray(parsed.system) || parsed.system.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

function getStableDeviceId(): string {
  let who = "unknown";
  try {
    who = `${userInfo().username}@${hostname()}`;
  } catch {
    try {
      who = `${process.env.USER || process.env.USERNAME || "unknown"}@${hostname()}`;
    } catch {}
  }
  return createHash("sha256").update(who).digest("hex");
}

function buildBillingHeader(
  sessionId: string,
  cachedClaudePromptCache: ClaudeSystemPromptCache | null,
  dynamicFingerprint?: string,
): string {
  const cch = BILLING_CCH
    || createHash("sha256").update(sessionId).digest("hex").slice(0, 5);
  // Prefer dynamic fingerprint (computed per-request from first user message)
  // over static cached value, matching Claude Code's actual behavior.
  const suffix = dynamicFingerprint
    || cachedClaudePromptCache?.billing?.ccVersionSuffix
    || CLI_BUILD_ID;
  const ccVersion = cachedClaudePromptCache?.billing?.ccVersion
    || `${getCliVersion()}.${suffix}`;
  const ccEntrypoint = cachedClaudePromptCache?.billing?.ccEntrypoint || ENTRYPOINT;
  return `x-anthropic-billing-header: cc_version=${ccVersion}; cc_entrypoint=${ccEntrypoint}; cch=${cch};`;
}

async function fetchOAuthProfile(accessToken: string): Promise<OAuthProfile | null> {
  const cached = oauthProfileCache.get(accessToken);
  if (cached) return cached;

  const pending = (async () => {
    try {
      const response = await globalThis.fetch(PROFILE_URL, {
        headers: {
          authorization: `Bearer ${accessToken}`,
          "content-type": "application/json",
          "user-agent": getUserAgent(),
          "anthropic-beta": OAUTH_BETA_FLAG,
        },
      });
      if (!response.ok) return null;
      return await response.json() as OAuthProfile;
    } catch {
      return null;
    }
  })();

  oauthProfileCache.set(accessToken, pending);
  return pending;
}

/**
 * Pick the OAuth access token for an Anthropic request, or null to pass the
 * request through untouched. The Claude CLI token is a fallback only when
 * nothing signals API-key auth: an `x-api-key` header or an active non-OAuth
 * connection means the user chose an API key, and taking that request over
 * would silently bill it to the subscription.
 */
export async function resolveOAuthAccessToken(
  request: Request,
  deps: {
    getActiveCredential: () => Promise<
      { type?: string; access?: string; expires?: number } | undefined
    >;
    getCliTokens: () => { access: string; expires: number } | null;
  },
): Promise<string | null> {
  const bearer = request.headers?.get?.("authorization");
  if (typeof bearer === "string" && bearer.startsWith("Bearer ") && bearer.length > 7) {
    return bearer.slice(7);
  }
  if (request.headers?.get?.("x-api-key")) return null;

  try {
    const credential = await deps.getActiveCredential();
    if (credential) {
      if (credential.type !== "oauth") return null;
      if (
        credential.access &&
        typeof credential.expires === "number" &&
        credential.expires > Date.now()
      ) {
        return credential.access;
      }
    }
  } catch {}

  try {
    const tokens = deps.getCliTokens();
    if (tokens?.access && tokens.expires > Date.now()) return tokens.access;
  } catch {}
  return null;
}

/** Append why the Keychain was unusable, when that's the likely cause. */
export function withKeychainHint(message: string, keychainError = getLastKeychainError()): string {
  return keychainError ? `${message} (Claude CLI credentials: ${keychainError})` : message;
}

function bearerOf(request: Request): string | undefined {
  const auth = request.headers.get("authorization");
  return auth?.startsWith("Bearer ") ? auth.slice(7) : undefined;
}

/**
 * Swap an access token Anthropic already rejected for the Claude CLI's
 * current one. Read-only: nothing is refreshed here.
 */
export function replaceRevokedToken(
  token: string,
  revoked: ReadonlySet<string>,
  getCliTokens: () => { access: string } | null,
): string {
  if (!revoked.has(token)) return token;
  try {
    const cli = getCliTokens();
    if (cli?.access && !revoked.has(cli.access)) return cli.access;
  } catch {}
  return token;
}

/**
 * On a 401, re-send the request once with the Claude CLI's current access
 * token when it differs from the one rejected — the CLI may have rotated the
 * shared credential mid-session. Read-only on credentials: nothing is
 * refreshed here (refresh tokens are single-use; only the integration
 * `refresh` callback rotates them). Returns the original response when no
 * retry applies.
 */
export async function retryUnauthorizedWithCliToken(
  request: Request,
  response: Response,
  deps: {
    getCliTokens: () => { access: string } | null;
    fetch: (request: Request) => Promise<Response>;
  },
): Promise<Response> {
  if (response.status !== 401) return response;
  let cli: { access: string } | null = null;
  try {
    cli = deps.getCliTokens();
  } catch {}
  if (!cli?.access || cli.access === bearerOf(request)) return response;

  const headers = new Headers(request.headers);
  headers.set("authorization", `Bearer ${cli.access}`);
  const body = request.method === "GET" || request.method === "HEAD"
    ? null
    : await request.clone().text();
  const retried = await deps.fetch(
    new Request(request.url, { method: request.method, headers, body, signal: request.signal }),
  );
  await response.body?.cancel().catch(() => {});

  // fetch already decoded the body; drop headers describing the wire encoding.
  const outHeaders = new Headers(retried.headers);
  outHeaders.delete("content-encoding");
  outHeaders.delete("content-length");
  return new Response(retried.body, {
    status: retried.status,
    statusText: retried.statusText,
    headers: outHeaders,
  });
}

// ── Request/response phases (used by the http.request / http.response
// hooks) ───────────────────────────────────────────────────────────

type AnthropicRequestState = {
  sessionId: string;
  deviceId: string;
  cachedClaudePromptCache: ClaudeSystemPromptCache | null;
  cachedClaudeSystem: NonNullable<ClaudeSystemPromptCache["system"]> | null;
};

/** /messages → /v1/messages and ?beta=true, matching Claude Code's URLs. */
function normalizeAnthropicApiUrl(rawUrl: string): string {
  let requestUrl: URL | null = null;
  try {
    requestUrl = new URL(rawUrl);
  } catch {}

  if (requestUrl?.pathname === "/messages") {
    requestUrl.pathname = "/v1/messages";
  }
  if (requestUrl?.pathname === "/v1/messages" && !requestUrl.searchParams.has("beta")) {
    requestUrl.searchParams.set("beta", "true");
  }
  return requestUrl?.toString() ?? rawUrl;
}

/**
 * Whether the bridge should inject adaptive thinking (plus context
 * management and effort) for a model. Haiku 4.x and Sonnet before 4.6
 * reject the field; Opus 4.x, Sonnet ≥ 4.6 and every 5.x family (Opus,
 * Sonnet, Haiku, Fable, Mythos) accept it.
 */
export function supportsAdaptiveThinking(model: unknown): boolean {
  if (typeof model !== "string") return false;
  // Minor is at most two digits so a date suffix (claude-opus-4-20250514)
  // is not mistaken for it.
  const m = model.toLowerCase().match(/claude-([a-z]+)-(\d+)(?:-(\d{1,2}))?(?!\d)/);
  if (!m) return false;
  const family = m[1];
  const major = Number(m[2]);
  const minor = Number(m[3] ?? 0);
  if (major >= 5) return true;
  if (major !== 4) return false;
  if (family === "opus") return true;
  if (family === "sonnet") return minor >= 6;
  return false;
}

/**
 * Inject adaptive thinking for supported models, request readable thinking
 * text (newer models omit it by default), and force temperature=1 when
 * thinking is on. Returns whether the model supports adaptive thinking.
 */
export function applyThinkingConfig(parsed: any): boolean {
  const supportsThinking = supportsAdaptiveThinking(parsed.model);
  if (!parsed.thinking && supportsThinking) {
    parsed.thinking = { type: "adaptive" };
  }

  const thinkingType = parsed.thinking?.type;
  const thinkingOn = thinkingType === "enabled" || thinkingType === "adaptive";
  if (thinkingOn && parsed.thinking.display === undefined && THINKING_DISPLAY) {
    parsed.thinking.display = THINKING_DISPLAY;
  }

  // Anthropic requires temperature=1 when thinking is enabled/adaptive
  if (thinkingOn && parsed.temperature !== undefined && parsed.temperature !== 1) {
    parsed.temperature = 1;
  }
  return supportsThinking;
}

/**
 * Mutate a parsed /v1/messages body in place to match Claude Code's wire
 * format: adaptive thinking, context management, billing metadata, system
 * prompt shaping, Claude tool schemas, and outbound tool-name/argument
 * translation. Returns the inbound tool-name map to use when translating the
 * response (differs from the static map when Claude schemas were injected).
 */
async function transformClaudeRequestBody(
  parsed: any,
  opts: AnthropicRequestState & { requestUrl: string; accessToken: string },
): Promise<{ inboundToolNameMap: Record<string, string> }> {
  let inboundToolNameMap = INBOUND_TOOL_NAME_MAP;

  const supportsThinking = applyThinkingConfig(parsed);

  if (!parsed.system) parsed.system = [];
  if (Array.isArray(parsed.messages)) {
    parsed.messages = stripAssistantPrefillForClaude(parsed.messages);
  }
  // context_management and output_config.effort are only
  // supported by thinking-capable models. Sending them to
  // haiku (used for title generation) produces a 400.
  if (supportsThinking) {
    if (!parsed.context_management) {
      parsed.context_management = {
        edits: [{ type: CLEAR_THINKING_TYPE, keep: "all" }],
      };
    }
    if (!parsed.output_config && DEFAULT_EFFORT) {
      parsed.output_config = { effort: DEFAULT_EFFORT };
    }
  }

  const profile = opts.accessToken
    ? await fetchOAuthProfile(opts.accessToken)
    : null;
  if (!parsed.metadata) parsed.metadata = {};
  if (!parsed.metadata.user_id && profile?.account?.uuid) {
    parsed.metadata.user_id = JSON.stringify({
      device_id: opts.deviceId,
      account_uuid: profile.account.uuid,
      session_id: opts.sessionId,
    });
  }

  // Compute dynamic fingerprint from the first user message
  // (matches Claude Code's per-request fingerprint behavior).
  let dynamicFingerprint: string | undefined;
  if (parsed.messages && Array.isArray(parsed.messages)) {
    const firstText = extractFirstUserMessageText(parsed.messages);
    if (firstText) {
      dynamicFingerprint = computeFingerprint(firstText);
    }
  }

  // Inject billing header as first system block (required for OAuth).
  const hasBilling = parsed.system.some(
    (s: { text?: string }) =>
      s.text?.startsWith("x-anthropic-billing-header:"),
  );
  let billingHeader = buildBillingHeader(opts.sessionId, opts.cachedClaudePromptCache, dynamicFingerprint);
  if (!hasBilling) {
    parsed.system.unshift({
      type: "text",
      text: billingHeader,
    });
  } else {
    billingHeader = parsed.system[0]?.text || billingHeader;
  }

  // If we have a locally captured Claude Code system prompt, lead with it but
  // keep OpenCode's system blocks (AGENTS.md, env, skills) after it.
  if (opts.cachedClaudeSystem) {
    const rewritten = rewriteSystemBlocksForModel(
      opts.cachedClaudeSystem,
      typeof parsed.model === "string" ? parsed.model : undefined,
    );
    parsed.system = mergeSystemBlocks(billingHeader, rewritten, parsed.system);
  } else if (parsed.system && Array.isArray(parsed.system)) {
    // Otherwise keep the existing prompt but shape it closer to Claude Code.
    parsed.system = normalizeSystemBlocks(parsed.system);
  }
  parsed.system = collapseSystemBlocks(parsed.system);
  // Keep OpenCode's default tool schemas for non-Claude targets.
  // Claude wire schemas are only useful for Anthropic-native
  // requests or Claude-family models on Anthropic-compatible
  // routers such as OpenRouter.
  if (shouldInjectClaudeTools({ model: parsed.model, requestUrl: opts.requestUrl, tools: parsed.tools })) {
    inboundToolNameMap = getInboundToolNameMapForActiveOpenCodeTools(parsed.tools);
    parsed.tools = getClaudeToolsForActiveOpenCodeTools(parsed.tools);
  }
  delete parsed.tool_choice;

  // Normalize message content blocks:
  // - Translate tool names (OpenCode → Claude Code)
  // - Translate tool_use input arguments (camelCase → snake_case)
  // - Clean up text blocks
  if (parsed.messages && Array.isArray(parsed.messages)) {
    for (const msg of parsed.messages) {
      if (!Array.isArray(msg.content)) continue;
      for (const block of msg.content) {
        if (block.type === "text" && typeof block.text === "string") {
          block.text = maybeUnquoteText(block.text);
        }
        if (block.type === "tool_use" && block.name) {
          block.name = mapOutboundToolName(block.name);
          // Consolidated OpenCode → Claude argument translation:
          // key renames, Agent/AskUserQuestion/Skill/WebFetch/TodoWrite
          // field bridging. Mirrors translateToolArgsJsonString
          // (inbound) so round-trips preserve meaning. See
          // translateArgsOpencodeToClaude in claude-tools.ts.
          if (block.input && typeof block.input === "object") {
            block.input = translateArgsOpencodeToClaude(
              block.name,
              block.input as Record<string, unknown>,
            );
          }
        }
        // tool_result blocks reference tool names via tool_use_id,
        // no name translation needed here.
      }
    }
  }

  applyCacheBreakpoints(parsed);

  return { inboundToolNameMap };
}

/**
 * Stream a response through the SSE processor so Claude-style tool names and
 * arguments are translated back to OpenCode on assembled content_block_stop
 * payloads. Buffers input_json_delta fragments per content-block index, which
 * avoids chunk-boundary corruption that regex-on-raw-bytes can't handle.
 */
function wrapAnthropicResponseWithSse(
  response: Response,
  inboundToolNameMap: Record<string, string>,
): Response {
  if (!response.body) return response;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const processor = createSseProcessor({
    inboundToolNameMap,
    translateToolArgs: translateToolArgsJsonString,
  });

  return new Response(
    new ReadableStream({
      start(controller) {
        // Use start() with an async loop rather than pull():
        // Bun's ReadableStream doesn't reliably re-invoke pull()
        // when the handler resolves without enqueuing (which
        // happens while we're buffering input_json_delta
        // fragments), so pull() can stall the stream.
        (async () => {
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                const tail = processor.flush();
                if (tail) controller.enqueue(encoder.encode(tail));
                controller.close();
                return;
              }
              const out = processor.feedChunk(decoder.decode(value, { stream: true }));
              if (out) controller.enqueue(encoder.encode(out));
            }
          } catch (e) {
            controller.error(e);
          }
        })();
      },
    }),
    { status: response.status, statusText: response.statusText, headers: response.headers },
  );
}

// ── Plugin ─────────────────────────────────────────────────────────
// opencode ≥ 2.0 loads plugins through `id` + `setup(ctx)`. The bridge hooks
// into it as follows:
//
//   integration.transform        OAuth method override on `anthropic` with
//                                authorize + refresh callbacks (API keys go
//                                through native /connect untouched)
//   session.hook("context")      Claude Code system prompt prefix
//   session.hook("http.request") header/body/URL rewrite to Claude Code's
//                                wire format
//   session.hook("http.response") SSE tool-name/argument translation back
//                                to OpenCode
//
// Notes:
//   - No model injection: the models.dev catalog already lists Claude models
//     and a stored OAuth credential activates the provider.
//   - No inline 429 refresh-retry: Claude refresh tokens are single-use, so
//     rotating them anywhere but the integration `refresh` callback (whose
//     result opencode persists) would invalidate the stored token. opencode's
//     native retry policy handles 429 backoff.
//   - Rewrites apply only to anthropic.com hosts, since the session hooks
//     observe every request.

type OAuthCredential = {
  type: "oauth";
  methodID: string;
  access: string;
  refresh: string;
  expires: number;
};

type OAuthAuthorization = {
  url: string;
  instructions: string;
} & (
  | { mode: "auto"; callback: Promise<OAuthCredential> }
  | { mode: "code"; callback: (code: string) => Promise<OAuthCredential> }
);

// Structural subset of the plugin context used here (kept local so the
// package gains no runtime dependency on @opencode/plugin).
type PluginContext = {
  integration: {
    transform: (
      editor: (editor: {
        method: {
          update(registration: {
            integrationID: string;
            method: { id: string; type: "oauth"; label: string };
            authorize: () => Promise<OAuthAuthorization>;
            refresh: (
              credential: OAuthCredential,
            ) => Promise<OAuthCredential>;
          }): void;
        };
      }) => void,
    ) => Promise<unknown>;
    connection: {
      active: (integrationID: string) => Promise<unknown>;
      resolve: (
        connection: unknown,
      ) => Promise<
        | {
            type?: string;
            access?: string;
            refresh?: string;
            expires?: number;
          }
        | undefined
      >;
    };
  };
  session: {
    hook: (
      name: string,
      callback: (event: any) => void | Promise<void>,
      options?: { providerID?: string },
    ) => Promise<unknown>;
  };
};

const OpenCodeClaudeBridge = {
  async setup(ctx: PluginContext) {
    // Not awaited: a slow registry must never delay startup.
    void refreshCliVersion();
    const cachedClaudePromptCache = readCachedClaudePromptCache();
    const requestState: AnthropicRequestState = {
      sessionId: SESSION_ID || randomUUID(),
      deviceId: getStableDeviceId(),
      cachedClaudePromptCache,
      cachedClaudeSystem: cachedClaudePromptCache?.system || null,
    };
    // Per-request inbound tool-name map, computed during the request rewrite
    // (differs from the static map when Claude schemas were injected). Keyed
    // weakly on the rewritten Request, which the response event carries.
    // Presence in this map is also what marks a request as OAuth-rewritten,
    // so concurrent API-key traffic can't flip a shared flag.
    const inboundMaps = new WeakMap<Request, Record<string, string>>();
    // Access tokens Anthropic answered 401 for while the CLI held a working
    // one. OpenCode keeps sending its stored token until its expiry triggers a
    // refresh, so later requests swap it up front instead of paying a 401 +
    // retry each time.
    const revokedTokens = new Set<string>();

    const toOAuthCredential = (
      tokens: { access: string; refresh: string; expires: number },
    ): OAuthCredential => ({
      type: "oauth",
      methodID: "oauth",
      access: tokens.access,
      refresh: tokens.refresh,
      expires: Math.trunc(tokens.expires),
    });

    const isAnthropicApiUrl = (url: string): boolean => {
      try {
        return new URL(url).host.endsWith("anthropic.com");
      } catch {
        return false;
      }
    };

    // Bearer already on the request (opencode injects it from the stored
    // credential, refreshing via the registered callback below) → stored
    // credential → Claude CLI keychain. Never rotates refresh tokens here:
    // Claude refresh tokens are single-use, so only the integration `refresh`
    // callback (whose result opencode persists) may rotate them.
    const resolveAccessToken = (request: Request): Promise<string | null> =>
      resolveOAuthAccessToken(request, {
        getActiveCredential: async () => {
          const connection = await ctx.integration.connection.active("anthropic");
          return connection ? ctx.integration.connection.resolve(connection) : undefined;
        },
        getCliTokens: () => getClaudeTokens(),
      });

    // ── Auth: take over the native `oauth` method on the anthropic
    // integration so stored credentials refresh through the bridge's layered
    // refresh instead of opencode's generic OAuth refresh.
    await ctx.integration.transform((editor) => {
      editor.method.update({
        integrationID: "anthropic",
        method: { id: "oauth", type: "oauth", label: "Claude Pro / Max (OAuth)" },
        async authorize(): Promise<OAuthAuthorization> {
          // First try: auto-bootstrap from Claude CLI keychain
          try {
            const tokens = getClaudeTokens();
            if (tokens) {
              return {
                url: "",
                instructions:
                  "Connected via Claude CLI keychain — no browser needed.\nIf Anthropic models aren't visible, restart OpenCode.",
                mode: "auto",
                callback: Promise.resolve(toOAuthCredential(tokens)),
              };
            }
          } catch {}
          const keychainError = getLastKeychainError();
          const keychainNote = keychainError
            ? `Claude CLI credentials unavailable: ${keychainError}.\n\n`
            : "";

          // Start a local HTTP server to capture the OAuth redirect. This
          // gives us an ephemeral port so the redirect_uri is
          // http://localhost:{port}/callback — matching Claude Code's flow.
          let callbackServer: Awaited<
            ReturnType<typeof startOAuthCallbackServer>
          > | null = null;
          try {
            callbackServer = await startOAuthCallbackServer();
          } catch (err) {
            console.error(`[opencode-oauth] Failed to start callback server: ${err}`);
          }

          if (callbackServer) {
            const { url, manualUrl, verifier, state, port } = callbackServer;
            const redirectUri = `http://localhost:${port}/callback`;
            // Best-effort browser open
            openBrowser(url);
            // Race: localhost callback vs manual code paste. If the browser
            // redirects to localhost, the callback server captures the code
            // automatically; otherwise the user pastes a code from the
            // manual-flow URL.
            let autoCode: string | null = null;
            callbackServer.waitForCode().then((code) => {
              autoCode = code;
            }).catch(() => {});
            return {
              url,
              instructions: `${keychainNote}Opening browser for OAuth...\n\nIf the browser redirected back successfully, just press Enter.\nOtherwise, open this URL and paste the code:\n${manualUrl}`,
              mode: "code",
              callback: async (manualInput: string) => {
                try {
                  // Brief pause to let the auto-flow resolve if it just arrived
                  if (!autoCode && !manualInput?.trim()) {
                    await new Promise((r) => setTimeout(r, 500));
                  }
                  if (autoCode) {
                    // Auto-flow captured the code via localhost redirect
                    const tokens = exchangeCodeForTokens(autoCode, verifier, {
                      redirectUri,
                      state,
                    });
                    callbackServer?.close();
                    return toOAuthCredential(tokens);
                  }
                  const parsed = parseAuthCode(manualInput);
                  if (!parsed.code) {
                    console.error("[opencode-oauth] No authorization code received");
                    throw new Error("No authorization code received");
                  }
                  // Manual flow — redirect_uri defaults to MANUAL_REDIRECT_URL
                  const tokens = exchangeCodeForTokens(parsed.code, verifier, {
                    redirectUri: MANUAL_REDIRECT_URL,
                    state: parsed.state || state,
                  });
                  callbackServer?.close();
                  return toOAuthCredential(tokens);
                } catch (err) {
                  callbackServer?.close();
                  throw err;
                }
              },
            };
          }

          // Fallback: no callback server — use the manual redirect URL flow
          // only (environments where we can't bind localhost).
          const { url, verifier, state } = createAuthorizationRequest();
          openBrowser(url);
          return {
            url,
            instructions: `${keychainNote}Opening browser for OAuth...\n\nAfter authorizing, you'll see a code — paste it below:\n${url}`,
            mode: "code",
            callback: async (code: string) => {
                const parsed = parseAuthCode(code);
                const tokens = exchangeCodeForTokens(parsed.code, verifier, {
                  redirectUri: MANUAL_REDIRECT_URL,
                  state: parsed.state || state,
                });
                return toOAuthCredential(tokens);
              },
          };
        },
        async refresh(credential) {
          // Layered refresh (keychain → stored → CLI). MUST return a promise:
          // opencode awaits it and persists the resolved credential. Never
          // refresh outside this callback — Claude refresh tokens are
          // single-use; spending the CLI's token writes the rotation back.
          let fresh: OAuthTokens | null;
          try {
            fresh = refreshCredentialLayered(credential, {
              readCli: readClaudeCliTokens,
              refresh: refreshTokens,
              writeBack: writeBackClaudeCredentials,
              warn: (message) => console.error(message),
            });
          } catch (err) {
            throw new Error(withKeychainHint(err instanceof Error ? err.message : String(err)));
          }
          if (!fresh) {
            throw new Error(
              withKeychainHint(
                "All Claude OAuth refresh layers failed. Run `claude login` or reconnect 'Claude Pro / Max (OAuth)'.",
              ),
            );
          }
          return toOAuthCredential(fresh);
        },
      });
    });

    // ── System prompt prefix (every request kind that reaches the model) ──
    for (const kind of SYSTEM_PROMPT_HOOKS) {
      await ctx.session.hook(
        kind,
        (event: {
          model?: { providerID?: string };
          system: Array<{ type?: string; text?: string }>;
        }) => {
          if (event.model?.providerID && event.model.providerID !== "anthropic") return;
          injectClaudePrefix(event.system);
        },
        { providerID: "anthropic" },
      );
    }

    // ── Wire rewrite (request phase) ──
    await ctx.session.hook(
      "http.request",
      async (event: { request: Request }) => {
        const request = event.request;
        if (!isAnthropicApiUrl(request.url)) return;
        let accessToken = await resolveAccessToken(request);
        if (!accessToken) return; // API-key or unauthenticated — passthrough
        accessToken = replaceRevokedToken(accessToken, revokedTokens, getClaudeTokens);
        try {
          // ── Headers ──
          const headers = new Headers(request.headers);
          headers.set("authorization", `Bearer ${accessToken}`);
          headers.delete("x-api-key");
          headers.delete("x-session-affinity");
          headers.delete("x-session-id");
          if (!headers.has("accept")) headers.set("accept", "application/json");
          headers.set("user-agent", getUserAgent());
          headers.set("x-app", "cli");
          headers.set("x-claude-code-session-id", requestState.sessionId);
          headers.set("x-client-request-id", randomUUID());
          headers.set("anthropic-version", ANTHROPIC_VERSION);
          headers.set("anthropic-dangerous-direct-browser-access", "true");
          for (const [k, v] of Object.entries(STAINLESS_HEADERS)) headers.set(k, v);
          if (!headers.has("x-stainless-retry-count")) headers.set("x-stainless-retry-count", "0");
          if (!headers.has("x-stainless-timeout")) headers.set("x-stainless-timeout", "600");
          headers.set("anthropic-beta", buildBetaFlags(headers.get("anthropic-beta") || ""));

          // ── Body ──
          let body: string | null = null;
          let inboundToolNameMap = INBOUND_TOOL_NAME_MAP;
          try {
            body = await request.text();
          } catch {
            body = null;
          }
          if (body) {
            try {
              const parsed = JSON.parse(body);
              const result = await transformClaudeRequestBody(parsed, {
                ...requestState,
                requestUrl: request.url,
                accessToken,
              });
              inboundToolNameMap = result.inboundToolNameMap;
              body = JSON.stringify(parsed);
            } catch {}
          }

          const finalUrl = normalizeAnthropicApiUrl(request.url);
          const rewritten = new Request(finalUrl, {
            method: request.method,
            headers,
            body,
            signal: request.signal,
          });
          inboundMaps.set(rewritten, inboundToolNameMap);
          event.request = rewritten;
        } catch (err) {
          console.error(`[opencode-claude-bridge] request rewrite failed, sending original: ${err}`);
        }
      },
      { providerID: "anthropic" },
    );

    // ── SSE tool-name/argument translation (response phase) ──
    await ctx.session.hook(
      "http.response",
      async (event: { request: Request; response: Response }) => {
        // Fallback to our session header covers another plugin replacing the
        // rewritten Request object after us.
        const inboundToolNameMap =
          inboundMaps.get(event.request) ??
          (event.request.headers.get("x-claude-code-session-id") === requestState.sessionId
            ? INBOUND_TOOL_NAME_MAP
            : undefined);
        if (!inboundToolNameMap) return;
        if (event.response.status === 401) {
          try {
            const retried = await retryUnauthorizedWithCliToken(event.request, event.response, {
              getCliTokens: getClaudeTokens,
              fetch: (request) => globalThis.fetch(request),
            });
            if (retried !== event.response) {
              const rejected = bearerOf(event.request);
              if (rejected && retried.status !== 401) revokedTokens.add(rejected);
              event.response = retried;
            }
          } catch (err) {
            console.error(`[opencode-claude-bridge] 401 retry failed: ${err}`);
          }
        }
        try {
          event.response = wrapAnthropicResponseWithSse(event.response, inboundToolNameMap);
        } catch (err) {
          console.error(`[opencode-claude-bridge] response wrap failed: ${err}`);
        }
      },
      { providerID: "anthropic" },
    );

    return () => {};
  },
};

/**
 * opencode plugin module.
 *
 * `id` + `setup` is the entrypoint opencode ≥ 2.0 requires. A bare function
 * default export makes loaders fall back to
 * their legacy path, which treats every named export of this module as a
 * plugin factory and invokes it with `(input, options)` — crashing on the
 * helpers exported here for tests.
 */
export default {
  id: PLUGIN_ID,
  setup: OpenCodeClaudeBridge.setup,
};
