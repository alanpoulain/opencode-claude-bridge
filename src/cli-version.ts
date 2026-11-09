import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { credentialsFilePath } from "./keychain.js";
import {
  CLI_VERSION_CACHE_PATH,
  CLI_VERSION_CACHE_TTL_MS,
  CLI_VERSION_FETCH_TIMEOUT_MS,
  CLI_VERSION_TAG,
  CLI_VERSION_URL,
  ENTRYPOINT,
  FALLBACK_CLI_VERSION,
} from "./constants.js";

const VERSION_PATTERN = /^\d+\.\d+\.\d+$/;

type CliVersionSource = "env" | "local" | "cache" | "fallback";

type CliVersionCache = { version: string; fetchedAt: number };

function detectLocalClaudeVersion(): string | null {
  // Installed Claude CLI
  try {
    const version = execFileSync("claude", ["--version"], {
      encoding: "utf8",
      timeout: 3000,
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    const match = version.match(/^(\d+\.\d+\.\d+)/);
    if (match) return match[1];
  } catch {}
  // Credentials file written by the Claude CLI
  try {
    const raw = readFileSync(credentialsFilePath(), "utf8");
    const parsed = JSON.parse(raw);
    if (parsed.version && typeof parsed.version === "string") {
      return parsed.version;
    }
  } catch {}
  return null;
}

export function readCliVersionCache(
  cachePath: string = CLI_VERSION_CACHE_PATH,
): CliVersionCache | null {
  try {
    const parsed = JSON.parse(readFileSync(cachePath, "utf8"));
    if (
      typeof parsed?.version === "string" &&
      VERSION_PATTERN.test(parsed.version) &&
      typeof parsed.fetchedAt === "number"
    ) {
      return { version: parsed.version, fetchedAt: parsed.fetchedAt };
    }
  } catch {}
  return null;
}

/** Extracts a version for `tag` from an npm dist-tags response body. */
export function parseDistTagsVersion(body: unknown, tag: string = CLI_VERSION_TAG): string | null {
  if (!body || typeof body !== "object") return null;
  const version = (body as Record<string, unknown>)[tag];
  return typeof version === "string" && VERSION_PATTERN.test(version) ? version : null;
}

// Priority: env override → local Claude install → cached npm version → fallback.
function resolveInitialCliVersion(): { version: string; source: CliVersionSource } {
  if (process.env.ANTHROPIC_CLI_VERSION) {
    return { version: process.env.ANTHROPIC_CLI_VERSION, source: "env" };
  }
  const local = detectLocalClaudeVersion();
  if (local) return { version: local, source: "local" };
  const cached = readCliVersionCache();
  if (cached) return { version: cached.version, source: "cache" };
  return { version: FALLBACK_CLI_VERSION, source: "fallback" };
}

let cliVersionState = resolveInitialCliVersion();

export function getCliVersion(): string {
  return cliVersionState.version;
}

export function getUserAgent(): string {
  return process.env.ANTHROPIC_USER_AGENT ||
    `claude-cli/${getCliVersion()} (external, ${ENTRYPOINT})`;
}

export type FetchCliVersionOptions = {
  fetch?: typeof globalThis.fetch;
  cachePath?: string;
  now?: number;
  tag?: string;
};

/**
 * Returns the published Claude Code version for the configured dist-tag,
 * served from a 24h on-disk cache when fresh. Falls back to a stale cache
 * entry, then null. Never throws.
 */
export async function fetchCliVersion(
  options: FetchCliVersionOptions = {},
): Promise<string | null> {
  const cachePath = options.cachePath ?? CLI_VERSION_CACHE_PATH;
  const now = options.now ?? Date.now();
  const cached = readCliVersionCache(cachePath);
  if (cached && now - cached.fetchedAt < CLI_VERSION_CACHE_TTL_MS) {
    return cached.version;
  }
  try {
    const doFetch = options.fetch ?? globalThis.fetch;
    const response = await doFetch(CLI_VERSION_URL, {
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(CLI_VERSION_FETCH_TIMEOUT_MS),
    });
    if (response.ok) {
      const version = parseDistTagsVersion(await response.json(), options.tag);
      if (version) {
        try {
          mkdirSync(dirname(cachePath), { recursive: true });
          writeFileSync(cachePath, JSON.stringify({ version, fetchedAt: now }));
        } catch {}
        return version;
      }
    }
  } catch {}
  return cached?.version ?? null;
}

/**
 * Updates the CLI version from the npm registry unless an env override or a
 * local Claude install already provided one.
 */
export async function refreshCliVersion(
  options: FetchCliVersionOptions = {},
): Promise<string> {
  if (cliVersionState.source === "env" || cliVersionState.source === "local") {
    return cliVersionState.version;
  }
  const version = await fetchCliVersion(options);
  if (version) cliVersionState = { version, source: "cache" };
  return cliVersionState.version;
}
