import { execFileSync } from "node:child_process";
import { chmodSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { homedir, userInfo } from "node:os";
import { join } from "node:path";
import type { OAuthTokens } from "./oauth.js";

interface KeychainCredentials {
  claudeAiOauth: {
    accessToken: string;
    refreshToken: string;
    expiresAt: number;
    scopes?: string[];
    subscriptionType?: string;
    rateLimitTier?: string;
  };
}

/** Where the Claude CLI credentials were read from, so a refresh can be written back. */
export type CredentialSource =
  | { kind: "keychain"; service: string; account?: string }
  | { kind: "file"; path: string };

export type ClaudeCliTokens = OAuthTokens & { source: CredentialSource };

const KEYCHAIN_SERVICE = "Claude Code-credentials";

function credentialsFilePath(): string {
  return join(homedir(), ".claude", ".credentials.json");
}

/**
 * Pick the first candidate that actually carries `claudeAiOauth`.
 *
 * More than one Keychain item can share the service name, and a match is not
 * the same thing as the right item: attaching an MCP server writes a second
 * item under this service holding only `mcpOAuth`. Selecting on content rather
 * than on "the first thing `security` returned" is what keeps that item from
 * shadowing the real credential.
 *
 * Exported for tests; the selection is pure so it can be exercised without a
 * Keychain.
 */
export function selectClaudeCredentials(
  candidates: Array<KeychainCredentials | null>,
): KeychainCredentials | null {
  for (const candidate of candidates) {
    if (candidate?.claudeAiOauth) return candidate;
  }
  return null;
}

/** Raw contents of one Keychain item, optionally scoped to an account. */
function readKeychainRaw(account?: string): string | null {
  const args = ["find-generic-password", "-s", KEYCHAIN_SERVICE];
  if (account) args.push("-a", account);
  args.push("-w");
  try {
    // execFileSync, not a shell string: the account name is user-controlled
    // and must not be interpolated into a command line.
    const raw = execFileSync("security", args, {
      encoding: "utf8",
      timeout: 5000,
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return raw || null;
  } catch {
    // Item missing or Keychain unavailable.
    return null;
  }
}

/** Read one Keychain item, optionally scoped to an account. */
function readKeychainItem(account?: string): KeychainCredentials | null {
  const raw = readKeychainRaw(account);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as KeychainCredentials;
  } catch {
    return null;
  }
}

function currentUsername(): string | undefined {
  try {
    return userInfo().username;
  } catch {
    // userInfo() throws when there is no passwd entry for the uid.
    return undefined;
  }
}

/**
 * Read Claude CLI credentials and where they came from: the macOS Keychain,
 * falling back to ~/.claude/.credentials.json (the only store elsewhere).
 */
export function readClaudeCredentialsWithSource():
  | { credentials: KeychainCredentials; source: CredentialSource }
  | null {
  if (process.platform === "darwin") {
    const account = currentUsername();
    // Scoped lookup first (Claude Code stores under the current account), then
    // the historical unscoped lookup so anyone whose item lives under a
    // different account name keeps working exactly as before.
    const scoped = account ? readKeychainItem(account) : null;
    const unscoped = readKeychainItem();
    const found = selectClaudeCredentials([scoped, unscoped]);
    if (found) {
      return {
        credentials: found,
        source: {
          kind: "keychain",
          service: KEYCHAIN_SERVICE,
          account: found === scoped ? account : undefined,
        },
      };
    }
    // Deliberately fall through when the Keychain had matches but none carried
    // claudeAiOauth. Returning such an item made the credentials-file fallback
    // below unreachable.
  }

  try {
    const path = credentialsFilePath();
    const credentials = JSON.parse(readFileSync(path, "utf8")) as KeychainCredentials;
    return { credentials, source: { kind: "file", path } };
  } catch {
    return null;
  }
}

/**
 * Read Claude CLI credentials from macOS Keychain.
 * Falls back to ~/.claude/.credentials.json on other platforms.
 */
export function readClaudeCredentials(): KeychainCredentials | null {
  return readClaudeCredentialsWithSource()?.credentials ?? null;
}

/** Claude CLI tokens regardless of expiry, with their store. */
export function readClaudeCliTokens(): ClaudeCliTokens | null {
  const found = readClaudeCredentialsWithSource();
  const oauth = found?.credentials.claudeAiOauth;
  if (
    !found ||
    typeof oauth?.accessToken !== "string" ||
    typeof oauth.refreshToken !== "string" ||
    typeof oauth.expiresAt !== "number"
  ) {
    return null;
  }
  return {
    access: oauth.accessToken,
    refresh: oauth.refreshToken,
    // The CLI can store fractional milliseconds; credential expiries are integers.
    expires: Math.trunc(oauth.expiresAt),
    source: found.source,
  };
}

/**
 * Valid (≥ 60s left) OAuth tokens from the Claude CLI, or null. Read-only:
 * Claude refresh tokens are single-use, so refreshing happens only in the
 * integration `refresh` callback, which writes the rotation back.
 */
export function getClaudeTokens(): OAuthTokens | null {
  const tokens = readClaudeCliTokens();
  if (!tokens || tokens.expires <= Date.now() + 60_000) return null;
  return { access: tokens.access, refresh: tokens.refresh, expires: tokens.expires };
}

/**
 * Apply refreshed tokens to a stored credential blob — only if it still holds
 * the refresh token that was spent (compare-and-swap), so a rotation by the
 * CLI or another OpenCode in the meantime is never overwritten. Other fields
 * (scopes, subscriptionType, …) are preserved. Returns null to skip the write.
 */
export function updateCredentialBlob(
  raw: string,
  spentRefreshToken: string,
  fresh: OAuthTokens,
): string | null {
  let parsed: { claudeAiOauth?: Record<string, unknown> } & Record<string, unknown>;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  const oauth = parsed?.claudeAiOauth;
  if (!oauth || typeof oauth !== "object" || oauth.refreshToken !== spentRefreshToken) {
    return null;
  }
  oauth.accessToken = fresh.access;
  oauth.refreshToken = fresh.refresh;
  oauth.expiresAt = fresh.expires;
  return JSON.stringify(parsed);
}

/** Write-back to a credentials file: atomic (temp file + rename), mode 0600. */
export function writeBackCredentialsFile(
  path: string,
  spentRefreshToken: string,
  fresh: OAuthTokens,
): boolean {
  const tmp = `${path}.${process.pid}.tmp`;
  try {
    const updated = updateCredentialBlob(readFileSync(path, "utf8"), spentRefreshToken, fresh);
    if (!updated) return false;
    writeFileSync(tmp, updated, { encoding: "utf8", mode: 0o600 });
    chmodSync(tmp, 0o600);
    renameSync(tmp, path);
    return true;
  } catch {
    try {
      unlinkSync(tmp);
    } catch {}
    return false;
  }
}

/**
 * Command for `security -i` (read from stdin) that updates a Keychain item.
 * Going through stdin keeps the secret out of argv, where `ps` would show it,
 * and `-X` (hex) avoids quoting the JSON. Returns null for names that can't be
 * quoted safely.
 */
export function buildKeychainWriteCommand(
  service: string,
  account: string,
  blob: string,
): string | null {
  const unsafe = /["\\\n\r]/;
  if (unsafe.test(service) || unsafe.test(account)) return null;
  const hex = Buffer.from(blob, "utf8").toString("hex");
  return `add-generic-password -U -s "${service}" -a "${account}" -X ${hex}\n`;
}

/** Account name of the unscoped Keychain item (`"acct"<blob>="…"`). */
function lookupKeychainAccount(service: string): string | null {
  try {
    const out = execFileSync("security", ["find-generic-password", "-s", service], {
      encoding: "utf8",
      timeout: 5000,
      stdio: ["ignore", "pipe", "ignore"],
    });
    return /"acct"<blob>="([^"]*)"/.exec(out)?.[1] ?? null;
  } catch {
    return null;
  }
}

function writeBackKeychain(
  source: { service: string; account?: string },
  spentRefreshToken: string,
  fresh: OAuthTokens,
): boolean {
  try {
    const raw = readKeychainRaw(source.account);
    if (!raw) return false;
    const updated = updateCredentialBlob(raw, spentRefreshToken, fresh);
    if (!updated) return false;
    const account = source.account ?? lookupKeychainAccount(source.service);
    if (!account) return false;
    const command = buildKeychainWriteCommand(source.service, account, updated);
    if (!command) return false;
    execFileSync("security", ["-i"], {
      input: command,
      timeout: 5000,
      stdio: ["pipe", "ignore", "ignore"],
    });
    // `security -i` can exit 0 when the command inside failed; verify.
    const written = readKeychainRaw(account);
    if (!written) return false;
    return (JSON.parse(written) as KeychainCredentials).claudeAiOauth?.refreshToken === fresh.refresh;
  } catch {
    return false;
  }
}

/**
 * Write refreshed tokens back to the Claude CLI's store after spending its
 * refresh token, so the CLI isn't left holding a dead one. Compare-and-swap on
 * the spent token; returns whether the store now holds the fresh tokens.
 */
export function writeBackClaudeCredentials(
  source: CredentialSource,
  spentRefreshToken: string,
  fresh: OAuthTokens,
): boolean {
  if (source.kind === "file") {
    return writeBackCredentialsFile(source.path, spentRefreshToken, fresh);
  }
  return writeBackKeychain(source, spentRefreshToken, fresh);
}
