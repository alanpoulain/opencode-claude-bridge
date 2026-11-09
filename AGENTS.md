# opencode-claude-bridge

## Architecture

- Only the OpenCode ≥ 2.0 plugin contract (`id` + `setup`) is supported. The V1
  `server()` implementation (auth.loader fetch wrapper, auth.methods, model
  injection, 429 refresh-retry) was removed; OpenCode 1.x users stay on the
  1.11.x releases.
- Never rotate Claude refresh tokens outside the integration `refresh`
  callback — they are single-use and only that callback's result is persisted.
- `resolveOAuthAccessToken` only falls back to the Claude CLI token when
  nothing signals API-key auth: an `x-api-key` header or an active non-OAuth
  connection means passthrough, never an OAuth takeover.
- `TOOL_NAME_MAP` (OpenCode → Claude) is the single source of truth for tool
  names; `INBOUND_TOOL_NAME_MAP` is derived from it.
- Only OpenCode ≥ 2.2 tool schemas are supported: read/write/edit take `path`
  (not `filePath`), skill takes `id` (not `name`), todowrite items have no
  `priority`, and read `offset` is 1-based. When OpenCode bumps a minor
  version, re-check the tool input schemas in the installed binary
  (`grep -a -o` for e.g. `oldString:` / `path:t.annotate` in `opencode.exe`)
  — schema drift makes tool calls fail with "Missing key" errors.

- Opus 4.7+ and all 5.x models default `thinking.display` to `"omitted"`
  (empty thinking text), so `applyThinkingConfig` sets `display: "summarized"`
  whenever thinking is on and no display was sent. Model support for injected
  thinking is version-based (`supportsAdaptiveThinking`), not a name list:
  all 5.x families including Haiku get it; Haiku 4.x and Sonnet < 4.6 don't.

- CLI version priority: `ANTHROPIC_CLI_VERSION` → local `claude --version` /
  `~/.claude/.credentials.json` → npm `stable` dist-tag (24h cache in
  `~/.cache/opencode-claude-bridge/cli-version.json`, refreshed non-blocking
  from `setup()`) → hard-coded fallback. That logic lives in
  `src/cli-version.ts`; `src/constants.ts` stays plain values only (no I/O,
  no mutable state). Read it via `getCliVersion()` / `getUserAgent()` at call
  time, never capture it in a module-level const.
  `CLI_BUILD_ID`, `BETA_FLAGS` and the Stainless SDK version are NOT
  auto-updated — npm metadata doesn't expose them.

## Testing

- `npm test` builds with `tsc` and runs only `dist/index.test.js`.
