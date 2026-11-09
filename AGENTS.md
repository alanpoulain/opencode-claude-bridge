# opencode-claude-bridge

## Architecture

- Only the OpenCode ≥ 2.0 plugin contract (`id` + `setup`) is supported. The V1
  `server()` implementation (auth.loader fetch wrapper, auth.methods, model
  injection, 429 refresh-retry) was removed; OpenCode 1.x users stay on the
  1.11.x releases.
- Never rotate Claude refresh tokens outside the integration `refresh`
  callback — they are single-use and only that callback's result is persisted.
- `TOOL_NAME_MAP` (OpenCode → Claude) is the single source of truth for tool
  names; `INBOUND_TOOL_NAME_MAP` is derived from it.

## Testing

- `npm test` builds with `tsc` and runs only `dist/index.test.js`.
