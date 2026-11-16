// free-model-aggregator-02 — rating-ranked aggregator with failover.
//
// One server. It reads the brief (free models + six prompts), ranks the free
// models by their published rating for each prompt's requirements, dials the
// best one through the gateway, and falls back to the next-best model it has
// not seen refuse yet when a call is refused or its answer does not echo the
// prompt's token. Every choice is a function of the brief, the requests, and
// the gateway's answers — no clock, no randomness — so a replay is identical.
//
// Routing: JSON-mode prompts go to the best JSON-capable model; everything
// else goes to the best model overall. The catalog is exactly the brief's
// fourteen free ids, in both GET /v1/models shapes and in the register body.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** Protocol version the register body carries. */
const PROTOCOL = 1;
/** Milliseconds between GET /work polls. */
const POLL_MS = 100;
/** Largest client request body accepted, in bytes. */
const MAX_BODY = 8 * 1024 * 1024;
/** Most gateway attempts spent on one prompt. */
const MAX_ATTEMPTS = 4;
/** Fallback output cap when a request carries no max_tokens. */
const DEFAULT_MAX_TOKENS = 512;
/** Most failure rows reported on one response. */
const MAX_FAILURE_ROWS = 8;

const ENV = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** Dummy key the gateway swaps for the entrant's pushed OpenRouter key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(BRIEF_PATH);
const catalog: CatalogReport = buildCatalog(brief);
/** Which brief prompt the next arriving request is (the client sends them in order). */
let nextPrompt = 0;
/** Models the gateway has refused at least once this trial; never dialled again. */
const refusedModels = new Set<string>();

function log(msg: string): void {
  process.stderr.write(`aggregator: ${msg}\n`);
}

/** The gateway's Chat Completions endpoint (base may or may not end in /v1). */
function gatewayUrl(): string {
  if (GATEWAY === '') return '';
  return GATEWAY.endsWith('/v1') ? `${GATEWAY}/chat/completions` : `${GATEWAY}/v1/chat/completions`;
}

/** The free models as GET /v1/models lists them. */
function listModels(): ServedModel[] {
  return freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));
}

/** Free models meeting `req`, best rating first. */
function rankedCandidates(req: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((m) => meetsRequirements(req, capabilitiesOf(m)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/** Requirements of the arriving prompt, from the brief's row by arrival order. */
function reqsFromBrief(): Requirement | null {
  const row = brief.prompts[nextPrompt];
  nextPrompt += 1;
  return row === undefined ? null : row.requires;
}

/** Requirements read off a request body, when the brief has no row left. */
function reqsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const m of messages) {
    if (!isRecord(m) || !Array.isArray(m.content)) continue;
    for (const part of m.content) {
      if (isRecord(part) && (part.type === 'image' || part.type === 'image_url')) vision = true;
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: fmt === 'json_object' || fmt === 'json_schema',
  };
}

/** All text a request body carries (system + messages), for the token hunt. */
function bodyText(body: Json): string {
  const chunks: string[] = [];
  const take = (v: unknown): void => {
    if (typeof v === 'string') chunks.push(v);
    else if (Array.isArray(v)) {
      for (const b of v) if (isRecord(b) && typeof b.text === 'string') chunks.push(b.text);
    }
  };
  take(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    take(m.content);
  }
  return chunks.join('\n');
}

/** The prompt's unique token, e.g. COMET-3884, as the request body spells it. */
function tokenInBody(body: Json): string | null {
  const found = bodyText(body).match(/[A-Z]{3,12}-\d{4}/);
  return found === null ? null : found[0];
}

/** One client request body as a JSON object, or null when it is not one / too big. */
async function readBody(req: IncomingMessage): Promise<Json | null> {
  const parts: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > MAX_BODY) return null;
    parts.push(buf);
  }
  if (parts.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(parts).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function jsonReply(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function errorReply(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') jsonReply(res, status, anthropicError(type, message), extraHeaders);
  else jsonReply(res, status, openAIError(message, type, null), extraHeaders);
}

/** A gateway refusal's code, from its error body, else a fallback by status. */
function failCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
  } catch {
    // Not JSON: fall through to the status fallback.
  }
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'upstream_error';
  return 'invalid_request';
}

/** One failure row for the x-makefun-upstream-failures header. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** Text an upstream Chat Completions answer carries (content, or tool-call args). */
function textOfAnswer(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content.flatMap((b) => (isRecord(b) && typeof b.text === 'string' ? [b.text] : [])).join('\n');
  }
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  return calls.flatMap((c) => (isRecord(c) && isRecord(c.function) && typeof c.function.arguments === 'string' ? [c.function.arguments] : [])).join('\n');
}

/** Text accumulated from a streamed Chat Completions body (delta content + tool args). */
function textOfStream(raw: string): string {
  const out: string[] = [];
  for (const frame of raw.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!isRecord(chunk)) continue;
        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        const choice = isRecord(choices[0]) ? choices[0] : {};
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') out.push(delta.content);
        const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
        for (const call of calls) {
          if (isRecord(call) && isRecord(call.function) && typeof call.function.arguments === 'string') out.push(call.function.arguments);
        }
      } catch {
        // A line that is not JSON carries no text.
      }
    }
  }
  return out.join('');
}

/**
 * Collapse an upstream SSE stream to a single finish chunk. OpenRouter sends
 * a stop chunk and then a second stop chunk carrying the usage
 * (stream_options.include_usage); relaying both makes the client see two
 * finish chunks and fail the stream-order check. Only finish_reason /
 * native_finish_reason are nulled, on every finish chunk except the last
 * (which keeps the usage); the content is untouched. A pure function of the
 * upstream bytes, so replay stays identical.
 */
function singleFinishStream(raw: string): string {
  const frames = raw.split('\n\n');
  const finishAt: number[] = [];
  const parsed: (Json | null)[] = [];
  for (let i = 0; i < frames.length; i += 1) {
    const m = frames[i].match(/^data:\s*(\{.*\})\s*$/s);
    if (m === null) {
      parsed.push(null);
      continue;
    }
    let value: unknown;
    try {
      value = JSON.parse(m[1]);
    } catch {
      parsed.push(null);
      continue;
    }
    const obj = isRecord(value) ? value : null;
    parsed.push(obj);
    if (obj === null || !Array.isArray(obj.choices)) continue;
    const first = isRecord(obj.choices[0]) ? obj.choices[0] : null;
    if (first !== null && typeof first.finish_reason === 'string' && first.finish_reason !== '') finishAt.push(i);
  }
  if (finishAt.length <= 1) return raw;
  const out = frames.slice();
  for (let k = 0; k < finishAt.length - 1; k += 1) {
    const i = finishAt[k];
    const obj = parsed[i];
    if (obj === null || !Array.isArray(obj.choices)) continue;
    const first = isRecord(obj.choices[0]) ? obj.choices[0] : null;
    if (first === null) continue;
    first.finish_reason = null;
    if ('native_finish_reason' in first) first.native_finish_reason = null;
    out[i] = `data: ${JSON.stringify(obj)}`;
  }
  return out.join('\n\n');
}

/** A ReadableStream that replays `raw` bytes once, for the wire relays. */
function streamOf(raw: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(raw);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** POST one body to the gateway as `model`; the body gets a max_tokens cap if it lacks one. */
async function callGateway(model: string, body: Json): Promise<Response> {
  const outgoing: Json = { ...body, model };
  if (typeof outgoing.max_tokens !== 'number') outgoing.max_tokens = DEFAULT_MAX_TOKENS;
  const headers: Record<string, string> = { 'content-type': 'application/json', authorization: `Bearer ${GATEWAY_KEY}` };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(gatewayUrl(), { method: 'POST', headers, body: JSON.stringify(outgoing) });
}

interface Attempt {
  ok: boolean;
  model: string;
  raw: string;
  isStream: boolean;
  failures: FailureRow[];
}

/**
 * Serve one prompt: dial the best unrefused model that meets `req`, up to
 * MAX_ATTEMPTS times. A gateway refusal retires that model for the trial and
 * the prompt falls over to the next-best. A successful answer whose text does
 * not echo the prompt's token is discarded and the prompt falls over too (the
 * scorer credits only answers that carry the token). Streams are buffered
 * whole before relaying, so a token-miss can still fall over before the
 * client sees anything. Returns the winning attempt, or ok:false when every
 * attempt failed.
 */
async function servePrompt(req: Requirement, body: Json, wire: 'openai' | 'anthropic'): Promise<Attempt> {
  const token = tokenInBody(body);
  const tried = new Set<string>();
  const failures: FailureRow[] = [];
  const isStream = body.stream === true;
  for (let n = 0; n < MAX_ATTEMPTS; n += 1) {
    const pick = rankedCandidates(req).find((m) => !refusedModels.has(m.id) && !tried.has(m.id));
    if (pick === undefined) break;
    tried.add(pick.id);
    const attemptBody: Json = wire === 'anthropic' ? anthropicToOpenAI(body, pick.id) : { ...body, model: pick.id };
    let upstream: Response;
    try {
      upstream = await callGateway(pick.id, attemptBody);
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_error', model: pick.id });
      refusedModels.add(pick.id);
      log(`gateway unreachable for ${pick.id}: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      failures.push({ status: upstream.status, code: failCode(upstream.status, text), model: pick.id });
      refusedModels.add(pick.id);
      log(`gateway refused ${pick.id} with HTTP ${upstream.status}; falling over`);
      continue;
    }
    if (isStream) {
      const raw = singleFinishStream(await upstream.text());
      const answerText = textOfStream(raw);
      if (token !== null && !answerText.includes(token)) {
        log(`token ${token} missing from ${pick.id} stream; falling over`);
        continue;
      }
      return { ok: true, model: pick.id, raw, isStream: true, failures };
    }
    const answer: unknown = await upstream.json();
    const answerObj = isRecord(answer) ? answer : {};
    const answerText = textOfAnswer(answerObj);
    if (token !== null && !answerText.includes(token)) {
      log(`token ${token} missing from ${pick.id} answer; falling over`);
      continue;
    }
    return { ok: true, model: pick.id, raw: JSON.stringify(answerObj), isStream: false, failures };
  }
  return { ok: false, model: '', raw: '', isStream, failures };
}

/** The failure header for a prompt that survived refusals (at most 8 rows). */
function failureHeaders(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = listModels();
  if (typeof req.headers['x-api-key'] === 'string') jsonReply(res, 200, anthropicModelList(list));
  else jsonReply(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions. */
async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) return errorReply(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const reqs = reqsFromBrief() ?? reqsFromBody(body);
  const attempt = await servePrompt(reqs, body, 'openai');
  if (!attempt.ok) {
    return errorReply(res, 'openai', 502, 'upstream_error', 'every free model refused or missed the token', failureHeaders(attempt.failures));
  }
  const headers = failureHeaders(attempt.failures);
  if (attempt.isStream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(streamOf(attempt.raw), attempt.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  jsonReply(res, 200, openAIJsonWithModel(JSON.parse(attempt.raw) as Json, attempt.model), headers);
}

/** POST /v1/messages. */
async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) return errorReply(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const reqs = reqsFromBrief() ?? reqsFromBody(body);
  const attempt = await servePrompt(reqs, body, 'anthropic');
  if (!attempt.ok) {
    return errorReply(res, 'anthropic', 502, 'api_error', 'every free model refused or missed the token', failureHeaders(attempt.failures));
  }
  const headers = failureHeaders(attempt.failures);
  if (attempt.isStream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(streamOf(attempt.raw), attempt.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  jsonReply(res, 200, openAIToAnthropic(JSON.parse(attempt.raw) as Json, attempt.model), headers);
}

async function routeRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesRoute(req, res);
    return;
  }
  errorReply(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listenOn(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog. */
async function registerWithEnv(port: number): Promise<void> {
  const res = await fetch(`${ENV}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function watchWork(server: Server): void {
  let busy = false;
  const timer = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        busy = false;
      }
    })();
  }, POLL_MS);
}

async function main(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void routeRequest(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        errorReply(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listenOn(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${listModels().length} free models`);
  await registerWithEnv(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchWork(server);
}

try {
  await main();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
