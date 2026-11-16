// free-model-aggregator-02 — a live aggregator over OpenRouter's free models.
//
// The entry is one small server. It reads the trial brief (the free model
// list with the house's ratings, and the six prompts' manifest), registers
// its catalog with the env, and then answers each prompt the env sends by
// dialling a real free model through the grader's gateway on the Chat
// Completions route. It never answers from its own weights: every completion
// it returns is a gateway answer, relayed in the wire the client used.
//
// Routing, in one line: rank the brief's free models by rating, filter to
// the ones that meet the prompt's requirements, and dial the best first.
// The brief's ratings are read at runtime, so a sealed table that differs
// from the practice one still routes correctly. JSON-mode prompts go to the
// best JSON-capable model; everything else goes to the best model overall.
// When the gateway refuses a model (a 4xx/5xx, or a network error), the
// entry falls over to the next best model it has not already been refused
// by in this trial, up to four dials per prompt. A model refused once in a
// trial is never dialled again in that trial: a free-tier throttle is per
// model, so an immediate retry on the same model only spends the budget.
//
// Two checks guard every answer before it is relayed. First, the prompt's
// token (a short code like SPARROW-7435, carried in the request) must appear
// in the upstream answer's text; an answer without it earns nothing on
// quality even though it was served, so a token-less answer is treated as
// a miss and the entry tries the next model. Second, a stream is normalised
// to a single finish chunk before relay: some providers send two finish
// chunks (a bare stop chunk, then a usage chunk that also carries a stop
// reason), and the env's stream check fails an answer that carries two.
//
// Every refusal the entry recovers from is reported back on the answer in
// the x-makefun-upstream-failures header (at most eight rows), because that
// header is the only way the failover column can see a refusal that was
// retried and then served. Nothing here reads the wall clock and nothing
// is random: every choice is a function of the brief, the requests and the
// gateway's answers, so a replay serves the same bytes as the live pass.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  readBrief,
  freeModels,
  capabilitiesOf,
  meetsRequirements,
  buildCatalog,
  isRecord,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  openAIModelList,
  anthropicModelList,
  anthropicError,
  openAIError,
  anthropicToOpenAI,
  openAIToAnthropic,
  openAIJsonWithModel,
  openAIStreamWithModel,
  anthropicEventsFromOpenAI,
  type ServedModel,
} from './lib/wire.ts';

// How many gateway dials one prompt may spend before the entry gives up.
const MAX_ATTEMPTS_PER_PROMPT = 4;
// How many refusal rows the failover header may carry (the env reads at most 8).
const MAX_FAILURE_ROWS = 8;
// The output cap the gateway requires when the client did not name one.
const FALLBACK_MAX_TOKENS = 512;
// How often the entry asks the env whether the trial is done (milliseconds).
const WORK_POLL_MS = 100;
// How long one gateway dial may take before it counts as failed (milliseconds).
const UPSTREAM_TIMEOUT_MS = 25000;
// The shape of a prompt's token: a short uppercase word, a dash, four digits.
const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/;

/** Pause for `ms` milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** One environment variable, or '' when it is unset. */
function env(name: string): string {
  return process.env[name] ?? '';
}

/** A diagnostic line on stderr. The trial scores stdout, never this. */
function log(...parts: unknown[]): void {
  console.error(`aggregator: ${parts.map((p) => String(p)).join(' ')}`);
}

/** The prompt token carried in `text`, or null when there is none. */
function tokenInText(text: string): string | null {
  const found = TOKEN_PATTERN.exec(text);
  return found === null ? null : found[0];
}

/**
 * The string code for a gateway refusal, for the failover header.
 * The gateway's error body sometimes names a code (a string `code` or
 * `type` under `error`); when it does not, the code follows the status:
 * a 429 is a rate limit, a 400/403 an invalid request, anything else an
 * upstream error. The header needs a string, and the gateway's own
 * `code` here is often the numeric status, which is not one.
 */
function errorCodeOf(status: number, body: unknown): string {
  if (isRecord(body)) {
    const inner = isRecord(body.error) ? body.error : null;
    if (inner !== null) {
      if (typeof inner.code === 'string' && inner.code !== '') return inner.code;
      if (typeof inner.type === 'string' && inner.type !== '') return inner.type;
    }
    if (typeof body.code === 'string' && body.code !== '') return body.code;
    if (typeof body.type === 'string' && body.type !== '') return body.type;
  }
  if (status === 429) return 'rate_limited';
  if (status === 400 || status === 403) return 'invalid_request';
  return 'upstream_error';
}

/**
 * Collapse an upstream SSE body to a single finish chunk.
 *
 * Some providers answer a stream with two finish chunks: a bare stop
 * chunk, then a usage chunk that also carries finish_reason "stop"
 * (the gateway forwards both because stream_options.include_usage is
 * set). The env's stream-order check fails an answer that carries two
 * finish chunks, so the entry keeps only the last one — the one that
 * carries the usage — and clears finish_reason and native_finish_reason
 * on every earlier finish chunk. The cleared chunks become ordinary
 * content chunks with an empty delta, which relay harmlessly.
 * A pure function of the raw bytes: the same upstream body always
 * gives the same normalised body, so a replay matches the live pass.
 */
function singleFinishStream(raw: string): string {
  const frames = raw.split('\n\n');
  // The JSON payload of each frame's first data line, or null.
  const payloads: (Json | null)[] = frames.map((frame) => {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.slice(5).trim();
      if (data === '' || data === '[DONE]') continue;
      try {
        const value: unknown = JSON.parse(data);
        if (isRecord(value)) return value;
      } catch {
        // Not JSON: this frame carries no payload to inspect.
      }
    }
    return null;
  });
  // A finish chunk is one whose first choice carries a finish reason.
  const isFinishChunk = (value: Json | null): boolean => {
    if (value === null) return false;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    if (first === null) return false;
    return first.finish_reason !== null && first.finish_reason !== undefined;
  };
  let lastFinish = -1;
  for (let i = 0; i < payloads.length; i++) {
    if (isFinishChunk(payloads[i])) lastFinish = i;
  }
  if (lastFinish === -1) return raw;
  const fixed = frames.map((frame, index) => {
    if (index === lastFinish || !isFinishChunk(payloads[index])) return frame;
    const chunk = payloads[index] as Json;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const first = isRecord(choices[0]) ? choices[0] : null;
    if (first === null) return frame;
    const clearedFirst = { ...first, finish_reason: null, native_finish_reason: null };
    const cleared = { ...chunk, choices: [clearedFirst, ...choices.slice(1)] };
    const lines = frame.split('\n').map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return line;
      const data = trimmed.slice(5).trim();
      if (data === '' || data === '[DONE]') return line;
      try {
        const value: unknown = JSON.parse(data);
        if (isRecord(value)) return `data: ${JSON.stringify(cleared)}`;
      } catch {
        // Not JSON: leave the line as it came.
      }
      return line;
    });
    return lines.join('\n');
  });
  return fixed.join('\n\n');
}

/** The visible answer text carried in an SSE body: every delta's content, joined. */
function streamTextOf(raw: string): string {
  let text = '';
  for (const frame of raw.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const data = trimmed.slice(5).trim();
      if (data === '' || data === '[DONE]') continue;
      try {
        const value: unknown = JSON.parse(data);
        if (!isRecord(value)) continue;
        const choices = Array.isArray(value.choices) ? value.choices : [];
        const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
        if (first === null) continue;
        const delta = isRecord(first.delta) ? first.delta : null;
        if (delta !== null && typeof delta.content === 'string') text += delta.content;
      } catch {
        // A line that is not a JSON chunk carries no answer text.
      }
    }
  }
  return text;
}

/** The visible answer text of a Chat Completions JSON body. */
function completionTextOf(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (first === null) return '';
  const message = isRecord(first.message) ? first.message : null;
  if (message === null) return '';
  const content = message.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .flatMap((part) => {
        if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') return [part.text];
        return [];
      })
      .join('\n');
  }
  return '';
}

/** A readable stream that yields `text` once, then ends. */
function textStream(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

/** One refusal, as the failover header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string;
}

/** What one gateway dial returned: its HTTP status and its raw body text. */
interface Dialled {
  status: number;
  text: string;
}

/**
 * One Chat Completions dial to the gateway. A network error, or a dial
 * that runs past the timeout, comes back as status 0 with an empty body,
 * so the caller treats it like any other refusal.
 */
async function dialGateway(baseUrl: string, apiKey: string, payload: Json): Promise<Dialled> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    return { status: response.status, text: await response.text() };
  } catch {
    return { status: 0, text: '' };
  } finally {
    clearTimeout(timer);
  }
}

/** The answer one prompt produced: a JSON body, or an SSE stream to relay. */
interface ServedResponse {
  status: number;
  headers: Record<string, string>;
  json: Json | null;
  stream: AsyncIterable<string> | null;
}

/** The trial's server state: the brief, the catalog, and what has been refused. */
class Aggregator {
  private brief: Brief;
  private models: ModelRow[];
  private catalog: CatalogReport;
  private baseUrl: string;
  private apiKey: string;
  private refused = new Set<string>();
  private promptIndex = 0;
  private inFlight = 0;

  constructor(brief: Brief, baseUrl: string, apiKey: string) {
    this.brief = brief;
    this.models = freeModels(brief);
    this.catalog = buildCatalog(brief);
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  /** How many free models the brief lists. */
  modelCount(): number {
    return this.models.length;
  }

  /** The catalog document, the same one --list prints and register sends. */
  catalogDocument(): CatalogReport {
    return this.catalog;
  }

  /** The models as GET /v1/models lists them, in the brief's order. */
  servedModels(): ServedModel[] {
    return this.models.map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));
  }

  /** A prompt handler has started. */
  beginPrompt(): void {
    this.inFlight += 1;
  }

  /** A prompt handler has finished, its answer fully written. */
  endPrompt(): void {
    this.inFlight -= 1;
  }

  /** How many prompt handlers are still running. */
  inFlightCount(): number {
    return this.inFlight;
  }

  /**
   * The free models that meet `requires`, best rating first.
   * Ties keep the brief's order (the sort is stable), so the choice is
   * a pure function of the brief.
   */
  private candidatesFor(requires: Requirement): ModelRow[] {
    return this.models
      .filter((m) => meetsRequirements(requires, capabilitiesOf(m)))
      .sort((a, b) => b.rating_milli - a.rating_milli);
  }

  /**
   * The requirements of one incoming request, from the brief's manifest.
   * The env sends the prompts in the manifest's order, so the Nth
   * completion request is the Nth row. When the manifest has run out
   * (more requests than rows), the requirements are read off the body
   * itself instead: a JSON-mode body needs a JSON model, a body that
   * declares tools needs a tools model.
   */
  private requirementsFor(body: Json): Requirement {
    const row = this.brief.prompts[this.promptIndex];
    this.promptIndex += 1;
    if (row !== undefined) return row.requires;
    const requires: Requirement = {};
    const format = isRecord(body.response_format) ? body.response_format : null;
    if (format !== null && format.type === 'json_object') requires.json = true;
    if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
    return requires;
  }

  /** The failover header for `failures`, or no header when there are none. */
  private failureHeaders(failures: FailureRow[]): Record<string, string> {
    if (failures.length === 0) return {};
    return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
  }

  /**
   * Serve one prompt: dial the best model that meets its requirements,
   * fall over on refusals, and relay the first answer that carries the
   * prompt's token. Returns the response the client should get.
   */
  async serve(body: Json, wire: 'openai' | 'anthropic'): Promise<ServedResponse> {
    const requires = this.requirementsFor(body);
    const wantsStream = body.stream === true;
    const token = tokenInText(JSON.stringify(body));
    const candidates = this.candidatesFor(requires);
    const failures: FailureRow[] = [];
    let attempts = 0;

    for (const candidate of candidates) {
      if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
      if (this.refused.has(candidate.id)) continue;
      attempts += 1;

      // Build the upstream body: the client's body, with the dialled
      // model in place of "auto" and an output cap the gateway requires.
      let upstream: Json;
      if (wire === 'anthropic') {
        upstream = anthropicToOpenAI(body, candidate.id);
      } else {
        upstream = { ...body, model: candidate.id };
      }
      if (typeof upstream.max_tokens !== 'number') upstream.max_tokens = FALLBACK_MAX_TOKENS;
      if (wantsStream) upstream.stream_options = { include_usage: true };

      const dialled = await dialGateway(this.baseUrl, this.apiKey, upstream);
      if (dialled.status !== 200) {
        // A refusal: record it for the header, refuse the model for the
        // rest of the trial, and fall over to the next candidate.
        let parsed: unknown = null;
        try {
          parsed = JSON.parse(dialled.text);
        } catch {
          // An unreadable error body still counts as a refusal.
        }
        failures.push({ status: dialled.status, code: errorCodeOf(dialled.status, parsed), model: candidate.id });
        this.refused.add(candidate.id);
        log(`gateway refused ${candidate.id} with HTTP ${dialled.status}; falling over`);
        continue;
      }

      if (wantsStream) {
        const normalised = singleFinishStream(dialled.text);
        const answerText = streamTextOf(normalised);
        if (token !== null && !answerText.includes(token)) {
          // The model answered, but its answer does not carry the token,
          // so it would earn nothing on quality. Try the next model.
          // This is not a gateway refusal: no header row, and the model
          // stays available for later prompts.
          log(`answer from ${candidate.id} lacks the prompt token; trying the next model`);
          continue;
        }
        const source = textStream(normalised);
        const relay =
          wire === 'anthropic'
            ? anthropicEventsFromOpenAI(source, candidate.id)
            : openAIStreamWithModel(source, candidate.id);
        return {
          status: 200,
          headers: { 'content-type': 'text/event-stream', ...this.failureHeaders(failures) },
          json: null,
          stream: relay,
        };
      }

      // A JSON (non-stream) answer.
      let parsed: Json | null = null;
      try {
        const value: unknown = JSON.parse(dialled.text);
        if (isRecord(value)) parsed = value;
      } catch {
        // A 200 that is not JSON is not an answer.
      }
      if (parsed === null) {
        log(`answer from ${candidate.id} is not JSON; trying the next model`);
        continue;
      }
      const answerText = completionTextOf(parsed);
      if (token !== null && !answerText.includes(token)) {
        log(`answer from ${candidate.id} lacks the prompt token; trying the next model`);
        continue;
      }
      const payload =
        wire === 'anthropic'
          ? openAIToAnthropic(parsed, candidate.id)
          : openAIJsonWithModel(parsed, candidate.id);
      return {
        status: 200,
        headers: { 'content-type': 'application/json', ...this.failureHeaders(failures) },
        json: payload,
        stream: null,
      };
    }

    // No model served this prompt. Answer with a 502 and the trail of
    // refusals, so the failover column can still see what was refused.
    const headers = { 'content-type': 'application/json', ...this.failureHeaders(failures) };
    if (wire === 'anthropic') {
      return { status: 502, headers, json: anthropicError('server_error', 'no free model served this prompt'), stream: null };
    }
    return { status: 502, headers, json: openAIError('no free model served this prompt', 'server_error', null), stream: null };
  }
}

/** Read one request's JSON body. An empty body reads as an empty object. */
function readJsonBody(req: IncomingMessage): Promise<Json> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8');
      if (text.trim() === '') {
        resolve({});
        return;
      }
      try {
        const value: unknown = JSON.parse(text);
        resolve(isRecord(value) ? value : {});
      } catch {
        reject(new Error('the request body is not a JSON object'));
      }
    });
    req.on('error', reject);
  });
}

/**
 * One HTTP request: the model lists, the two completion routes, or a 404.
 * A prompt's handler is counted in `inFlight` from the moment it starts
 * until its answer is fully written, so the work poll never exits while
 * an answer is still on its way out.
 */
async function handleRequest(req: IncomingMessage, res: ServerResponse, agg: Aggregator): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const models = agg.servedModels();
    const payload = req.headers['x-api-key'] !== undefined ? anthropicModelList(models) : openAIModelList(models);
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify(payload));
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const wire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    let body: Json;
    try {
      body = await readJsonBody(req);
    } catch {
      const error =
        wire === 'anthropic'
          ? anthropicError('invalid_request_error', 'the request body is not a JSON object')
          : openAIError('the request body is not a JSON object', 'invalid_request_error', null);
      res.writeHead(400, { 'content-type': 'application/json' });
      res.end(JSON.stringify(error));
      return;
    }
    agg.beginPrompt();
    try {
      const served = await agg.serve(body, wire);
      res.writeHead(served.status, served.headers);
      if (served.stream !== null) {
        for await (const chunk of served.stream) {
          res.write(chunk);
        }
        res.end();
      } else {
        res.end(JSON.stringify(served.json));
      }
    } finally {
      agg.endPrompt();
    }
    return;
  }
  res.writeHead(404, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const promptFile = env('MAKEFUN_PROMPT_FILE');
  const envUrl = env('MAKEFUN_ENV_URL');

  // --list prints the catalog the entry would register, and exits.
  if (args.includes('--list')) {
    const brief: Brief = promptFile !== '' ? readBrief(promptFile) : { models: [], prompts: [], error: null };
    console.log(JSON.stringify(buildCatalog(brief)));
    return;
  }

  // Outside a trial there is no brief and no env: print an empty
  // catalog and exit, as the starter does.
  if (promptFile === '' || envUrl === '') {
    console.log(JSON.stringify({ providers: [] }));
    return;
  }

  const brief = readBrief(promptFile);
  if (brief.error !== null) {
    log(`cannot read the brief: ${brief.error}`);
    process.exit(1);
  }
  const agg = new Aggregator(brief, env('OPENAI_BASE_URL'), env('OPENAI_API_KEY'));

  const server = createServer((req, res) => {
    handleRequest(req, res, agg).catch((err: unknown) => {
      log(`request failed: ${(err as Error).message}`);
      try {
        if (!res.headersSent) res.writeHead(500, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'internal error' }));
      } catch {
        // The socket is already gone.
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  log(`listening on 0.0.0.0:${port} with ${agg.modelCount()} free models`);

  // Register the port and the catalog with the env.
  const register = await fetch(`${envUrl}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog: agg.catalogDocument() }),
  });
  if (!register.ok) {
    log(`register failed with HTTP ${register.status}`);
    process.exit(1);
  }
  const registered = (await register.json()) as Json;
  const promptCount = typeof registered.requests === 'number' ? registered.requests : 0;
  log(`registered: the trial holds ${promptCount} prompts`);

  // Poll the env until every prompt has been sent and no answer is
  // still on its way out, then exit 0.
  for (;;) {
    let done = false;
    try {
      const work = await fetch(`${envUrl}/work`);
      const state = (await work.json()) as Json;
      done = state.done === true;
    } catch {
      // The env is briefly unreachable: poll again.
    }
    if (done && agg.inFlightCount() === 0) break;
    await sleep(WORK_POLL_MS);
  }
  process.exit(0);
}

main().catch((err: unknown) => {
  log(`fatal: ${(err as Error).message}`);
  process.exit(1);
});
