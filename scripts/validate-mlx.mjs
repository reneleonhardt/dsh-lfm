// Opt-in hardware validation. Does not modify the user's DSH profiles or run tools.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Runtime } from "../lib/runtime.js";
import { LfmAdapter } from "../lib/adapter.js";

const [modelPath, pythonPath] = process.argv.slice(2);
assert.ok(
  modelPath && pythonPath,
  "Usage: node scripts/validate-mlx.mjs MODEL_DIRECTORY VENV_PYTHON",
);
assert.ok(
  process.platform === "darwin" && process.arch === "arm64",
  "Run on native arm64 macOS",
);
const root = await mkdtemp(join(tmpdir(), "dsh-mlx-validation-"));
const runtime = new Runtime({ dataDir: root });
try {
  await runtime.init();
  let p = await runtime.addModel(resolve(modelPath));
  assert.equal(p.backend, "mlx-vlm");
  p.mlxPython = resolve(pythonPath);
  p.port = Number(process.env.DSH_MLX_VALIDATION_PORT || 18270);
  p.maxTokens = 512;
  p.context = 4096;
  p.thinking = false;
  p = await runtime.save(p);
  if (process.env.DSH_MLX_CANCEL_LOAD_MS) {
    const ms = Number(process.env.DSH_MLX_CANCEL_LOAD_MS);
    assert.ok(Number.isFinite(ms) && ms > 0);
    const signal = AbortSignal.timeout(ms);
    let child;
    const monitor = setInterval(() => {
      child ||= runtime.running.values().next().value?.child;
    }, 20);
    try {
      await assert.rejects(runtime.acquire(p, signal), () => signal.aborted);
    } finally {
      clearInterval(monitor);
    }
    assert.ok(
      child,
      "Timer expired before server spawn; increase cancellation interval",
    );
    assert.equal(runtime.running.size, 0);
    assert.equal(runtime.starting.size, 0);
    assert.ok(child.exitCode !== null || child.signalCode);
    console.log(
      "PASS: timed load cancellation stopped the owned server and cleaned up state",
    );
  } else {
    const lease = await runtime.acquire(p, AbortSignal.timeout(900000));
    const state = lease.state;
    lease.release();
    assert.equal(state.ready, true);
    const unauthenticated = await fetch(state.url + "/v1/models");
    assert.equal(
      unauthenticated.status,
      401,
      "Server must require authentication",
    );
    console.log("PASS: model preload, authenticated readiness, identity check");
    const adapter = new LfmAdapter({}, runtime);
    const base = {
      provider: "lfm",
      model: p.id,
      maxTokens: 256,
      reasoningEffort: "off",
    };
    const user = (text) => ({
      role: "user",
      content: [{ type: "text", text }],
    });
    async function generate(messages, extra = {}) {
      const blocks = [];
      let finish;
      for await (const chunk of adapter.stream({
        ...base,
        messages,
        signal: AbortSignal.timeout(120000),
        ...extra,
      })) {
        if (chunk.type === "block-end") blocks.push(chunk.block);
        if (chunk.type === "finish") finish = chunk;
      }
      return { blocks, finish };
    }
    const first = await generate([user("Reply with exactly: hello")]);
    assert.equal(first.finish?.reason.kind, "stop", JSON.stringify(first));
    assert.ok(first.blocks.some((b) => b.type === "text" && b.text.trim()));
    console.log("PASS: streamed generation", JSON.stringify(first.blocks));

    const thinking = await generate(
      [user("What is 17 * 19? Think briefly, then answer.")],
      { reasoningEffort: "on" },
    );
    assert.equal(thinking.finish?.reason.kind, "stop");
    assert.ok(
      thinking.blocks.some((b) => b.type === "reasoning" && b.text.trim()),
    );
    assert.ok(
      thinking.blocks.some((b) => b.type === "text" && b.text.includes("323")),
    );
    console.log("PASS: Thinking mode separates reasoning from the answer");

    const limited = await generate(
      [user("Count from 1 to 1000, one number per line.")],
      { maxTokens: 1 },
    );
    assert.equal(limited.finish?.reason.kind, "max-tokens");
    console.log("PASS: per-call output token limit");

    const overBudget = await fetch(state.url + "/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: "Bearer " + state.apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: p.modelPath,
        messages: [{ role: "user", content: "hello" }],
        max_completion_tokens: p.context,
        context_management: [],
        stream: true,
      }),
    });
    assert.equal(overBudget.status, 400);
    assert.match(await overBudget.text(), /context tokens/);
    console.log("PASS: prompt plus output overflow rejected before streaming");

    const long = await generate([
      user(
        "Here are repeated notes:\n" +
          "The blue key opens the garden gate.\n".repeat(150) +
          "\nReply with exactly: ready",
      ),
    ]);
    assert.equal(long.finish?.reason.kind, "stop");
    assert.ok(
      long.blocks.some((b) => b.type === "text" && b.text.includes("ready")),
    );
    console.log("PASS: chunked prefill of a longer prompt");

    const abort = new AbortController();
    const deadline = setTimeout(
      () => abort.abort(Error("Cancellation test timeout")),
      120000,
    );
    let sawToken = false,
      finish;
    const started = Date.now();
    try {
      for await (const chunk of adapter.stream({
        ...base,
        messages: [user("Count from 1 to 1000, one number per line.")],
        signal: abort.signal,
        maxTokens: 512,
      })) {
        if (chunk.type === "text-delta" && !sawToken) {
          sawToken = true;
          abort.abort(Error("Intentional validation cancellation"));
        }
        if (chunk.type === "finish") finish = chunk.reason.kind;
      }
    } catch (error) {
      if (!sawToken) throw error;
    } finally {
      clearTimeout(deadline);
    }
    assert.ok(sawToken && abort.signal.aborted);
    assert.equal(state.active, 0);
    assert.ok(
      !finish || finish === "aborted",
      `Unexpected cancellation finish: ${finish}`,
    );
    const recovered = await generate([user("Reply briefly: still here?")]);
    assert.equal(recovered.finish?.reason.kind, "stop");
    console.log(
      "PASS: cancel after first token and generate again; elapsed ms",
      Date.now() - started,
    );
    console.log(
      "NOTE: observe Activity Monitor GPU/memory during cancellation; client abort alone does not prove GPU work stopped promptly.",
    );

    const tools = [
      {
        name: "lookup_number",
        description: "Return the value for a named constant.",
        parameters: {
          type: "object",
          properties: { name: { type: "string" } },
          required: ["name"],
        },
      },
    ];
    const prompt = user(
      "Use lookup_number with name='answer'. Do not guess the value. After the result, report the value.",
    );
    const called = await generate([prompt], { tools });
    assert.equal(
      called.finish?.reason.kind,
      "tool-calls",
      "Model did not produce a structured tool call",
    );
    const call = called.blocks.find((b) => b.type === "tool-call");
    assert.equal(call?.name, "lookup_number");
    assert.equal(JSON.parse(call.arguments).name, "answer");
    const final = await generate(
      [
        prompt,
        {
          role: "assistant",
          content: called.blocks,
          source: { kind: "model", provider: "lfm", model: p.id },
        },
        {
          role: "tool",
          toolCallId: call.id,
          source: { kind: "tool", callId: call.id },
          content: [{ type: "text", text: '{"value":42}' }],
        },
      ],
      { tools },
    );
    assert.equal(final.finish?.reason.kind, "stop");
    assert.ok(
      final.blocks.some((b) => b.type === "text" && b.text.includes("42")),
    );
    console.log(
      "PASS: structured tool call, arguments and tool-result round trip",
    );
    await runtime.unload(p.id);
    assert.equal(runtime.running.size, 0);
    console.log("PASS: unload owned process group");
  }
} catch (error) {
  for (const state of runtime.running.values()) console.error(state.log);
  throw error;
} finally {
  await runtime.dispose();
  await rm(root, { recursive: true, force: true });
}
