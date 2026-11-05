# Ripple-65B-Preview on one M5 Ultra (TensorFold native, the NVFP4 release)

Serves [Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview), ricursiveAI's dense 65B model
with contexts up to 1M tokens, from the publisher's own
[NVFP4 release](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4) as published, with TensorFold's native
Metal engine, as an OpenAI-compatible server with reasoning, tool calls, structured output, the prompt cache and
several streams. Nothing is converted. The NVFP4 release keeps attention, embeddings and head in bf16 and stores the
MLPs in NVFP4: 53 GiB against the bf16 release's 121 GiB, which leaves 135 GiB of a 256 GB Mac for caches.

The engine's `ricursive` family is on the `ripple-metal` branch of
[BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold), proposed to TensorFold as
[PR #2231](https://github.com/ashhart/TensorFold/pull/2231) and not yet on its main. This recipe builds that PR's head,
commit `1f520992`.

## Quick start

Needs an M5 Mac (the family refuses M1-M4: its prompts run on the M5's tensor units), Python 3.11+, git, `hf` and
[Zig 0.17.0](https://ziglang.org/download/).

```bash
git clone https://github.com/ashhart/TensorFold ~/github/TensorFold
hf download ricursiveAI/Ripple-65B-Preview-NVFP4 --revision 99eea4477a001a41b86914e41feb4b2ed86105e5 \
  --local-dir ~/models/Ripple-65B-Preview-NVFP4
cp local.example.toml local.toml      # then set zig (and paths, if yours differ)
python3 -m mac build                  # the pinned engine and libtfgrammar, under ~/.cache/ripple-65b
python3 -m mac up                     # http://127.0.0.1:8897/v1, model "ripple"
python3 -m mac chat "Write a Python function that merges overlapping intervals."
```

`python3 -m mac down|status|logs|command|validate` stop, inspect and check it. To serve the LAN on login:
`python3 -m mac launchd --port 8888` writes `~/Library/LaunchAgents/com.ricursiveai.ripple-65b.plist` (not loaded),
then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ricursiveai.ripple-65b.plist`.

## Settings

`recipe.toml` holds the defaults; `local.toml` overrides them for one Mac, and an environment variable of the same
name in upper case overrides both (`PORT=8888 HOST=0.0.0.0 python3 -m mac up`).

| setting | default | |
|---|---|---|
| `model_repo`, `model_revision`, `model_dir` | the NVFP4 release | the bf16 release serves the same way (`ricursiveAI/Ripple-65B-Preview` at `240d5966cbe00f0627c699e59ef58560b60303cf`) |
| `context` | 262144 | a stream's window; the engine takes up to 1,048,576 (the card's 1M), checked to 258,039 |
| `parallel` | 2 | streams decoded together, each reply equal to its run alone |
| `max_tokens` | 32768 | a reply's default ceiling; thinking counts |
| `temperature`, `top_p`, `top_k` | unset | the release names no sampling settings: the server's defaults (greedy) |
| `prompt_cache_gib` | 16 | kept prompt states between turns; they give way to streams |
| `drafts` | true | the lane engine's copy drafts (replies are identical with them off) |

**Memory.** The NVFP4 weights take 54.8 GiB, wired between requests. 64 of the 80 layers slide over 1,024 keys and
keep a 3,072-slot ring; every fifth layer keeps every key, in bf16: 16.75 GiB a stream at 262,144 tokens (64.8 GiB
at 1,048,576). The engine
holds weights and caches to 75% of RAM, so a 256 GB Mac has 135 GiB for streams and kept prompts.

**Thinking.** Ripple decides for itself whether to think: it writes `<think>` when it does, and the server puts that
part in `reasoning_content`; a reply without it is all `content`. The chat template has no thinking switch.
Ripple's think markers are spelled from several tokens, so a `thinking_budget` and `logprobs` are refused by name.

## What works, and how it was checked

On one M5 Ultra (256 GB, macOS 27), the engine at commit `1f520992` (the speed and prompt runs on the same family
code before its last rebase onto TensorFold main), checkpoint revision 99eea447. The full account, with the bf16
release's numbers, is the engine's
[recipe page](https://github.com/BobClawblaw/TensorFold/blob/ripple-metal/docs/recipes/ripple-65b-metal.md).

- **Exactness.** Against a float64 run of the same checkpoint (`tools/zig/ricursive_reference.py`), over 2,586
  teacher-forced positions: mean KL 0.0007 for prompt chunks and 0.0011 for decode windows. Prompt chunks of 2,048,
  1,000 and 333 rows give bit-identical logits, and decode windows equal one-row steps bit for bit.
- **Kernels.** Every kernel the family runs passes TensorFold's kernel contract: 36 of 36, with 15 planted controls
  all caught.
- **Serving** (the recipe harness's quick plan): structured output 18/18, drafted == plain 12/12; tool calls 5/5;
  seeded sampling 9/9; a resumed turn equal to the cold one after a restart; a 60-turn soak and 12 concurrent replies
  each equal to the request alone; streaming 4/4, multi-turn recall 8/8, stop strings 3/3, max_tokens 3/3.
- **Long context.** A passphrase a third of the way into 120,041- and 258,039-token prompts comes back exactly.
- **Thinking** scores 5/10 on the suite's rule, which wants reasoning in every reply: Ripple thought on five of the
  ten problems and answered the rest directly, nine of the ten answers right.

### What the NVFP4 release costs

Against the bf16 release's float64 run on the same 2,586 positions, this checkpoint's own float64 run gives mean KL
0.0059 and the same top token at 95.8% of positions; the engine serving it gives 0.0060 and 96.1%. The cost is the
publisher's quantization: the engine adds nothing measurable to it.

The publisher serves this release with SGLang, which also quantizes activations to fp4. TensorFold keeps activations
in bf16 with fp32 sums, so this recipe serves the release's NVFP4 weights with bf16 activations: the numbers here are
that model, not the publisher's W4A4 kernels.

In one prompt of 5,200 tokens, one position's top token differs between the prompt path and decode (after "every
segment gets 4,", `096` against `000`). The flip comes from the shared bf16 GEMM on the attention projections, at a
position where this checkpoint is nearly torn between two answers; the bf16 release does not flip there.

## Measured

The recipe harness (4 Nov 2026), `--context 262144 --parallel 2`, greedy 256-token replies, nothing else loaded:

| streams | tokens/s (all streams) | first token | a round |
|---:|---:|---:|---:|
| 1 | 10.1-10.2 | 0.61-0.74 s | 99 ms |
| 2 | 18.6-19.1 | 0.93-1.10 s | 107 ms |

| cold prompt | 2,220 | 8,776 | 35,015 | 139,723 tokens |
|---|---:|---:|---:|---:|
| tokens a second | 569 | 605 | 587 | 434 |
| time to first token | 3.9 s | 14.5 s | 59.6 s | 322 s |

A multi-turn chat's resumed turn starts in 0.67 s against 8.2 s cold. One stream reads the 53 GiB of weights each
step; a second stream costs 8% more a round. The harness's quality suite was not run on this build.

On the [TensorFold leaderboard](https://tensorfold.dev/benchmarks) (community-v1, submitted 5 Nov 2026, pending
moderation): code at 11.0 tok/s sampled and 12.2 greedy, first text in 0.35 s. The chat cell shows as incomplete:
the suite asks for 256-token replies without ignoring end-of-sequence, and Ripple ends most of its chat answers
sooner, so those samples count as early stops (about 10.3 tok/s while they ran).

## Notes

- **Not served:** images (Ripple has no vision tower) and the publisher's DFlash2 drafter. Drafts come only from
  the lane engine's copies.
- Only the M5 Ultra was checked; a smaller Mac needs fewer or shorter streams.

## License

The recipe is MIT. Ripple-65B-Preview's weights, both releases, are Apache-2.0 (ricursiveAI); TensorFold is
Apache-2.0.
