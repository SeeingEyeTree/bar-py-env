# A practical AlphaStar-inspired plan for BAR

This document proposes a model architecture and a phased training plan for
building an RL agent for Beyond All Reason. It deliberately scales DeepMind's
AlphaStar approach **down** to what's achievable on a single workstation with
the data we can realistically collect.

## What AlphaStar actually did, in two paragraphs

DeepMind's agent took inputs through three pathways: spatial feature maps
(minimap + screen) processed by a small ResNet, a variable-length set of
**entities** — every unit, building, resource, and effect — processed by a
Transformer with self-attention, and a vector of scalar features (resources,
supply, game time, last-action one-hot). The spatial and entity branches
are fused via **scatter connections** — entity embeddings are written back
into the spatial map at each unit's location before the spatial CNN runs.
A deep LSTM carries temporal state. Outputs are produced **autoregressively**:
action type → delay until next action → queued flag → which units to select
(via a pointer network) → target unit (pointer over visible enemies) →
target screen location. A separate value head estimates the discounted
reward, and — crucially — **the value head during training also sees the
opponent's observations** (a privileged-information trick that they ablate
at 82% win-rate vs 22% without; one of the largest single contributors to
performance).

Training had three phases. **Supervised learning** on ~971,000 anonymized
human replays trained the policy via KL-divergence against human actions —
this alone produced a top-16% rated agent. **Reinforcement learning**
continued from those weights using a combination of three off-policy
corrections: TD(λ) for the value function, V-trace clipped-importance
sampling for the policy gradient, and **UPGO** (a self-imitation update
that biases the policy toward trajectories with better-than-average return).
A KL penalty toward the supervised policy throughout RL prevents the agent
from forgetting basic strategies during exploration. The decisive third
phase is the **AlphaStar League** — a population of ~900 distinct agents
across three roles (main, main exploiter, league exploiter) playing
**Prioritized Fictitious Self-Play (PFSP)**, where opponent-mixture
probabilities are proportional to the win rate against each opponent.
Main exploiters reset to the supervised checkpoint periodically and only
play the main agent; league exploiters target any agent the league hasn't
beaten yet. Each agent trained for 44 days on 32 TPU v3 cores.

The total bill was on the order of millions of dollars in TPU-hours. We are
not going to reproduce that. But the architecture, the three-phase structure,
and several specific tricks (scatter connections, opponent-info value head,
KL anchor, PFSP) absolutely transfer.

## Why BAR is easier than StarCraft 2 in ways that matter for us

Five differences cut the problem down:

1. **Lower APM ceiling.** SC2's pros click 300+ APM with meaningful per-click
   decisions. BAR sits around 50-100 APM with build queues handling most
   labor. The action space at a per-frame level is much sparser. We can step
   the policy every ~30 sim frames (1 second) instead of every 5-10.
2. **Simpler unit micro.** No blink-stalkers, no force-fields, no
   chronoboosting. Most BAR units just auto-attack-move. Combat policy mostly
   reduces to "where do I send the army."
3. **Two resources instead of three layers.** Metal + energy. No
   gas-vs-mineral juggling, no supply, no separate upgrades-tied-to-tech.
4. **Smaller community, smaller pool of replays.** Hundreds of thousands of
   ranked BAR duels, not millions. But we don't need millions — see below.
5. **A native opponent ladder already exists.** BARbarianAI in three
   difficulties is shipped with the game. We can train against `BARb easy`,
   then `medium`, then `hard`, then self-play — that's a free curriculum.

Three differences make it harder:

1. **No native ML API.** We're going through the widget bridge — observations
   come over TCP as JSON, not as protobufs from an in-engine API. Every
   per-step round-trip is ~1 ms more than pysc2's would be. Tolerable when
   we step once per second.
2. **Maps are bigger.** Comet Catcher is 8192×6144 elmos vs SC2's typical
   ~200×200 cells. Spatial feature maps need to either downsample more
   aggressively or use a multi-resolution pyramid.
3. **No spatial feature layer API.** We'd have to compute our own LOS maps,
   threat maps, etc. in either the widget or Python from the unit list.

## Proposed architecture

Smaller than AlphaStar, same skeleton.

```
Inputs
======

  Scalar features (~32 floats)
      game_frame_norm, my_metal, my_energy, my_metal_income,
      my_energy_income, my_metal_storage, my_energy_storage,
      n_my_units, n_visible_enemies, n_my_workers, n_my_combat,
      n_my_buildings, n_enemy_workers_seen, ..., last_action_type_onehot
        |
        v
      MLP(32 -> 128 -> 128, ReLU)
        |
        v
      [128]

  Entity features (variable-length set of units, max ~256)
      Per-unit fields:
        - one-hot unit_def_id (top-K by frequency; rare units bucketed)
        - my-side flag, ally-side flag
        - position (x, z, height) normalized
        - current hp / max_hp
        - current command type (one-hot, ~20 categories)
        - is-being-built / build-progress
        - cloaked / paralyzed / stunned flags
        |
        v
      Linear projection -> 64 per entity
        |
        v
      Transformer encoder, 2 layers, 4 heads, 64 d_model
        |
        v
      Set-pooling (mean + max + my-side mean concat) -> [256]

  Scatter connections (AlphaStar's trick; do this BEFORE the spatial CNN)
      For each entity, take its 64-d embedding from the Transformer output
      and scatter-add it into the spatial map at the entity's (x, z) cell.
      This gives the spatial CNN per-pixel knowledge of which units are
      where, fusing the entity and spatial pathways without flattening
      one of them. The Transformer still sees the raw entity set; the
      CNN sees the same units projected back into 2D space.

  Spatial features (8 channels at 64x64 resolution = 128 elmos per cell)
      ch0: height map
      ch1: LOS (1 if I can see this cell)
      ch2: my-unit density
      ch3: my-unit power (sum of HP)
      ch4: visible-enemy density
      ch5: visible-enemy power
      ch6: metal-spot mask (1 at known mexes)
      ch7: is-buildable mask
        |
        v
      Small ResNet:
        Conv 8 -> 32, stride 1
        ResBlock x 3 (32 channels)
        Conv 32 -> 64, stride 2 (downsample to 32x32)
        ResBlock x 3 (64 channels)
        GlobalAvgPool + FC -> [256]

Concat -> [128 + 256 + 256] = [640]
       -> Linear 640 -> 384

Core
====

  LSTM, 1 layer, 384 hidden (~16x smaller than AlphaStar's core)

  The recurrent state propagates between policy steps in the same episode.

Heads
=====

  Value head:           MLP 384 -> 128 -> 1
                        IMPORTANT: during *training* the value head also
                        sees the opponent's observations (god's-eye state).
                        This is AlphaStar's biggest single ablation win
                        (82% vs 22% win-rate). We already have a god's-eye
                        observation mode in the replay extractor; the live
                        env's synced gadget needs to expose it too so the
                        critic has the full state. The policy still only
                        sees the player's POV, so deployed inference is
                        unchanged.
  Policy head, autoregressive:
    1. action_type      Categorical over ~12 high-level types
                        {NO_OP, MOVE, ATTACK, BUILD, RECLAIM, REPAIR,
                         FACTORY_QUEUE, STOP, GUARD, PATROL, FIGHT,
                         SELF_DESTRUCT}
    2. selected_units   Pointer attention over the entity-transformer
                        outputs. Predicts a multi-hot mask. (Variable-length
                        output -- use the same trick AlphaStar used: a single
                        softmax with a "stop" symbol.)
    3. target_position  Two-step spatial: first a heatmap over the 32x32
                        post-CNN grid (categorical), then a small
                        regression head for the within-cell offset.
                        Skipped for action types that don't need a position.
    4. target_unit      Pointer attention over entities (only for ATTACK,
                        REPAIR, RECLAIM, GUARD).
    5. unit_def         Categorical over the ~30 most common build-options
                        (only for BUILD / FACTORY_QUEUE).
    6. queued           Bernoulli (shift-queue or not).
```

**Total parameters: ~5-8 million.** That's deliberately under 10M so that:

- It trains in single-digit hours on a consumer GPU (RTX 4070-class) per
  supervised epoch with a few hundred thousand transitions.
- Inference is fast enough that the per-step Python overhead is negligible.
- We can iterate on architecture changes in a day rather than a week.

AlphaStar was ~140M parameters. We're going for roughly 1/20th of that, which
is consistent with the much smaller action space and shorter horizon BAR
asks for.

## Training plan, five phases

Each phase has a *concrete success criterion* — don't move to the next one
until the current one converges.

### Phase 0 (where we are now) — wire the bridge fully

What's left before training is even meaningful:

- Action vocabulary: extend the widget from `MOVE/STOP/ATTACK/FIGHT/GUARD/PATROL`
  to also include `BUILD` (with unit-def parameter), `RECLAIM`, `REPAIR`, and
  factory queue commands.
- Observation enrichment: emit unit-def-id metadata once at game start
  (UnitDefs dump), threat-map and LOS channels each step.
- Live mode: add a god's-eye observation mode for training (we already have
  it in replay; needs a synced gadget for live games).

**Done when**: `random_agent.py` can issue a `BUILD` and the widget executes
it, observations include the threat map, and a god's-eye live obs has every
unit on the map.

### Phase 1 — replay extraction at scale

Goal: a curated dataset of 1,000-5,000 1v1 Comet Catcher games, all from your
currently-installed BAR version, sorted by player skill, extracted to a
disk-cheap binary format.

Pipeline:

1. Run `download_replays.py` with `--max 5000` and the player list bumped
   to "anyone with skill > 25". That's a few hours of API + downloads.
2. Run `extract_replay.py` in parallel across N spring-headless processes
   (more on parallelism below). At 20x sim speed, a 20-minute game extracts
   in ~60 wall-clock seconds. 5,000 games on 8 parallel workers ≈ 10 hours.
3. Post-process: each JSONL gets folded into a Parquet file (`pandas` or
   `pyarrow`) with one row per `(frame, team)` tuple. A 5k-replay corpus
   is roughly 5-15 GB on disk.

**Done when**: you have a Parquet dataset with ~5M transitions, viewable
in pandas, sliceable by player / game length / win-loss.

### Phase 2 — supervised behavior cloning

This is the phase where the most learning happens for the least compute.
AlphaStar found that a BC-only agent was already roughly equivalent to a
solid amateur StarCraft 2 player.

- Standard cross-entropy on each action component (action_type, unit_def,
  selected_units mask, target_position).
- Validation: hold out 10% of replays. Track top-1 and top-5 accuracy on
  `action_type` and `unit_def` at three game-time buckets (0-5min, 5-15min,
  15+min).
- Train for 5-10 epochs on a single GPU. Total: probably 4-12 hours.

**Done when**: the BC agent in a live game reliably:

- Builds a mex in the first 60 seconds.
- Builds at least one factory by 3 minutes.
- Has the commander somewhere reasonable (not the corner) at 5 minutes.
- Beats `BARb easy` 50% of the time without RL fine-tuning.

If it doesn't hit those, the issue is in the architecture or data, not RL —
go fix that before adding more complexity.

### Phase 3 — RL fine-tuning vs scripted opponents

Curriculum: `BARb easy` → `BARb medium` → `BARb hard`. Maybe 1-3 days per
rung.

- Algorithm: **PPO with a KL penalty toward the BC policy.** AlphaStar used
  V-trace + UPGO because they were doing distributed off-policy RL; PPO is
  much simpler to debug, and the KL anchor to BC prevents the policy from
  forgetting basic build orders during exploration.
- Reward: dense + sparse. Sparse: +1 win / -1 loss at game end.
  Dense (small weights so they don't dominate the win signal): per-minute
  metal income, per-minute unit kills, time-to-first-factory under a target.
- Parallel rollouts: 8 spring-headless processes feeding one PyTorch policy
  server. The policy holds a 5-step rolling buffer and updates every 1024
  transitions. PPO doesn't care about strict on-policy as long as the KL
  stays bounded.

**Done when**: agent beats `BARb hard` 60%+ on Comet Catcher 1v1.

### Phase 4 — self-play (modest league)

You don't need AlphaStar's 600+ agent league to see the next jump. A 4-agent
league is enough to expose a single-agent self-play policy to varied
opponents:

- 1 **main agent**: the one we care about. Trained against the league
  weighted by predicted win-rate.
- 1 **main exploiter**: reset to BC weights every N hours, trains only
  against the main agent. Finds main's weaknesses.
- 2 **league exploiters**: train against everyone, prioritize agents the
  league hasn't beaten yet. Prevents cycles.

All 4 agents share replay collection. With 16 CPU cores and 1 GPU you can
sustain ~12-16 parallel BAR instances total. Spread across 4 agents that's
3-4 parallel games per agent. Slow but functional.

**Done when**: the main agent is qualitatively distinct from the BC-only
policy — it scouts, it pivots tech, it builds counter-units.

### Phase 5 (optional) — online tuning against the community

If you ever want it to be competitive: the BAR community will play your bot
on the public lobby. Each game becomes another training data point.

## Parallelism, given that the engine is single-threaded

BAR's simulation thread is single-threaded — you cannot make one match run
faster by adding cores. **But you can run N matches simultaneously**, each
in its own `spring-headless` process, and they don't share state. This is
exactly how AlphaStar generated rollouts (just with TPU-fed parallelism on
top).

What we need to make this work:

1. **Port-per-process.** Each spring-headless instance has a widget that
   opens a TCP connection back to Python on its own port. The current
   `BarEnv(EnvConfig)` already takes a `port` parameter — we just need a
   wrapper that spawns N envs on ports `8765, 8766, 8767, ...`.
2. **Write-dir-per-process.** We already do this — every run gets its own
   timestamped folder. Concurrent runs need unique folders, which the
   timestamp + a PID suffix handles.
3. **Shared engine binary.** All N processes can `exec()` the same
   `spring-headless.exe`. They don't share locks on it because each writes
   only into its own write-dir.
4. **A Python-side actor pool.** One process holds the policy weights on
   GPU. N worker processes each own an env and ship observations to the
   policy via shared memory or a small queue. The policy returns actions
   in batched inference calls.

Per-process resource use: spring-headless uses ~1 core and 1-2 GB RAM on a
running Comet Catcher game. On a 16-core / 32-GB machine you can comfortably
host **8-12 parallel BAR instances**. That's the practical limit.

**Wall-clock-rollout math:**
- 1 BAR instance @ 20x speed = ~3 wall-minutes per 1-hour-sim game
- 8 parallel instances = ~21,600 sim-seconds-of-experience per wall-minute
- That's roughly 24,000 transitions/minute at 1-Hz policy stepping
- A typical PPO epoch wants ~1M transitions = ~40 minutes of wall clock

That's the same order of magnitude as a per-epoch training step on the GPU,
so you can keep both saturated. Compare to AlphaStar's hundreds of TPUs
generating millions of transitions per minute — we're roughly 100x slower.
That means our project takes weeks where theirs took days, but it's
quantitatively the same shape.

## Data we have vs data we need

**Have now** (after the recent pipeline work):

- 1 extracted JSONL (the one you just ran). ~600 obs frames. Enough for the
  tiny linear BC baseline; not enough for a real neural net.
- The pipeline to extract more: `extract_replay.py` runs unattended, takes
  ~1 min per replay at default speed.

**Easy to get in 1-2 days**:

- 500-2,000 replays via `download_replays.py` filtered to the installed game
  version. The API caps requests but a polite delay gets us there in a few
  hours of background pulls.
- Extract them in 8 hours of parallel processing on the user's box.

**Harder, would need engineering**:

- 10,000+ replays. We'd need to either:
  - Run pr-downloader against many historical BAR versions and accept that
    the network sees slightly different unit balance across them.
  - Or restrict to the most recent ~2 weeks of replays, all on the current
    version. That naturally caps the corpus at whatever the community
    produced lately (typically 1,000-5,000 1v1 duels per week).

**Not realistic without months of effort**:

- 100,000+ replays comparable to AlphaStar's training corpus. The community
  size doesn't support it.

This is fine. The architecture proposed above is sized for ~5M transitions,
not 5B.

## Compute budget reality check

Assume a single workstation: RTX 4070 + 16-core Ryzen + 64 GB RAM.

| Phase | Time | What dominates |
|---|---|---|
| 1 — Replay extraction | 1-2 days | CPU (8 parallel BAR instances) |
| 2 — Supervised pretraining | 1-2 days | GPU (~6h per epoch on 5M transitions, 5-10 epochs) |
| 3 — RL vs BARb easy/medium/hard | 1 week | Mix: CPU for rollouts, GPU for gradient steps |
| 4 — Self-play league | 2-4 weeks | Same as phase 3, slower convergence |

End-to-end to "agent that beats BARb-hard": realistic in **4-6 weeks of
elapsed time** on one workstation. That's roughly what an undergrad final
project looks like, not "$5M of TPUs."

## Concrete next 3 things to build

In order, before any neural net work:

1. **Action vocabulary extension** (Phase 0): widget supports `BUILD <unit_def>`,
   `RECLAIM`, `REPAIR`, factory queues. ~1 day. This is the load-bearing
   change — until the agent can issue a BUILD, behavior cloning is just
   "predict commander movement," which is what the current toy POC does.
2. **Observation enrichment** (Phase 0): emit a `unit_defs` dump on the first
   obs (def_id → name, cost, build options) so Python can convert raw IDs into
   meaningful features without hard-coded tables. Also add a threat-map
   channel computed in the widget. ~1-2 days.
3. **A `bar_env.parallel.PoolEnv` class** (early Phase 1): launch N envs on
   N ports, expose `step(batched_actions) -> batched_obs`. Without this, the
   eventual RL phase is sequential and 8x slower than it needs to be. ~1 day.

After those three, the supervised training script (#5M-param network, PyTorch,
mixed entity-transformer + small ResNet + LSTM) is maybe 2-3 days of focused
work. Then we have something we can actually train.

## Extension: mixture-of-experts managers (recommended, but optional)

The plan above is a single end-to-end policy in the AlphaStar mold. There's
a natural way to layer a **manager/expert decomposition** on top of it that
is well-suited to BAR specifically and brings the reward signals closer to
the components that actually drive them.

### Why this fits BAR

The strongest hand-coded BAR AI — BARbarianAI, the same CircuitAI codebase we
mapped earlier — is **already** organized around this decomposition. From
`src/circuit/module/`:

- `CEconomyManager` — metal/energy balance, mex/solar/fusion build pacing
- `CBuilderManager` — construction worker tasking and build-site selection
- `CFactoryManager` — factory placement and unit production queues
- `CMilitaryManager` — squad assembly, attack timing, retreat, defense

Those four classes do not share state through a shared neural network; they
share state through a `CCircuitAI` object that owns them and a task scheduler
that arbitrates. **A learned MoE policy with the same decomposition is just
the same idea with the hand-coded logic in each module replaced by a small
network.** The structural prior comes from BAR's actual game shape — macro,
production, military, scouting are genuinely separable subproblems on the
1-minute timescale.

### Two flavors of MoE, and which we want

There are two ways to do MoE in RL, both with research backing:

1. **Gating MoE** (Shazeer et al, Mixture-of-Experts, 2017): N expert networks
   each propose an action distribution; a learned gate produces a soft
   mixture. All experts are trained jointly on one global loss. Simpler.
2. **Hierarchical / Feudal** (Vezhnevets et al, FuN, 2017; Bacon et al,
   Option-Critic, 2017): a manager policy outputs a sub-goal or selects an
   expert to act; the chosen expert produces concrete actions; manager and
   experts have separate losses, often with per-expert intrinsic rewards.
   Closer to how BARbarianAI is structured.

For BAR we want **hybrid**: hierarchical at the action-routing level
(manager picks which expert(s) fire on this step) but with a shared
perception trunk (same observations → same encoded features into every
expert), and per-expert auxiliary value heads trained on shaped rewards.
The shared trunk avoids representation duplication; the routing structure
preserves interpretability and gives us per-expert reward signals.

### Architectural sketch

```
                  Shared trunk
                  ============
Scalars -> MLP    Entities -> Transformer    Spatial -> ResNet
       \              |                          /
        +---- concat + linear ---- LSTM ----+
                         |
                         v
                  shared state  s_t  (384-d)
                         |
   +---------+-----------+-------------+-----------+
   |         |           |             |           |
   v         v           v             v           v
 Economy   Factory    Military    Strategist     Critic
  head      head        head        head        (single shared
   |         |           |             |         value, but with
   v         v           v             v         per-expert
 econ      fac          mil         scout/      auxiliary
 action    action       action      tech        values for
 dist      dist         dist        action       shaping)
   |         |           |             |
   +---------+-----+-----+-------------+
                   |
                   v
        Manager gate (softmax over experts)
                   |
                   v
   For each step, sample which expert(s) fire
```

Concretely:

- **Shared trunk** is the same Transformer + ResNet + LSTM from the base
  architecture. The trunk is trained against the joint loss.
- **Four expert action heads**, each producing an autoregressive action of
  *its own* type. Each head has a tiny MLP (2 layers, 128 hidden) on top of
  the shared state, then the action-component sub-heads (`unit_def` for
  Economy and Factory experts, `target_position` and `target_unit` for
  Military, `scout_target` for Strategist).
- **Manager gate**: a small MLP from the shared state to a softmax over
  the 4 experts plus a NO_OP. Trained jointly via the PPO objective on
  the gate distribution.
- **Per-expert auxiliary value heads** alongside the main critic. Each
  expert has its own scalar value estimating the discounted *expert-specific*
  reward over the next horizon.

**Multi-expert firing.** The user clicks many simultaneous commands in BAR —
build mex AND order commander move AND queue factory — so a strict
one-expert-per-step gate is overly restrictive. We resolve this by
sampling the gate independently for each expert as a **Bernoulli** rather
than a categorical softmax. On any given step, 0-4 experts can fire. The
joint action is the concatenation of whichever experts' actions are sampled.

### Per-expert reward signals

Each expert gets its own dense reward, with the joint win/loss signal still
flowing through the main critic. This is where "reward signals closer to the
training" lands concretely:

| Expert | Auxiliary reward (per step) |
|---|---|
| Economy | Δ(metal_income + energy_income / 60) — units per minute of resource gained |
| Factory | Build-power added; sum of HP-equivalent units exiting factories per minute |
| Military | (enemy_HP_destroyed − own_HP_lost) / step_seconds; small bonus for territory held |
| Strategist | Scouted-area gained; correctly-predicted enemy-tech reveal; -1 for commander-dying |
| **Main critic** | Win = +1, Loss = -1 at game end. Discount factor 0.997. |

The auxiliary rewards are **only consumed by their own expert's auxiliary
value head**. The policy gradient that updates the manager gate and the
shared trunk still uses the main critic. This is important: if you let the
experts maximize their own rewards directly, you get failure modes like the
Economy expert spamming mexes forever because that maximizes m_income.
Auxiliary rewards inform a richer learned value function; they should not
*replace* the win-loss signal.

This is the same structural trick AlphaStar uses with the opponent-info
value head — privileged information shapes the critic, but doesn't change
the deployed policy. PsychLab / SC2LE call this *auxiliary tasks*; Jaderberg
et al's UNREAL paper showed it stabilizes training significantly.

### How supervised pretraining works in the MoE setting

Behavior cloning splits cleanly because the human's commands are typed:

- A human BUILD command with `unit_def ∈ {mex, solar, fusion, energy_storage,
  metal_storage, energy_converter}` → **Economy expert** training example
- A human FACTORY_QUEUE command, or a BUILD for `unit_def ∈ {bot_lab, vehicle_plant,
  aircraft_plant, shipyard}` → **Factory expert** training example
- A human MOVE / ATTACK / FIGHT / GUARD / PATROL / RECLAIM (when issued to
  combat units, identified by their unit-def class) → **Military expert**
- BUILD of scout-type units (raider, scout-bot), MOVE/PATROL of the
  Commander outside its base, and the manager's choice of "do nothing this
  step" → **Strategist expert** (a bit fuzzier; we can refine later)

Each expert has its own cross-entropy loss against the human action;
unsupervised steps (no command of that expert's type at frame N) just
mask out that expert's loss for that frame.

The manager gate's BC supervision is per-frame: "did the human issue an
economy command this step? a military command? both?" → Bernoulli targets
for each of the 4 experts.

### Risks

Honest accounting:

1. **MoE is finicky to train.** The classical failure mode is **expert
   collapse** — the gate routes everything to one expert and the others
   atrophy. Mitigation: a load-balancing entropy bonus on the gate (Shazeer
   et al call this the "noisy-top-k gating with load balancing loss"),
   plus the supervised pretraining naturally distributes commands across
   experts because the data is roughly balanced.
2. **Per-expert rewards can fight each other.** Economy says "build more
   mex," Military says "attack now and lose those builders." Mitigation:
   the auxiliary rewards only feed auxiliary value heads, never the policy
   gradient directly. The manager gate is the only thing trained on a
   joint objective.
3. **Less proven at AlphaStar scale.** AlphaStar uses a flat policy.
   Hierarchical RL has mixed track record on hard games — OpenAI Five
   tried hierarchy and ended up with a flat LSTM. The argument for trying
   it here is that BAR's decomposition is much cleaner than Dota's (where
   every hero is its own combinatorial action space).
4. **Implementation complexity.** This is 2-3x the code of a flat policy.
   Worth it if interpretability matters or if you specifically want to
   intervene on individual managers post-training; not worth it if your
   only goal is win-rate.

### My recommendation on timing

Don't start with MoE. Start with the flat AlphaStar-style policy from the
main design above through Phase 2 (supervised BC). Once that's working,
**fork** Phase 3 into two branches:

- Branch A: continue with flat PPO. Lower variance, faster to debug.
- Branch B: re-architect into the 4-expert MoE described here, retrain
  supervised on the typed-command split, then PPO with auxiliary rewards.

Compare both at end of Phase 3 by win-rate against BARb-hard. If MoE wins,
take it to Phase 4 (league). If flat wins or ties, you've validated a
useful structural finding (often the case for these decompositions on
small data — the structural prior helps less when there's enough data to
learn it implicitly) and you keep flat.

That's the honest evaluation discipline.

## What references I'm drawing on, besides AlphaStar

- **OpenAI Five (Dota 2)**: similar architecture, no spatial input — they
  found a flat LSTM over a fixed-size entity list was enough. Argues we
  could skip the spatial CNN initially. I'd keep it for BAR because the map
  matters more.
- **DreamerV3**: world-model-based RL — would let us do a lot of "imagined"
  training off a small replay corpus. Out of scope for first pass but worth
  knowing exists if we hit a data wall.
- **Gym-μRTS**: closest analogue published. They demonstrated PPO on a
  similar simplified RTS with much smaller compute. Useful as a sanity
  check on hyperparameters.
- **CircuitAI's threat-map and influence-map structures**: as we discussed,
  these are the "right" spatial features. Re-implementing them in Lua for
  the widget is straightforward.
- **Shazeer et al, "Outrageously Large Neural Networks" (2017)**: original
  MoE paper. The noisy-top-k gating + load-balancing loss is the trick
  that makes gated experts stable.
- **Vezhnevets et al, "Feudal Networks for Hierarchical RL" (2017)**:
  decomposes policy into manager (sets sub-goals) + worker (executes).
  Closest published analogue to the manager decomposition proposed in the
  MoE addendum.
- **Jaderberg et al, "UNREAL: Reinforcement Learning with Unsupervised
  Auxiliary Tasks" (2016)**: auxiliary value heads for shaped rewards
  without polluting the main objective. The training trick we use for
  the per-expert auxiliary rewards.
- **Bacon et al, "The Option-Critic Architecture" (2017)**: end-to-end
  learning of options (sub-policies). Mathematically clean; in practice
  the Feudal-style discrete-manager-with-shared-trunk we propose is
  more stable to train.

## What we should NOT copy from AlphaStar

- **The full action delay distribution.** AlphaStar predicted the *delay
  until next action* per step (and ablations show it matters in SC2). In
  BAR our step interval is fixed at 1 sec because the engine and the
  meta both tolerate that — so the delay head adds complexity without
  payoff. Worth revisiting in Phase 5 if we ever want sub-second reactivity.
- **The Z (player strategy) latent.** AlphaStar conditioned on a sampled
  "strategy statistic" mined from human replays (build orders, unit
  composition prefixes). This was decisive in SC2 because matchups are
  highly varied (PvP vs PvT vs PvZ behave differently). For BAR
  Comet-Catcher 1v1 we have one map and one matchup; the z variable adds
  parameters but the data won't pay it back. Skip in v1; reconsider once
  we expand to multiple maps.
- **The full 600+ agent league.** A 4-agent league is plenty for our scale.
- **The exact V-trace + UPGO algorithm.** PPO with a KL anchor to the BC
  policy gets us 80% of the way with one-tenth the implementation effort.
  If we hit a sample-efficiency wall later (signals: high variance in
  RL training, performance plateau against league), revisit V-trace and
  UPGO. The ablations show they're worth ~10-15% Elo each at scale.
- **Action-rate limits.** AlphaStar had a monitoring layer that throttled
  to ~22 actions per 5 seconds because the SC2 community insisted on a
  human-comparable rate. BAR's 1-Hz step interval already enforces a much
  stricter rate-limit naturally — we get this for free.
