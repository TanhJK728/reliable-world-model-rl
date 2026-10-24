<div align="center">

# World-Model RL

**How uncertainty controls change policy learning**

Continuous control · Learned world models · Policy gradients · Uncertainty

[Research paper](docs/research_paper_v2/research_paper.pdf) · [Results](#what-the-experiments-show) · [Mathematics](#what-the-mathematics-establishes) · [Code](#explore-the-code) · [Get started](#get-started)

</div>

![Learn from experience, refine a policy inside a five-model dynamics ensemble, and compare learned-model and environment returns from matched initial states.](docs/assets/research-overview.svg)

An embodied agent can use a world model to practice before acting. The challenge is deciding how much of that practice will help in the environment.

This project studies that question in **Pendulum-v1**, a continuous-control task with angle, angular velocity, and bounded torque. It connects **PPO, learned dynamics, imagined policy refinement, and matched evaluation** in a compact PyTorch implementation built from scratch.

**The research paper studies how uncertainty controls change the learning objective.** It derives the error introduced by rollout boundaries and gives necessary and sufficient conditions for positive weights to preserve gradient alignment at a finite-action decision state. Exact finite-action and Gaussian examples show how weighting can reverse an update. The existing Pendulum experiments connect these questions to continuous control.

| Control | World modeling | Research design |
| :--- | :--- | :--- |
| Tanh-Gaussian PPO with GAE and reward normalization | Five dynamics models with state geometry constraints | Four refinement methods, two policy initializations, paired evaluation |
| **−153.0 ± 4.4** return across three PPO seeds | Policy, random-action, and extreme-torque datasets | **200-step** comparisons from the same initial states |

## The research question

**How do uncertainty-based stopping and advantage weights change what a policy learns?**

The analysis follows three parts of that question:

1. **Targets:** what continuation value does an imagined rollout receive when it stops?
2. **Updates:** when do positive confidence weights preserve an improving policy direction?
3. **Control:** how do the resulting policies perform under the environment's dynamics?

The world model learns state transitions. Imagined rollouts use the exact Pendulum reward, so the comparison focuses on learned dynamics and policy optimization. Here, **environment return** means evaluation in the original Gymnasium simulator.

## What the mathematics establishes

**Stopping assigns a boundary value.** For a common finite horizon, discount, initial distribution, and dynamics, the difference between a stopped objective and the full return is

```math
J_{\tau,b}^{\pi}-J_H^{\pi}
=\mathbb{E}_{\pi}\left[\gamma^{\tau}\left(b_{\tau}(S_{\tau})-V_{\tau}^{\pi}(S_{\tau})\right)\right].
```

Here, τ is the stopping time, b is the assigned boundary value, and V is the remaining value under the policy. The paper also characterizes when adding a constant to every reward preserves policy rankings under stopping.

**Positive weights can change the direction of learning.** With exact centered advantages A and policy scores ψ, compare the original gradient with its weighted version:

```math
g=\mathbb{E}[A\psi],\qquad g_w=\mathbb{E}[wA\psi].
```

At one finite-action decision state, the paper gives necessary and sufficient conditions for fixed positive weights to preserve nonnegative gradient alignment for every centered advantage. The conditions separate the policy's score space from the geometry of its parameter update. Fisher preconditioning gives strict ascent under the score-space condition whenever the original gradient is nonzero.

Exact finite-action and bounded-reward Gaussian examples show how positive weights can produce a direction that lowers the original return. The proofs analyze these mechanisms directly; the existing experiments report the combined behavior of the implemented learning rules.

[Read the research paper](docs/research_paper_v2/research_paper.pdf) · [LaTeX source](docs/research_paper_v2/main.tex) · [Original study](docs/world_model_rl_study.pdf)

## What the experiments show

### 1. Measure performance alongside the return gap

Each policy is evaluated under learned and environment dynamics using the same initial states, deterministic actions, and 200-step horizon. Define

```math
G(\pi)=J_{\mathrm{model}}(\pi)-J_{\mathrm{env}}(\pi).
```

A positive gap means the model overestimates the policy's return. The code calls this quantity the *exploitation gap*.

For any policy update, let $`\Delta`$ denote the change from the old policy to the new one. The definition gives a useful accounting identity:

```math
\Delta J_{\mathrm{env}}=\Delta J_{\mathrm{model}}-\Delta G.
```

This separates predicted improvement from changes in model optimism. The experiments report both environment performance and the return gap.

The main comparison starts from a partially trained PPO policy and refines it for approximately 500,000 imagined transitions. The dynamics model stays fixed and was trained on data from a stronger policy.

| Refinement method | Environment return ↑ | Model − environment return | Final rollout length |
| :--- | ---: | ---: | ---: |
| **Starting policy** | **−576.4** | −3.0 | — |
| Fixed horizon, H = 5 | −792.4 ± 248.7 | +20.0 ± 22.2 | 5.0 |
| Fixed horizon, H = 20 | −753.8 ± 105.5 | +27.7 ± 20.7 | 20.0 |
| Uncertainty termination | −1472.7 ± 24.1 | +8.9 ± 25.6 | 4.3 |
| Uncertainty-weighted advantage | −1462.1 ± 4.2 | −13.4 ± 1.2 | 20.0 |

Values are mean ± sample standard deviation across **three refinement seeds**, each evaluated on 50 matched episodes. The starting policy is one shared checkpoint. Final rollout length averages the last 20 completed training segments per seed.

![Model and environment returns for the starting policy and four refinement methods, with error bars across refinement seeds.](runs/comparison_seeds/model_vs_real_return_seeds.png)

**All four methods reduce mean environment return from the starting policy in this setting.** Uncertainty termination shortens rollouts to about four steps. Advantage weighting also produces poor control despite a small return gap. These results make the trade-off concrete: uncertainty controls change the training signal, and their value must be judged by the resulting behavior.

With the stronger policy as the starting point, fixed H = 20 and uncertainty termination retain returns near −150. Fixed H = 5 and weighted advantages reach about −296 and −289. This comparison uses one refinement run per method.

[Three-seed results](runs/comparison_seeds/comparison_aggregated.csv) · [Strong-policy comparison](runs/comparison/comparison.md)

### 2. Connect prediction error to data coverage

The ensemble trains on a 60,000-transition dataset collected by the strong PPO policy, split by episode into training and validation. Two additional datasets probe random actions and extreme torques.

| Evaluation dataset | Transitions | One-step state MSE ↓ | Mean ensemble disagreement |
| :--- | ---: | ---: | ---: |
| Policy data | 60,000 | 1.96 × 10⁻⁶ | 1.17 × 10⁻⁶ |
| Uniform random actions | 30,000 | 2.97 × 10⁻³ | 2.19 × 10⁻⁴ |
| Extreme torques | 30,000 | 5.17 × 10⁻³ | 3.85 × 10⁻⁴ |

Prediction error and disagreement both increase outside the policy-data distribution. The policy row evaluates the full source dataset, including training transitions. Together, these measurements motivate checking uncertainty on the states and actions a refining policy actually visits.

| Learning a control policy | Predicting over longer rollouts |
| :---: | :---: |
| ![PPO learning curves across three seeds.](runs/ppo_summary/learning_curve_mean_std.png) | ![Horizon-averaged squared state error for policy, random-action, and extreme-torque data.](runs/world_model_eval/rollout_error.png) |
| PPO reaches **−153.0 ± 4.4** across seed means, with 30 evaluation episodes per seed. Random actions average **−1198.6**. | Policy-data errors remain below both shifted datasets at all measured horizons. The plotted error averages squared state distance over the rollout. |

[PPO evaluation](runs/ppo_summary/evaluation.csv) · [One-step prediction](runs/world_model_eval/one_step_results.csv) · [Rollout prediction](runs/world_model_eval/rollout_results.csv)

## Inside the method

### Continuous control with bounded actions

The policy samples a Gaussian latent action and maps it to the torque range:

```math
z\sim\mathcal N\!\left(\mu_\theta(s),\sigma_\theta^2\right),
\qquad a=2\tanh z.
```

The implementation includes the change-of-variables correction in the action log probability, GAE advantages, clipped policy and value objectives, and KL-based early stopping. Initial PPO training uses reward normalization. Imagined refinement uses original rewards and initializes both actor and critic from the starting checkpoint. Evaluation reports the original environment reward.

<details>
<summary><strong>From the action density to the PPO update</strong></summary>

For Pendulum's scalar action, the transformed density is

```math
\log \pi_\theta(a \mid s)
=
\log \mathcal{N}\left(z;\mu_\theta(s),\sigma_\theta^2\right)
-
\log\left(2(1-\tanh^2 z)\right).
```

The mean depends on the state. The learned standard deviation is shared across states. The code adds a small numerical stabilizer inside the logarithm.

PPO uses the corrected density to form the likelihood ratio

```math
\rho_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}.
```

The PPO clipped surrogate objective is

```math
L^{\mathrm{clip}}
=
\mathbb{E}_t\left[
\min\left(
\rho_t\hat{A}_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
```

The implementation maximizes this surrogate with normalized GAE advantages and jointly fits the critic.

</details>

[Policy and density](ppo_core.py) · [PPO training](train_ppo.py)

### Dynamics with physical state geometry

The observation is $`s=(\cos\varphi,\sin\varphi,\dot\varphi)`$. Each ensemble member learns a normalized state increment, then projects its prediction:

```math
\hat s_{t+1}^{(j)}=\Pi\!\left(s_t+f_{\phi_j}(s_t,a_t)\right),
\qquad j=1,\ldots,5.
```

Here, $`f_{\phi_j}`$ includes input normalization and output rescaling. The projection normalizes each member's angular coordinates and clips angular velocity to the environment range. The ensemble mean supplies the next imagined state.

Each member trains on a bootstrap sample. An episode-grouped split keeps all recorded transitions from each episode in the same subset.

[Dynamics and projection](world_model.py) · [Ensemble training](train_world_model.py)

### Uncertainty as a control on learning

Disagreement among the five predictions supplies an uncertainty proxy:

```math
u(s,a)=\frac{1}{3}\sum_{k=1}^{3}\mathrm{Var}_{j}\!\left[\hat s_{k}^{(j)}\right].
```

The experiments compare fixed rollouts against two ways to use this signal:

- **Terminate:** end an imagined segment when disagreement exceeds the validation-set 95th percentile, $`\tau\approx6.63\times10^{-6}`$, or the horizon limit.
- **Weight:** multiply normalized policy advantages by $`w_t=\max\{0.02,\exp(-\beta u_t/\tau)\}`$ while retaining the full rollout.

Termination also cuts the critic's continuation value at the segment boundary. Weighting acts on the policy objective while the critic keeps its usual loss. These choices connect uncertainty directly to credit assignment and optimization.

[Imagined environment](imagined_env.py) · [Policy refinement](train_imagined_ppo.py)

## Explore the code

| Start here | What to look for |
| :--- | :--- |
| [`ppo_core.py`](ppo_core.py) · [`train_ppo.py`](train_ppo.py) | Action distributions, actor–critic learning, reward scaling |
| [`collect_transitions.py`](collect_transitions.py) | Policy, random-action, and extreme-torque data collection |
| [`world_model.py`](world_model.py) · [`train_world_model.py`](train_world_model.py) | Residual dynamics, state projection, ensemble disagreement |
| [`imagined_env.py`](imagined_env.py) · [`train_imagined_ppo.py`](train_imagined_ppo.py) | Model-based rollouts and the four refinement variants |
| [`compare_imagined_methods.py`](compare_imagined_methods.py) · [`aggregate_comparison.py`](aggregate_comparison.py) | Paired evaluation and summaries across seeds |

The [research paper](docs/research_paper_v2/research_paper.pdf) characterizes when positive advantage weights preserve alignment with the original policy gradient and derives how rollout boundaries change the learning objective. It includes complete proofs, finite-action and Gaussian examples, and the existing Pendulum control study. See the [file map](PROJECT_FILES.md) for the remaining utilities.

## Get started

The figures and result tables above are available directly in the repository.

```bash
git clone https://github.com/TanhJK728/reliable-world-model-rl.git
cd reliable-world-model-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To train a new PPO policy with the reported configuration, choose a fresh output directory:

```bash
RUN_DIR="runs/manual_$(date +%Y%m%d_%H%M%S)"
python train_ppo.py \
  --seed 0 --total-timesteps 1000000 \
  --anneal-lr 0 --ent-coef 0.01 --normalize-reward 1 \
  --output-dir "$RUN_DIR" --device cpu
```

CPU, CUDA, and Apple MPS are supported. Model checkpoints (`.pt`) and transition arrays (`.npz`) are generated locally. [`run_pipeline.sh`](run_pipeline.sh) records the full sequence for the strong-policy experiment and writes to fixed paths under `runs/` and `data/`.

## Questions to take further

- **Broader experience:** how does mixing expert and exploratory data change model error and downstream control?
- **Adaptive uncertainty:** how should rollout thresholds change as the policy visits new states?
- **Iterative learning:** when should an agent collect more environment data and update its world model?
- **Richer observations:** which findings carry over to latent dynamics and vision-based manipulation?

The project provides a small, inspectable setting for studying how prediction, uncertainty, and policy optimization interact in embodied control.
