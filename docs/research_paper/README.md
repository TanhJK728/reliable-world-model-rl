# How Uncertainty Controls Change Policy Learning in World Models

The paper studies the learning objectives created by rollout stopping and positive advantage weights. It develops a finite-action alignment characterization, a Fisher-preconditioned version, stopping-boundary identities, and exact finite-action and continuous-action examples. The empirical section uses the repository's existing Pendulum results.

- Paper: [research_paper.pdf](research_paper.pdf)
- Main source: [main.tex](main.tex)

Complete proofs appear in the paper and its appendices. The main results characterize when positive advantage weights preserve alignment with the original return gradient, and how rollout boundaries affect returns and reward-shift invariance.

Open this directory in the Codex LaTeX editor for a live preview. A local PDF build is also available with:

```sh
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build -jobname=research_paper main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build -jobname=research_paper main.tex
```

The two images in `images/` are selected copies of existing figures from `runs/comparison_seeds/` and `runs/ppo_summary/`. No experiment output was regenerated for this manuscript.
