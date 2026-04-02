# Teacher's Guide — TDA for Financial Markets
**MCD524 · Deep Learning Aplicado a las Finanzas · Instituto Tecnológico de Santo Domingo**

---

## How to Use This Guide

This guide mirrors the slide deck (`slides.html`) slide by slide. For each slide you'll find:
- **Goal** — what the slide is trying to accomplish
- **Talking points** — things to say beyond what's printed on the slide
- **Likely student questions** — and suggested responses

**Timing:** The deck is designed for ~60 minutes. Part labels indicate natural pause/question points. The optional slide 24 (Topological Hurst) can be skipped if time is short.

**Prerequisites assumed:** LSTM/CNN familiarity, basic statistics, no topology background.

---

## Slide 1 — Title

**Goal:** Frame the session as building *new tools*, not replacing existing ones.

**Talking points:**
- This is an interdisciplinary topic: pure mathematics (algebraic topology) applied to practical ML.
- The title says "extracting shape from time series" — keep returning to this phrase throughout the lecture.
- The result of today is not memorizing formulas but developing intuition for *what topology sees that statistics misses*.

**Student questions:**
- *"Is this a real technique used in industry?"* — Yes. Hedge funds (including Two Sigma) have published on TDA for signals. Gidea & Katz (2018) is peer-reviewed and influential.
- *"Do I need a math background?"* — No. We build every concept from scratch. The goal is intuition, not proof.

---

## Slide 2 — Why Does TDA Matter?

**Goal:** Establish the complementarity framing early. TDA does not compete with LSTM — it sees different things.

**Talking points:**
- LSTM sees a 1D sequence and learns local patterns: "after this candle shape, the price often rises."
- TDA treats the same data as a geometric object floating in high-dimensional space and asks: "what is its shape?"
- The 2008 crash example is the hook: topological features changed weeks before traditional models noticed anything.
- Use the SVG diagram to contrast: left side is a list of numbers; right side is a point cloud with a hole in it. Both came from the same series.

**Student questions:**
- *"How can a sequence of returns become a point cloud?"* — Great question — we'll answer that precisely in Part 3 (slide 14). For now, just accept that it's possible.
- *"Is the 2008 result cherry-picked?"* — Gidea & Katz tested on multiple crises (2008 and 2015–2016 Chinese market). The signal appears consistently. We'll see it in detail in slide 18.
- *"Can TDA overfit?"* — Like any feature, yes. But the stability theorem (slide 12) gives a theoretical noise floor that standard features lack.

---

## Slide 3 — Roadmap

**Goal:** Orient students so they can track their own understanding.

**Talking points:**
- Parts 1–3 are conceptual (topology → persistent homology → time series connection).
- Parts 4–6 are applied (finance, ML pipelines, code).
- Every part builds on the previous. If someone is lost in Part 2, they will struggle with Part 4.
- Invite students to stop you at any point — the concepts are sequential.

**Student questions:**
- *"Will there be code?"* — Yes, slides 25–26 are runnable Python.
- *"Is there a project angle here?"* — TDA features can be added to Unit 7's LSTM asset allocation model. Slide 22 shows exactly how.

---

## Slide 4 — What Is Topology?

**Goal:** Build the core intuition: topology studies shape, not size.

**Talking points:**
- The coffee cup / donut equivalence is the classic example. Both have exactly one handle (one loop). Under topology, they are the same object.
- Contrast with geometry: geometry cares about lengths, angles, curvature. Topology only cares about connectivity.
- Key operations allowed: stretching, bending, twisting. Forbidden: cutting, gluing.
- The three things topology counts (connected components, loops, voids) correspond to Betti numbers — coming in the next slide.

**Student questions:**
- *"So a sphere and a cube are topologically the same?"* — Yes. You can deform a cube into a sphere continuously without cutting. Both have β₀=1, β₁=0, β₂=1.
- *"What does 'continuous deformation' mean formally?"* — A homeomorphism: a continuous bijection with a continuous inverse. For today, think "rubber sheet" deformation.
- *"Why does cutting matter?"* — Cutting can create or destroy holes. Topology is precisely the study of holes — so operations that change the hole structure are forbidden.

---

## Slide 5 — Betti Numbers

**Goal:** Make Betti numbers concrete and memorable with four visual examples.

**Talking points:**
- β₀ counts connected pieces. Hold up your hand: 5 disconnected points → β₀=5.
- β₁ counts independent loops. A circle has one loop. A figure-8 has two loops — each circle is independent.
- The disk is filled, so its boundary loop is "filled in" → β₁=0. This distinction between a hollow circle and a filled disk is crucial for understanding how persistence works.
- Higher Betti numbers (β₂ = voids enclosed by a surface, like a hollow sphere) exist but we focus on β₀ and β₁ today.

**Student questions:**
- *"What is β₂ for a donut?"* — β₂=0 (the surface of a torus encloses no 3D void, unlike a hollow sphere which has β₂=1).
- *"Can Betti numbers be computed algorithmically?"* — Yes, that's exactly what Ripser does. The algorithm is the column reduction of a boundary matrix — covered in the TDA literature but beyond today's scope.
- *"Are Betti numbers unique? Can two different shapes have the same Betti numbers?"* — Yes. Betti numbers are invariants but not complete invariants. Two spaces can have identical Betti numbers but different topology. Complete invariants are harder to compute.

---

## Slide 6 — From Data Points to Shape

**Goal:** Bridge the gap between discrete data (a finite point cloud) and continuous topology.

**Talking points:**
- A point cloud alone has no topology — it's just isolated points (all β₀ = N, β₁ = 0).
- We need to *connect* points to create structure. The rule: connect points that are "close enough."
- A 0-simplex is a point. A 1-simplex is an edge. A 2-simplex is a filled triangle. A 3-simplex is a filled tetrahedron.
- The word "simplicial complex" just means a collection of these building blocks glued consistently (if a triangle is in the complex, its three edges must be too).

**Student questions:**
- *"How do we decide what 'close enough' means?"* — That's the parameter r. Choosing one r is the old approach. Persistent homology tracks *all* r simultaneously — coming in slide 9.
- *"Is the simplicial complex unique given the data?"* — No. Different choices of r (or different complex constructions like Čech vs. Vietoris-Rips) give different complexes. That's why we track *all* scales.

---

## Slide 7 — Vietoris–Rips Complex

**Goal:** Give the precise definition of the most common complex construction.

**Talking points:**
- VR(X, r): include a k-simplex for every subset of (k+1) points with pairwise distances ≤ 2r.
- The factor of 2r comes from thinking of balls of radius r centered at each point — two balls overlap if and only if their centers are within 2r.
- The left figure shows small r: no edges, just isolated points. The right shows larger r: some edges, one triangle.
- In practice, Ripser computes VR complexes very efficiently using the fact that a clique in the 1-skeleton determines a simplex.

**Student questions:**
- *"Why not Čech complex instead?"* — The Čech complex (using actual ball intersections) is theoretically cleaner (Nerve Theorem applies directly) but harder to compute. VR is faster and satisfies `Čech(r) ⊆ VR(2r) ⊆ Čech(2r)`, so they carry the same topological information up to a scale factor of 2.
- *"What's the computational complexity?"* — Naively O(2^n) but Ripser's optimizations bring it to near-linear in practice for moderate dimensions.

---

## Slide 8 — Interactive Rips Animation

**Goal:** Let students *see* β₀ and β₁ change as r increases.

**Talking points:**
- Run the animation before students arrive (or demo live). The key moment: β₁ jumps to 1 when a loop forms, then falls back to 0 when a triangle fills it in.
- This is the birth and death of a topological feature.
- Point out: the loop does not persist forever — it gets killed when enough nearby points fill in the hole.
- Ask students: "Would you trust this loop as a real feature of the data, or is it just an artifact of the chosen r?" → That's exactly the question that persistent homology answers.

**Student questions:**
- *"What happens to β₀ as r grows?"* — It decreases monotonically: isolated points connect into components. β₀ starts at N (number of points) and ends at 1 (one connected component).
- *"Can β₁ go above 1?"* — Yes. Multiple loops can coexist at the same scale. In financial data, many H₁ features appear simultaneously during turbulent periods.

---

## Slide 9 — The Filtration

**Goal:** Introduce the key idea: track topology across ALL scales simultaneously.

**Talking points:**
- Instead of asking "what is the topology at scale r=0.5?", persistent homology asks "what is the full history of topology as r sweeps from 0 to ∞?"
- Walk through the four stages left to right: isolated points → edges form → loop appears (β₁=1) → triangle fills the loop (β₁=0) → 3-simplex.
- The persistence of the loop = death_r - birth_r = the length of time it existed. Long life = real structure.
- The key insight: "Long persistence = signal. Short persistence = noise." Return to this repeatedly.

**Student questions:**
- *"How is persistence different from just picking the right r?"* — If you pick one r, you're making a subjective choice that might miss features at other scales. Persistence gives you all features at all scales and lets the *data* tell you which ones are significant (by their lifetime).
- *"What counts as 'long' persistence?"* — Relative to other features in the diagram. A feature with persistence 10x the median is clearly signal. Features near the diagonal (persistence ≈ 0) are noise.

---

## Slide 10 — Persistence Diagram

**Goal:** Introduce the persistence diagram as the main output object of TDA.

**Talking points:**
- Each feature gets a point (birth, death). All points lie above the diagonal because death > birth always.
- Teal points (H₀) are connected components. They all birth at r=0 (all points start isolated) and die when they merge with another component.
- Red points (H₁) are loops. They birth at some positive r and die when filled by a triangle.
- Distance from the diagonal = persistence = lifetime. The big red point far from the diagonal is the significant feature.
- The diagonal itself represents features with zero lifetime — infinitesimal noise.
- One H₀ feature never dies (the last connected component). It goes to ∞ on the y-axis — shown at the top.

**Student questions:**
- *"Why is there a point labeled 'impossible' below the diagonal?"* — Because a feature can't die before it's born. All real points are strictly above the diagonal.
- *"Can we compare two persistence diagrams?"* — Yes, using the bottleneck distance or Wasserstein distance. The stability theorem (next slide) bounds how much a diagram can change when data changes.
- *"Is there one diagram per dimension k?"* — Yes. Dgm₀ tracks components, Dgm₁ tracks loops, Dgm₂ tracks voids. In finance we primarily use Dgm₁.

---

## Slide 11 — The Barcode

**Goal:** Show the barcode as an equivalent but more intuitive view of the persistence diagram.

**Talking points:**
- Barcode = persistence diagram rotated: each point becomes a horizontal bar from birth to death.
- Long bars = significant features. Short bars = noise. This is visually immediate.
- Read the barcode at a fixed r: the number of active H₁ bars at that r equals β₁ at that scale.
- The H₀ bars all start at 0 and die at various r's when components merge. One H₀ bar extends to infinity (the last component).
- The single long red bar is the feature Gidea & Katz tracked: it grows dramatically before market crashes.

**Student questions:**
- *"How do I know which r to use for a feature count?"* — You don't need to choose. The barcode gives you the full picture. For a scalar summary, use the total length of H₁ bars (= L¹ norm of the persistence landscape, coming in slide 19).
- *"What if there are hundreds of short bars?"* — That's normal. They represent noise. The signal is the handful of long bars that stand out. Filtering by persistence threshold (keeping only bars longer than some ε) is standard practice.

---

## Slide 12 — Stability Theorem

**Goal:** Give students confidence that persistence diagrams are not fragile constructs.

**Talking points:**
- Plain English version first: if you perturb your data by at most δ, the diagram moves by at most δ.
- This means noise cannot create long bars from nothing. A long bar in your diagram is either real structure or came from a large, systematic perturbation — not random noise.
- The bottleneck distance d_B is the cost of the optimal matching between two diagrams, where each point can also be matched to its projection on the diagonal.
- Compare to ML: this is like having L2 regularization built into the representation itself, not as a hyperparameter you tune.
- The formal statement: d_B(Dgm(f), Dgm(g)) ≤ ||f - g||_∞. This is the Cohen-Steiner–Edelsbrunner–Harer theorem (2007).

**Student questions:**
- *"Does stability mean TDA is immune to adversarial attacks?"* — No. It means small random perturbations have small effects. A large adversarial perturbation can still move the diagram significantly.
- *"What is the bottleneck distance intuitively?"* — The largest displacement needed to match points between two diagrams (with the option to kill/create near-diagonal points). It's the L∞ version of diagram comparison.
- *"Is there a stronger stability result?"* — Yes. The Wasserstein stability result bounds the p-Wasserstein distance between diagrams, not just the bottleneck. More sensitive to accumulation of small errors.

---

## Slide 13 — TDA in One Diagram

**Goal:** Synthesize Part 1 and Part 2 into a single pipeline before moving to applications.

**Talking points:**
- The four boxes represent the complete workflow: Data → Filtration → PD → Feature vector.
- The feature vector is what connects TDA to standard ML. After this step, any model works.
- Ripser handles the heavy lifting (Filtration → PD). The vectorization step (PD → feature vector) is done by persim or giotto-tda.
- The stability guarantee applies to the diagram step, not necessarily the vectorization. Some vectorizations (landscapes, images) preserve stability; others (naive coordinate vectors) do not.

**Student questions:**
- *"Does the feature vector size depend on the diagram?"* — It depends on the vectorization. Persistence landscapes and images produce fixed-size vectors regardless of how many points are in the diagram. That's why they're preferred for ML.

---

## Slide 13b — The Key Bridge: Time Series Have Shape

**Goal:** Prepare students for the delay embedding before introducing the formal definition.

**Talking points:**
- This slide previews the three archetypal cases: periodic → circle, random walk → blob, trending → arc.
- Show it before the formal definition so students have a mental target.
- The key message: the *type* of dynamics (oscillatory, diffusive, trending) determines the *shape* of the point cloud. Topology reads that shape.
- The same TDA pipeline works for all three cases — the output (diagram) tells you which regime you're in.

**Student questions:**
- *"How do we get from a 1D time series to a point cloud?"* — Delay embedding, formally defined in the next slide.
- *"Is the circle in the periodic case a perfect circle?"* — No, it's a noisy approximation. But it's topologically a circle (β₁=1) and the persistence diagram will show one long H₁ bar.

---

## Slide 14 — Sliding Window Embedding

**Goal:** Define the delay embedding precisely and motivate with Takens' theorem.

**Talking points:**
- The formula: each window of length d starting at time t (with stride τ) becomes a point in ℝᵈ.
- Walk through the SVG slowly: the highlighted window (t=3, t=4) maps to the yellow point in the 2D scatter plot with coordinates (x₃, x₄).
- d=2 gives a 2D point cloud (easy to visualize). In practice d=3 or d=5 is common.
- τ (the lag) controls how much "time" each step covers. For daily financial data, τ=1 is common.
- Takens' theorem (1981): for a deterministic dynamical system, the delay embedding *topologically recovers* the attractor. For financial data, the connection is empirical but well-documented.

**Student questions:**
- *"How do I choose d and τ?"* — d: use False Nearest Neighbor analysis (choose smallest d where the fraction of false neighbors drops near zero). τ: use mutual information minimization or just set τ=1 for financial returns. In practice, giotto-tda has utilities for this.
- *"What if the series is not deterministic?"* — The theorem doesn't apply formally, but empirically the embedding still captures regime information. Think of it as a feature engineering choice justified by empirical results.
- *"Why log returns instead of raw prices?"* — Log returns are approximately stationary. The shape of the point cloud is comparable across time windows. Raw price levels have non-stationary scales.

---

## Slide 15 — Embedding Shapes: Interactive Animation

**Goal:** Reinforce the three regimes visually and interactively.

**Talking points:**
- Click each button and narrate what you see.
- Periodic: the cloud forms a closed loop (visually a circle in the embedding). Ripser will find β₁=1 with high persistence.
- Random walk: the cloud is a diffuse blob. No persistent loops.
- Trending: the cloud traces an arc (elongated, correlated). The topology is different from both — β₀ features persist (the cloud doesn't close).
- Ask students to predict the topology before revealing it.

**Student questions:**
- *"Can real financial data show multiple regimes in the same period?"* — Yes. That's why we use a sliding window: the diagram changes over time, tracking regime transitions.
- *"What does a market bubble look like topologically?"* — Likely similar to trending (arc-shaped) with increasing persistence, then a rapid topological change near the peak as the bubble structure breaks down.

---

## Slide 16 — Signal Interpretation

**Goal:** Map topological signatures to financial regimes.

**Talking points:**
- Four regimes, four topological signatures. Return to this table during the finance slides.
- Pre-crash regime is the most important for financial applications: the topology *changes rapidly* before crashes (the landscape norm spikes).
- Mean-reverting markets (H < 0.5) are attractive for arbitrage strategies.
- Trending markets (H > 0.5) suit momentum strategies.
- The Hurst exponent H connects all four regimes — covered in slide 23.

**Student questions:**
- *"Can we detect regime changes automatically with TDA?"* — Yes. Monitoring the rate of change of the persistence landscape L¹ norm is one approach. A sudden jump signals a regime transition.
- *"What H value do real stock indices have?"* — Typically H ≈ 0.5–0.65 at daily scales. Intraday high-frequency data often shows H < 0.5 (mean-reverting microstructure). Long-horizon (monthly) data sometimes shows H > 0.6.

---

## Slide 17 — Financial Time Series as Geometry

**Goal:** Walk through the concrete implementation steps for the S&P 500 application.

**Talking points:**
- This slide is the bridge between the theory (slides 1–16) and the implementation (slides 25–26).
- Step by step: log returns → sliding window → 3D point cloud → VR filtration → persistence diagram → one diagram per trading day.
- The output is a *time series of persistence diagrams* — one topological fingerprint per day.
- Each fingerprint can be compared to previous days. Rapid changes in the sequence of diagrams = market stress.

**Student questions:**
- *"Why d=3 and τ=1?"* — Common default for daily financial returns. Increasing d captures longer-range memory but makes Ripser slower. For exploration, start with d=3.
- *"Why window W=50?"* — Roughly 2 months of trading days. Long enough to capture medium-term structure, short enough to be responsive to regime changes. This is a hyperparameter to tune.
- *"Does this work for individual stocks or just indices?"* — Both. Gidea & Katz used index data (DJIA, S&P 500, NASDAQ). For individual stocks, the signal-to-noise is lower but the approach still applies.

---

## Slide 18 — Crash Detection: Gidea & Katz (2018)

**Goal:** Present the flagship empirical result motivating TDA in finance.

**Talking points:**
- This is the paper that put TDA in financial ML. Published in *Physica A* (peer-reviewed, ~500 citations).
- The L¹ norm of the persistence landscape is a scalar summary of the persistence diagram at each time step.
- The norm spiked sharply in the months *before* the Lehman collapse (September 2008), not after.
- The same method predicted the 2015–2016 Chinese stock market correction. Not cherry-picked.
- The chart on the right is stylized; actual results are in the paper.

**Student questions:**
- *"What is the false positive rate?"* — The paper reports results qualitatively; formal precision/recall analysis was done in follow-up work (Gidea et al. 2020). The signal is earlier than VIX but noisier.
- *"Can we use this for live trading?"* — Yes, conceptually. You'd compute the rolling L¹ norm and issue an alert when it exceeds a threshold (e.g., 2 standard deviations above its 1-year mean).
- *"Why does the norm spike before a crash?"* — The interpretation: as the market approaches a crash, correlations across assets break down and the collective dynamics become more "loopy" — more cyclical instability. The topology captures this structural change before price movements do.

---

## Slide 19 — Persistence Landscapes

**Goal:** Explain the most popular vectorization method and why it connects to ML.

**Talking points:**
- The persistence landscape converts a diagram (a set of points) into a sequence of functions.
- For each diagram point (b, d), define a tent function peaked at ((b+d)/2, (d-b)/2) with height equal to the persistence (d-b)/2.
- λ_k(j, t) = the j-th largest tent function value at scale t.
- The result lives in L²([0,∞)), a Hilbert space. This means you can take means, compute norms, use kernel methods, etc.
- The L¹ norm = total area under all landscape functions ≈ total persistence = total bar length in the barcode.
- This is the scalar that Gidea & Katz tracked.

**Student questions:**
- *"Is the landscape differentiable with respect to the input data?"* — Not everywhere (tent functions have corners). Differentiable alternatives exist (e.g., the soft landscape or Topological AutoEncoder approaches), needed for end-to-end deep learning.
- *"How many landscape levels k do we keep?"* — Usually the first 1–3 levels contain most of the signal. Higher levels represent rare coincidences of multiple features at the same scale.

---

## Slide 20 — Persistence Images

**Goal:** Introduce the CNN-friendly vectorization.

**Talking points:**
- Persistence images (Adams et al., JMLR 2017) convert the diagram to a 2D grayscale image.
- Steps: rotate axes to (birth, persistence), apply a weighting function (w(b,p) = p, so long bars are brighter), convolve with a Gaussian kernel, discretize to a fixed pixel grid.
- Output: a fixed-size image that any standard CNN can process.
- The Gaussian smoothing is what makes persistence images differentiable (unlike landscapes).
- The pixel resolution is a hyperparameter: 20×20 to 50×50 are common for financial data.

**Student questions:**
- *"Which is better: landscapes or images?"* — Depends on downstream task. Images work better with CNNs; landscapes work better with linear models and kernel methods. Empirically comparable performance in most benchmarks.
- *"Can I use both?"* — Yes. Concatenate landscape statistics and image features as input to your model.

---

## Slide 21 — Full TDA → Deep Learning Pipeline

**Goal:** Show the complete operational pipeline from raw data to predictions.

**Talking points:**
- The six boxes represent the complete system. Walk left to right slowly.
- The crucial message: TDA features are *additive* to your existing pipeline. You don't discard your LSTM; you add topological features as an extra input branch.
- The time axis at the bottom: this entire computation runs for each trading day (with a sliding window). The result is a time series of topological features.
- In production: the bottleneck is the Ripser computation per window (milliseconds for d=3, W=50). Fast enough for daily data.

**Student questions:**
- *"How many TDA features should I add?"* — Start with 5–10 scalar statistics from the persistence diagram (count, sum, max, mean, std of H₁ persistence). Add landscape or image features if you need more expressiveness.
- *"Do TDA features need normalization before entering an LSTM?"* — Yes. StandardScaler or RobustScaler. The scale of persistence values depends on the data's distance scale.

---

## Slide 22 — TDA + LSTM Architecture

**Goal:** Give a concrete architectural blueprint students can implement in their projects.

**Talking points:**
- Two input streams: raw returns sequence (for LSTM) and TDA feature vector (for FC branch).
- Each branch produces a 64-dimensional embedding. These are concatenated to form a 128-dim joint representation.
- The concatenated representation goes through two FC layers with dropout before the output head.
- This architecture adds ~10k parameters over a standalone LSTM — negligible overhead.
- The box on the right explains *why* this works: LSTM captures local temporal patterns; TDA captures global structural patterns. They are complementary by construction.

**Student questions:**
- *"Should TDA features be computed online or offline?"* — For research: compute offline and store as a feature matrix. For production: compute online with a rolling window.
- *"What loss function should I use?"* — Same as your baseline: cross-entropy for direction classification, MSE for return regression, Sharpe-based for portfolio optimization.

---

## Slide 23 — The Hurst Exponent

**Goal:** Introduce the Hurst exponent as the quantitative measure of long memory.

**Talking points:**
- Standard Black-Scholes uses Brownian motion: no memory, increments are i.i.d.
- Mandelbrot argued (and empirically documented) that real financial time series have long memory.
- The fractional Brownian motion (fBm) B^H_t parameterizes this: H=0.5 → standard BM (no memory), H>0.5 → persistent trends, H<0.5 → mean-reverting.
- The covariance decay Cov ∼ t^{2H-2} is non-summable for H>0.5, meaning past shocks have lasting effects.
- Look at the three paths on the right: the smooth trending line (H=0.8), the jagged mean-reverting line (H=0.3), and the standard random walk in between (H=0.5).

**Student questions:**
- *"How do you estimate H in practice?"* — Standard methods: R/S analysis (Hurst 1951), DFA (detrended fluctuation analysis), periodogram regression. All have weaknesses. The topological estimator (next slide) is a new alternative.
- *"Is H constant over time for a stock?"* — No. H is non-stationary. Rolling window estimation shows it fluctuates significantly. Some stocks shift between trending and mean-reverting regimes.

---

## Slide 24 — Topological Hurst Estimator *(Optional — Research Frontier)*

**Goal:** Connect the course material to active research. Skip if time is short.

**Talking points:**
- This slide describes the paper currently being written in this research group.
- The key conjecture (now proved): the expected number of H₁ bars longer than ε in the persistence diagram of a delay-embedded fBm follows a power law in ε, with exponent α(H) that is *strictly monotone* in H.
- This means: fit the log-log slope of the barcode count function → invert α(H) → obtain a TDA-based estimate of H.
- Advantage over R/S and DFA: the topological estimator is invariant under monotone transformations of the series (e.g., taking absolute values, applying a nonlinear function). Standard estimators are not.
- This is a genuinely new result — not in any textbook yet.

**Student questions:**
- *"How do you compute α(H) in practice?"* — Compute ripser on the delay embedding at multiple values of ε, plot log(count) vs. log(ε), fit a line. The slope is -α(H). Then look up H from the calibration curve α(H).
- *"Is the paper published?"* — Not yet (2026, in preparation). The theoretical results are complete; computational experiments are ongoing.
- *"Why is invariance to monotone transformations important?"* — In finance, log returns are related to price returns by a log transform. If your estimator changes when you apply a monotone transformation, it's measuring something scale-dependent, not the true memory structure.

---

## Slide 25 — Python Code: Computing PH from Financial Data

**Goal:** Show a complete runnable example end-to-end.

**Talking points:**
- Walk through each section with the `# ---` comments as guides.
- Section 1: yfinance download is straightforward. The log-return calculation is the standard ln(P_t / P_{t-1}).
- Section 2: the `sliding_window_cloud` function is the delay embedding from slide 14.
- Section 3: `ripser(cloud, maxdim=1)` computes H₀ and H₁. The result is a dict with key `'dgms'`.
- Section 4: `plot_diagrams` from persim gives the standard birth-death scatter plot. The barcode loop manually plots H₁ intervals.
- Warn students: infinite death values (np.inf) in H₀ represent the never-dying component. Filter these before computing statistics.

**Practical tips to share:**
- `pip install ripser persim yfinance` (all on PyPI, work in Colab)
- For large windows (W > 200), use `ripser(cloud, maxdim=1, thresh=0.5)` to limit the filtration radius and speed up computation.
- GUDHI is the fallback if ripser has installation issues on Windows.

**Student questions:**
- *"What do I do with `diagrams[0]` vs `diagrams[1]`?"* — diagrams[0] is H₀ (components), diagrams[1] is H₁ (loops). For financial regime detection, use diagrams[1].
- *"How long does ripser take on W=50?"* — < 1 second on a laptop. The sliding window over 3000 trading days runs in < 1 hour total.

---

## Slide 26 — Python Code: TDA Features + ML Model

**Goal:** Show how to extract scalar TDA features and train a classifier.

**Talking points:**
- `tda_features` returns 5 numbers: count, sum, max, mean, std of H₁ persistence. Simple but surprisingly effective.
- The label is binary: will the next 5 days return be positive?
- StandardScaler is critical before Gradient Boosting.
- The claimed accuracy (~55–60%) is typical for directional prediction on daily returns. Better than 50% (random) and comparable to LSTM baselines. The improvement is larger during high-volatility periods.
- This is the *simplest* TDA ML model. Production systems use persistence landscapes, persistence images, or end-to-end differentiable TDA.

**Student questions:**
- *"What's the baseline accuracy without TDA features?"* — For S&P 500 daily direction, a naive majority-class classifier gives ~52%. LSTM alone typically achieves 53–56%. Adding TDA features often pushes this to 56–60%.
- *"Can I use these features in my Unit 7 LSTM assignment?"* — Yes. Replace `GradientBoostingClassifier` with your LSTM and append TDA features as auxiliary inputs (slide 22 architecture).
- *"The code labels `future > 0` — isn't this lookahead bias?"* — Good catch. In a real system you'd be careful to use only information available at time t. Here we're illustrating the feature construction, not building a live trading strategy.

---

## Slide 27 — Python Ecosystem for TDA

**Goal:** Give students a practical reference for the available tools.

**Talking points:**
- Ripser: fastest PH computation. Written in C++, called from Python. The standard choice.
- giotto-tda: the sklearn-compatible pipeline library. Handles the full workflow (embedding → PH → vectorization) in one pipeline object.
- persim: companion to ripser. Provides vectorizations (landscapes, images) and diagram metrics.
- GUDHI: the most comprehensive library. Includes alpha complexes (useful for spatial data), cubical complexes (images), and Mapper. Slightly harder to install.
- KeplerMapper: for the Mapper algorithm (topological data visualization, not covered today).
- torch-tda: for differentiable PH, enabling end-to-end gradient-based training.

**Recommendation for students:** Start with `ripser + persim`. Upgrade to `giotto-tda` for production-quality pipelines.

---

## Slide 28 — What TDA Adds to Your Deep Learning Toolkit

**Goal:** Summarize the take-home messages.

**Talking points:**
- Each of the five points is a distinct value proposition. Don't rush — give each 20–30 seconds.
- Point ①: Shape features are fundamentally different from statistical moments (mean, variance, skewness). They capture multi-scale geometric structure.
- Point ②: The stability theorem is the TDA analog of Lipschitz continuity. It's a mathematical guarantee, not an empirical claim.
- Point ③: Regime detection is the killer app. LSTM detects regime changes *after* they've happened (via loss of prediction accuracy). TDA detects them *before*.
- Point ④: TDA features are model-agnostic. They work as extra inputs to any architecture.
- Point ⑤: This is a live research area. Papers from 2023–2026 are still establishing baselines.

**Student questions:**
- *"Is TDA always better?"* — No. For datasets without meaningful geometric structure, TDA adds noise. It shines when the data has topological features (cycles, voids) that encode dynamics — which financial data does.
- *"What's the main weakness of TDA for finance?"* — Interpretability. A long H₁ bar tells you "there's a loop", but explaining what that loop *means* economically is non-trivial. This is an open research problem.

---

## Slide 29 — Further Reading

**Goal:** Point students to next steps for self-study.

**Talking points:**
- Dey & Wang (2022) is available free as a PDF — most complete textbook.
- Gidea & Katz (2018) is the must-read for financial applications.
- Bubenik (2015) for persistence landscapes.
- The giotto-tda tutorial notebooks on GitHub are excellent for hands-on practice.
- For the advanced students: Chazal et al. (2016) "Structure and Stability of Persistence Modules" for the theoretical foundations.

---

## Slide 30 — Q&A

**Goal:** Consolidate the four core concepts and open discussion.

**Talking points:**
- The four memory boxes summarize the session in four phrases. Use them as a checklist: can each student explain each phrase in one sentence?
- Suggested closing question for discussion: "Where else in your curriculum could TDA features be applied?" (Expected answers: NLP embeddings, protein folding, network analysis, anomaly detection.)

**Possible advanced discussion questions:**
- *"What is the relationship between TDA and manifold learning (t-SNE, UMAP)?"* — Both study geometric structure of data. TDA is more rigorous (provable stability, no optimization). UMAP/t-SNE are better at visualization.
- *"Can TDA be applied to graph-structured financial data (e.g., market correlation networks)?"* — Yes. Use the graph filtration (edge weights as filtration values) or the clique complex of the correlation matrix. Active research area.
- *"How does TDA relate to information geometry?"* — Both study geometry of statistical manifolds, but from different angles. Connections are being developed (2024–2026 literature).

---

## Common Misconceptions to Address

| Misconception | Correction |
|---------------|------------|
| "TDA replaces my LSTM" | No. TDA features *complement* LSTM by capturing different signal |
| "You need to choose r before computing PH" | No. Persistent homology tracks *all* r simultaneously |
| "Short bars are errors in the algorithm" | No. Short bars represent real but ephemeral topological features (noise). Long bars are signal |
| "Topology requires continuous spaces; financial data is discrete" | The Vietoris-Rips construction works on finite point clouds. No continuity required |
| "The stability theorem means PH is invariant to noise" | No. It means small noise → small diagram change. Large noise can still change the diagram |

---

## Timing Guide

| Part | Slides | Target time |
|------|--------|-------------|
| Part 1: Topology foundations | 4–7 | 12 min |
| Interactive demo | 8 | 3 min |
| Part 2: Persistent homology | 9–13 | 12 min |
| Part 3: Time series → geometry | 13b–16 | 10 min |
| Part 4: Finance applications | 17–20 | 10 min |
| Part 5: ML pipeline | 21–22 | 5 min |
| Part 6: Hurst exponent | 23–24 | 5 min (24 optional) |
| Part 7: Code + ecosystem | 25–27 | 8 min |
| Summary + Q&A | 28–30 | 5 min |
| **Total** | | **~70 min** |

---

*Prepared for MCD524 · TDA for Financial Markets · 2026*
