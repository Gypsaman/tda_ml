
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()

OUTDIR = BASE / "outputs"
OUTDIR.mkdir(exist_ok=True)

PAIRS = [(1, 5), (2, 6), (4, 8)]


def tent_value(b, d, t):
    return max(0.0, min(t - b, d - t))


def tent_piecewise_text(b, d, name="f"):
    m = (b + d) / 2
    return f"""{name}(t) =
    0                  for t <= {b}
    t - {b}            for {b} <= t <= {m}
    {d} - t            for {m} <= t <= {d}
    0                  for t >= {d}"""


def landscape_values(pairs, t_values):
    values = np.zeros((len(pairs), len(t_values)))
    for i, (b, d) in enumerate(pairs):
        values[i, :] = [tent_value(b, d, t) for t in t_values]
    sorted_desc = np.sort(values, axis=0)[::-1]
    return values, sorted_desc


def print_header(title):
    print("\\n" + "=" * 80)
    print(title)
    print("=" * 80)


def explain_pairs(pairs):
    print_header("1) INPUT PERSISTENCE PAIRS")
    print("We use the same three birth-death pairs as in the earlier explanation:")
    for i, (b, d) in enumerate(pairs, start=1):
        persistence = d - b
        midpoint = (b + d) / 2
        height = persistence / 2
        print(f"Pair {i}: (b, d) = ({b}, {d})")
        print(f"  persistence d-b = {d}-{b} = {persistence}")
        print(f"  midpoint (b+d)/2 = ({b}+{d})/2 = {midpoint}")
        print(f"  triangle peak height = (d-b)/2 = {height}")
        print()

    print("For each pair (b, d), the tent function is")
    print("f_(b,d)(t) = max(0, min(t-b, d-t))")
    print()

    for i, (b, d) in enumerate(pairs, start=1):
        print(tent_piecewise_text(b, d, name=f"f{i}"))
        print()


def sample_point_explanations(pairs):
    print_header("2) SAMPLE POINT CALCULATIONS")
    sample_points = [2.5, 4.2, 4.8, 5.5]
    for t in sample_points:
        vals = [tent_value(b, d, t) for (b, d) in pairs]
        ordered = sorted(vals, reverse=True)
        print(f"At t = {t}:")
        for i, ((b, d), v) in enumerate(zip(pairs, vals), start=1):
            print(f"  f{i}({t}) from pair ({b},{d}) = {v}")
        print(f"  Sorted values (largest to smallest): {ordered}")
        print(f"  lambda_1({t}) = {ordered[0]}")
        print(f"  lambda_2({t}) = {ordered[1]}")
        print(f"  lambda_3({t}) = {ordered[2]}")
        print()


def breakpoint_analysis(pairs):
    print_header("3) BREAKPOINT ANALYSIS")
    print("Important breakpoints come from births, deaths, and triangle midpoints/crossings.")
    births = sorted({b for b, _ in pairs})
    deaths = sorted({d for _, d in pairs})
    mids = sorted({(b + d) / 2 for b, d in pairs})
    print(f"Births: {births}")
    print(f"Deaths: {deaths}")
    print(f"Midpoints: {mids}")
    print("Crossing points for this example appear at t = 3.5, 4.5, and 5.0.")
    print("These are where the ordering of the triangle heights changes.")
    print()
    print("Region summary:")
    print("  t < 1                : all zero")
    print("  1 <= t <= 2          : only f1 active")
    print("  2 <= t <= 3          : f1 and f2 active")
    print("  3 <= t <= 4          : compare f1 = 5-t with f2 = t-2; crossing at 3.5")
    print("  4 <= t <= 5          : all three active; ordering changes at 4.5")
    print("  5 <= t <= 6          : f2 and f3 active")
    print("  6 <= t <= 8          : only f3 active")
    print("  t >= 8               : all zero")


def plot_single_tent(pair, idx):
    b, d = pair
    t = np.linspace(0, 9, 1000)
    y = np.array([tent_value(b, d, x) for x in t])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, y)
    ax.set_title(f"Tent function f{idx}(t) for pair ({b}, {d})")
    ax.set_xlabel("t")
    ax.set_ylabel("height")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, max(y) + 0.4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUTDIR / f"tent_{idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_overlay(pairs):
    t = np.linspace(0, 9, 1200)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (b, d) in enumerate(pairs, start=1):
        y = np.array([tent_value(b, d, x) for x in t])
        ax.plot(t, y, label=f"f{i} from ({b},{d})")
    ax.set_title("Overlay of the three tent functions")
    ax.set_xlabel("t")
    ax.set_ylabel("height")
    ax.set_xlim(0, 9)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = OUTDIR / "overlay_tents.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_landscape_layers(pairs):
    t = np.linspace(0, 9, 2400)
    raw, lambdas = landscape_values(pairs, t)
    n_layers = lambdas.shape[0]

    for k in range(n_layers):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, lambdas[k])
        ax.set_title(f"Persistence landscape layer lambda_{k+1}(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("height")
        ax.set_xlim(0, 9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = OUTDIR / f"lambda_{k+1}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for k in range(n_layers):
        ax.plot(t, lambdas[k], label=f"lambda_{k+1}")
    ax.set_title("Persistence landscape layers together")
    ax.set_xlabel("t")
    ax.set_ylabel("height")
    ax.set_xlim(0, 9)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = OUTDIR / "landscape_all_layers.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def report_piecewise_landscape():
    print_header("4) PIECEWISE FORMULAS FOR THE LANDSCAPE")
    print("For this example, the exact landscape layers are:")
    print()
    print("lambda_1(t) =")
    print("  0          for t <= 1")
    print("  t - 1      for 1 <= t <= 3")
    print("  5 - t      for 3 <= t <= 3.5")
    print("  t - 2      for 3.5 <= t <= 4")
    print("  6 - t      for 4 <= t <= 5")
    print("  t - 4      for 5 <= t <= 6")
    print("  8 - t      for 6 <= t <= 8")
    print("  0          for t >= 8")
    print()
    print("lambda_2(t) =")
    print("  0          for t <= 2")
    print("  t - 2      for 2 <= t <= 3.5")
    print("  5 - t      for 3.5 <= t <= 4.5")
    print("  t - 4      for 4.5 <= t <= 5")
    print("  6 - t      for 5 <= t <= 6")
    print("  0          for t >= 6")
    print()
    print("lambda_3(t) =")
    print("  0          for t <= 4")
    print("  t - 4      for 4 <= t <= 4.5")
    print("  5 - t      for 4.5 <= t <= 5")
    print("  0          for t >= 5")
    print()


def numerical_L1_norm(pairs):
    print_header("5) L1 QUANTITY FOR THE PERSISTENCE LANDSCAPE")
    print("We compute the landscape L1 norm numerically as")
    print("||lambda||_1 = sum_k integral |lambda_k(t)| dt")
    print()

    t = np.linspace(0, 9, 5000)
    _, lambdas = landscape_values(pairs, t)

    total = 0.0
    for k in range(lambdas.shape[0]):
        layer_area = np.trapezoid(np.abs(lambdas[k]), t)
        print(f"Layer lambda_{k+1}: approximate integral of |lambda_{k+1}(t)| dt = {layer_area:.6f}")
        total += layer_area

    print(f"Total approximate L1 norm of the landscape = {total:.6f}")
    print()
    print("Because each layer here is nonnegative, this is just the total area under lambda_1, lambda_2, and lambda_3.")
    print()
    print("For a sanity check, the exact geometric areas are:")
    print("  area(lambda_1) = 8.75")
    print("  area(lambda_2) = 3.00")
    print("  area(lambda_3) = 0.25")
    print("  exact total    = 12.00")
    print("This also matches the identity:")
    print("  sum_k lambda_k(t) = f1(t) + f2(t) + f3(t)  pointwise,")
    print("so the total area under all landscape layers must equal the total area")
    print("under the three original tents.")


def plot_landscape_area(pairs):
    t = np.linspace(0, 9, 2400)
    _, lambdas = landscape_values(pairs, t)

    for k in range(lambdas.shape[0]):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, lambdas[k])
        ax.fill_between(t, 0, np.abs(lambdas[k]), alpha=0.3)
        ax.set_title(f"L1 contribution from lambda_{k+1}(t): area under |lambda_{k+1}|")
        ax.set_xlabel("t")
        ax.set_ylabel("height")
        ax.set_xlim(0, 9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = OUTDIR / f"lambda_{k+1}_L1_area.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def numerical_L1_distance_demo():
    print_header("6) OPTIONAL: L1 DISTANCE BETWEEN TWO LANDSCAPES")
    print("To go one step further, we compare our original diagram against a second diagram.")
    print("Original pairs:  (1,5), (2,6), (4,8)")
    print("Second pairs:    (1.2,4.8), (2.1,5.8), (4.2,7.9)")
    print()
    print("We compute")
    print("d_1(lambda, mu) = sum_k integral |lambda_k(t) - mu_k(t)| dt")
    print()

    pairs_a = [(1, 5), (2, 6), (4, 8)]
    pairs_b = [(1.2, 4.8), (2.1, 5.8), (4.2, 7.9)]

    t = np.linspace(0, 9, 5000)
    _, la = landscape_values(pairs_a, t)
    _, lb = landscape_values(pairs_b, t)

    n_layers = max(la.shape[0], lb.shape[0])
    if la.shape[0] < n_layers:
        la = np.vstack([la, np.zeros((n_layers - la.shape[0], la.shape[1]))])
    if lb.shape[0] < n_layers:
        lb = np.vstack([lb, np.zeros((n_layers - lb.shape[0], lb.shape[1]))])

    total = 0.0
    for k in range(n_layers):
        dist_k = np.trapezoid(np.abs(la[k] - lb[k]), t)
        print(f"Layer {k+1}: approximate integral |lambda_{k+1} - mu_{k+1}| = {dist_k:.6f}")
        total += dist_k

    print(f"Approximate total L1 distance between the two landscapes = {total:.6f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, la[0], label="lambda_1 (original)")
    ax.plot(t, lb[0], label="mu_1 (second diagram)")
    ax.set_title("First landscape layers of two diagrams")
    ax.set_xlabel("t")
    ax.set_ylabel("height")
    ax.set_xlim(0, 9)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = OUTDIR / "L1_distance_lambda1_compare.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print_header("PERSISTENCE LANDSCAPE DEMO")
    print("This script works from the beginning using the three intervals:")
    print(PAIRS)
    print("It explains each step, computes the tent functions, builds the persistence landscape,")
    print("plots the results, and then computes an L1 norm and an optional L1 distance example.")

    explain_pairs(PAIRS)
    sample_point_explanations(PAIRS)
    breakpoint_analysis(PAIRS)

    print_header("4) SAVING PLOTS")
    for i, pair in enumerate(PAIRS, start=1):
        plot_single_tent(pair, i)
    plot_overlay(PAIRS)
    plot_landscape_layers(PAIRS)

    report_piecewise_landscape()

    print_header("5) SAVING L1 AREA PLOTS")
    plot_landscape_area(PAIRS)

    numerical_L1_norm(PAIRS)
    numerical_L1_distance_demo()

    print_header("DONE")
    print(f"All images were saved in: {OUTDIR}")


if __name__ == "__main__":
    main()
