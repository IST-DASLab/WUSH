import numpy as np
import sympy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle


def closed_form_slope_exprs():
    # solve (a^2 + b^2 + c^2 + d^2) * max{(a * x + b * y)^2, (c * x + d * y)^2} = 2 * (a * d - b * c)^2 * max{x^2, y^2}
    # y = m * x
    a, b, c, d, m = sp.symbols("a b c d m", real=True)
    S = a**2 + b**2 + c**2 + d**2
    Delta = a*d - b*c
    K = sp.simplify(2*Delta**2 / S)

    eqs = [
        sp.Eq((a + b*m)**2, K),
        sp.Eq((c + d*m)**2, K),
        sp.Eq((a + b*m)**2, K*m**2),
        sp.Eq((c + d*m)**2, K*m**2),
    ]

    exprs = []
    for eq in eqs:
        exprs += sp.solve(eq, m)

    exprs = [sp.simplify(e) for e in exprs]
    exprs = list(dict.fromkeys(exprs))  # dedupe
    return (a, b, c, d), exprs


def rank_of_2x2(a, b, c, d, tol=1e-12):
    M = np.array([[a, b], [c, d]], dtype=float)
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol).sum())


def valid_slopes_from_exprs(a0, b0, c0, d0, syms, exprs, tol=1e-9):
    a0, b0, c0, d0 = map(float, [a0, b0, c0, d0])
    a_sym, b_sym, c_sym, d_sym = syms

    S = a0*a0 + b0*b0 + c0*c0 + d0*d0
    Delta = a0*d0 - b0*c0

    if abs(S) < tol:
        return {"all_lines": True}

    K = 2*(Delta**2) / S

    # Δ=0 special handling
    if abs(Delta) < tol:
        r = rank_of_2x2(a0, b0, c0, d0)
        if r == 2:
            return {"slopes": [], "vertical": False, "note": "Δ=0, rank=2: only origin; no lines."}
        if r == 0:
            return {"all_lines": True}
        A, B = (a0, b0) if (abs(a0) + abs(b0) > 0) else (c0, d0)
        if abs(B) < tol:
            return {"slopes": [], "vertical": True, "note": "Δ=0, rank=1: vertical nullspace line."}
        return {"slopes": [-A/B], "vertical": False, "note": "Δ=0, rank=1: single nullspace line."}

    def ok(m):
        L1 = (a0 + b0*m)**2
        L2 = (c0 + d0*m)**2
        lhs = max(L1, L2)
        rhs = K * max(1.0, m*m)
        return abs(lhs - rhs) <= tol * max(1.0, lhs, rhs)

    # IMPORTANT: substitute using symbols, not strings
    subs = {a_sym: a0, b_sym: b0, c_sym: c0, d_sym: d0}

    ms = []
    expr_map = {}

    for e in exprs:
        try:
            ev = sp.N(e.subs(subs), 80)  # high precision
            # If anything symbolic remains, skip
            if ev.free_symbols:
                continue
            evc = complex(ev)  # convert to python complex
        except Exception:
            continue

        if abs(evc.imag) < 1e-10 and np.isfinite(evc.real):
            mval = float(evc.real)
            if ok(mval):
                ms.append(mval)
                expr_map[mval] = e  # best-effort label

    # numeric dedupe
    ms_sorted = []
    for mval in sorted(ms):
        if not ms_sorted or abs(mval - ms_sorted[-1]) > 1e-7:
            ms_sorted.append(mval)

    # vertical line condition: max(b^2,d^2) == K
    vertical = abs(max(b0*b0, d0*d0) - K) <= tol * max(1.0, K)

    return {
        "slopes": ms_sorted,
        "vertical": vertical,
        "K": K,
        "expr_map": expr_map,
        "note": f"Found {len(ms_sorted)} slope(s) + vertical={vertical}."
    }


def build_sympy_implicit_expr(a0, b0, c0, d0):
    x, y = sp.symbols("x y", real=True)
    a, b, c, d = map(sp.nsimplify, [a0, b0, c0, d0])
    S = a*a + b*b + c*c + d*d
    Delta = a*d - b*c
    F = S*sp.Max((a*x + b*y)**2, (c*x + d*y)**2) - 2*(Delta**2)*sp.Max(x**2, y**2)
    return sp.lambdify((x, y), F, modules=[{"Max": np.maximum}, "numpy"])


def plot_closed_form_lines(a0, b0, c0, d0,
                           xlim=(-5, 5), ylim=(-5, 5),
                           show_contour=True, contour_grid=400,
                           show_legend=True):
    syms, exprs = closed_form_slope_exprs()
    res = valid_slopes_from_exprs(a0, b0, c0, d0, syms, exprs)

    if res.get("all_lines", False):
        print("Degenerate case: equation holds on all lines (infinitely many).")
        return

    print(res.get("note", ""))
    slopes = res.get("slopes", [])
    vertical = res.get("vertical", False)

    fig, ax = plt.subplots(figsize=(6, 6))
    xs = np.linspace(xlim[0], xlim[1], 600)

    for mval in slopes:
        # optional symbolic label (can get long); fall back to numeric slope
        e = res.get("expr_map", {}).get(mval, None)
        if e is not None:
            label = r"$m=" + sp.latex(sp.simplify(e)) + r"$"
        else:
            label = f"m={mval:.6g}"
        ax.plot(xs, mval*xs, linewidth=2, label=label)

    if vertical:
        ax.plot([0, 0], [ylim[0], ylim[1]], linewidth=2, label=r"$x=0$")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Closed-form solution lines")

    if show_contour:
        F_np = build_sympy_implicit_expr(a0, b0, c0, d0)
        X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], contour_grid),
                           np.linspace(ylim[0], ylim[1], contour_grid))
        F = F_np(X, Y)
        ax.contour(X, Y, F, levels=[0], linestyles="dashed", linewidths=1)

    if show_legend:
        ax.legend(loc="best", fontsize=8)

    plt.show()


def get_slopes_from_transform(transform):
    syms, exprs = closed_form_slope_exprs()
    res = valid_slopes_from_exprs(transform[0, 0], transform[0, 1], transform[1, 0], transform[1, 1], syms, exprs)
    if res.get("all_lines", False):
        print("Degenerate case: equation holds on all lines (infinitely many).")
        slopes = []
    else:
        print(res.get("note", ""))
        slopes = res.get("slopes", [])
        if res.get("vertical", False):
            slopes.append(np.inf)
    return np.asarray(slopes)


class CmapLegend:
    def __init__(self, cmap, norm=None, n=64, orientation="horizontal"):
        self.cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        self.norm = norm
        self.n = n
        self.orientation = orientation


class HandlerCmap(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        n = orig_handle.n
        cmap = orig_handle.cmap
        norm = orig_handle.norm or mpl.colors.Normalize(0, 1)

        vmin = getattr(norm, "vmin", 0.0)
        vmax = getattr(norm, "vmax", 1.0)
        vals = np.linspace(vmin, vmax, n)
        colors = cmap(norm(vals))

        artists = []
        for i, c in enumerate(colors):
            artists.append(Rectangle(
                (xdescent + width * (i / n), ydescent),
                width / n, height, transform=trans,
                facecolor=c, edgecolor="none"
            ))
        artists.append(Rectangle(
            (xdescent, ydescent), width, height, transform=trans,
            facecolor="none", edgecolor="0.3", linewidth=0.8
        ))
        return artists


if __name__ == "__main__":
    # Demo: visualize the closed-form equilibrium lines for an example transform.
    a0, b0, c0, d0 = 2. ** -.5, 2. ** -.5, 2. ** -.5, -2. ** -.5
    plot_closed_form_lines(a0, b0, c0, d0, show_contour=True)
