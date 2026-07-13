from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

import sre_real_data_simulation_accelerated_final as sim


def one_stratum_data(Y: np.ndarray, X: np.ndarray, B: np.ndarray, nh1: np.ndarray, h: int):
    # load_data() converts R stratum labels 1..H to Python labels 0..H-1.
    mask = B == (h - 1)
    Yh = Y[mask].copy()
    Xh = X[mask].copy()
    n = Yh.size
    n1 = int(nh1[h - 1])
    return (
        Yh,
        Xh,
        np.zeros(n, dtype=np.int64),
        np.array([n], dtype=np.int64),
        np.array([n1], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        np.array([0], dtype=np.int64),
    )


def analyze_design(
    Z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    nh: np.ndarray,
    nh1: np.ndarray,
    pi: np.ndarray,
    starts: np.ndarray,
    subset_sizes: np.ndarray,
    subset_indices: np.ndarray,
    total_x: np.ndarray,
    total_xx: np.ndarray,
    total_y: np.ndarray,
    total_xy: np.ndarray,
):
    return sim.analyze_ss_joint_batch(
        Z,
        X,
        Y,
        nh,
        nh1,
        pi,
        starts,
        subset_sizes,
        subset_indices,
        total_x,
        total_xx,
        total_y,
        total_xy,
    )


def main():
    workdir = Path(os.environ.get("WORKDIR", ".")).resolve()
    data_path = Path(
        os.environ.get(
            "DATA_PATH", str(workdir / "SRE_real_data_sharp_null_top4.Rdata")
        )
    )
    num_rep = int(os.environ.get("NUM_REP", "1000"))
    threshold_draws = int(os.environ.get("NUM_THRESHOLD_DRAWS", "1000000"))
    seed = int(os.environ.get("SEED", "2"))
    threshold_seed = int(os.environ.get("THRESHOLD_SEED", str(seed + 1000)))
    num_threads = int(os.environ.get("NUM_THREADS", "8"))
    chunk_size = int(os.environ.get("CHUNK_SIZE", "1000"))
    p_values = [
        float(x)
        for x in os.environ.get("P_VALUES", "0.1,0.01,0.001").split(",")
        if x.strip()
    ]
    output_rdata = Path(
        os.environ.get(
            "OUTPUT_RDATA", str(workdir / f"stratum_cre_B{num_rep}.Rdata")
        )
    )
    output_csv = output_rdata.with_suffix(".csv")

    sim.set_num_threads(num_threads)
    Y0, Y1, X_all, B_all, K, H, nh_all, nh1_all, pi_all, starts_all = sim.load_data(
        data_path
    )
    if not np.allclose(Y0, Y1):
        raise ValueError("Expected sharp-null data with Y0 == Y1.")
    Y_all = Y0.copy()

    print(
        f"Separate-CRE by stratum: H={H}, K={K}, B={num_rep}, "
        f"p_values={p_values}, threads={num_threads}",
        flush=True,
    )

    rows: list[dict[str, float | int | str]] = []
    rdata_output: dict[str, object] = {
        "num_rep": np.array([num_rep]),
        "p_values": np.asarray(p_values, dtype=np.float64),
        "strata": np.arange(1, H + 1, dtype=np.int64),
    }

    global_rng = np.random.default_rng(seed)
    base_seeds = global_rng.integers(1, 2**63 - 1, size=num_rep, dtype=np.uint64)

    # Compile once on the first stratum.
    warm_Y, warm_X, _, warm_nh, warm_nh1, warm_pi, warm_starts = one_stratum_data(
        Y_all, X_all, B_all, nh1_all, 1
    )
    warm_sizes, warm_indices = sim.build_subsets(K)
    warm_tx, warm_txx, warm_ty, warm_txy = sim.build_ss_totals(
        warm_X, warm_Y, warm_nh, warm_starts
    )
    warm_seed = np.array([np.uint64(seed + 98765)], dtype=np.uint64)
    warm_z = sim.generate_sre_numba(warm_seed, warm_nh, warm_nh1, warm_starts, warm_Y.size)
    _ = analyze_design(
        warm_z,
        warm_X,
        warm_Y,
        warm_nh,
        warm_nh1,
        warm_pi,
        warm_starts,
        warm_sizes,
        warm_indices,
        warm_tx,
        warm_txx,
        warm_ty,
        warm_txy,
    )

    start_all = time.perf_counter()
    for h in range(1, H + 1):
        Y, X, B, nh, nh1, pi, starts = one_stratum_data(
            Y_all, X_all, B_all, nh1_all, h
        )
        N = Y.size
        subset_sizes, subset_indices = sim.build_subsets(K)
        total_x, total_xx, total_y, total_xy = sim.build_ss_totals(
            X, Y, nh, starts
        )
        invcovs = np.zeros((1, K, K), dtype=np.float64)
        invcovs[0] = np.linalg.pinv(np.cov(X, rowvar=False, ddof=1), rcond=1e-12)

        print(f"Stratum {h}: n={N}, n1={int(nh1[0])}", flush=True)

        p_cre = np.empty(num_rep, dtype=np.float64)
        for lo in range(0, num_rep, chunk_size):
            hi = min(lo + chunk_size, num_rep)
            z = sim.generate_sre_numba(base_seeds[lo:hi], nh, nh1, starts, N)
            p_cre[lo:hi] = analyze_design(
                z,
                X,
                Y,
                nh,
                nh1,
                pi,
                starts,
                subset_sizes,
                subset_indices,
                total_x,
                total_xx,
                total_y,
                total_xy,
            )
        er = float(np.mean(p_cre <= 0.05))
        rows.append(
            {
                "stratum": h,
                "design": "CRE",
                "accept_prob": 1.0,
                "type_I_error": er,
                "mcse": float(np.sqrt(er * (1.0 - er) / num_rep)),
                "mean_attempts": 1.0,
            }
        )
        rdata_output[f"p_h{h}_CRE"] = p_cre
        print(f"  CRE: ER={er:.4f}", flush=True)

        for p_re in p_values:
            cache_path = workdir / (
                f"stratum_cre_v2_h{h}_thresholds_p{p_re:g}_"
                f"draws{threshold_draws}_seed{threshold_seed}.npz"
            )
            if cache_path.exists():
                cache = np.load(cache_path)
                a_rem = cache["a_rem_ss"]
                t_rep = cache["t_threshold_rep_ss"]
                print(f"  thresholds p={p_re:g}: cache", flush=True)
            else:
                t0 = time.perf_counter()
                a_rem, t_rep, _ = sim.calibrate_thresholds(
                    X=X,
                    nh=nh,
                    nh1=nh1,
                    pi=pi,
                    starts=starts,
                    p_re=p_re,
                    draws=threshold_draws,
                    seed=threshold_seed,
                    batch_size=5000,
                )
                np.savez_compressed(
                    cache_path,
                    a_rem_ss=a_rem,
                    t_threshold_rep_ss=t_rep,
                    p_re=np.array([p_re]),
                    threshold_draws=np.array([threshold_draws]),
                    threshold_seed=np.array([threshold_seed]),
                )
                print(
                    f"  thresholds p={p_re:g}: calibrated in "
                    f"{time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

            for design in ("ReM", "ReP"):
                p_vec = np.empty(num_rep, dtype=np.float64)
                tries = np.empty(num_rep, dtype=np.int64)
                t0 = time.perf_counter()
                for lo in range(0, num_rep, chunk_size):
                    hi = min(lo + chunk_size, num_rep)
                    seeds_chunk = base_seeds[lo:hi]
                    if design == "ReM":
                        z, attempts = sim.generate_rem_ss_numba(
                            seeds_chunk, X, nh, nh1, starts, invcovs, a_rem
                        )
                    else:
                        z, attempts = sim.generate_rep_ss_numba(
                            seeds_chunk, X, nh, nh1, starts, t_rep
                        )
                    tries[lo:hi] = attempts
                    p_vec[lo:hi] = analyze_design(
                        z,
                        X,
                        Y,
                        nh,
                        nh1,
                        pi,
                        starts,
                        subset_sizes,
                        subset_indices,
                        total_x,
                        total_xx,
                        total_y,
                        total_xy,
                    )
                er = float(np.mean(p_vec <= 0.05))
                rows.append(
                    {
                        "stratum": h,
                        "design": design,
                        "accept_prob": p_re,
                        "type_I_error": er,
                        "mcse": float(np.sqrt(er * (1.0 - er) / num_rep)),
                        "mean_attempts": float(np.mean(tries)),
                    }
                )
                rdata_output[f"p_h{h}_{design}_p{p_re:g}"] = p_vec
                print(
                    f"  {design} p={p_re:g}: ER={er:.4f}, "
                    f"mean_attempts={np.mean(tries):.1f}, "
                    f"seconds={time.perf_counter() - t0:.1f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    rdata_output["df"] = df
    rdata.write_rda(output_rdata, rdata_output, compression="gzip")
    print(f"Saved CSV: {output_csv}", flush=True)
    print(f"Saved RData: {output_rdata}", flush=True)
    print(f"Total seconds: {time.perf_counter() - start_all:.1f}", flush=True)


if __name__ == "__main__":
    main()
