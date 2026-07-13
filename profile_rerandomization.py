from __future__ import annotations

import os
import time

import numpy as np
from numba import njit, set_num_threads

import sre_real_data_simulation_accelerated_final as sim


@njit
def local_rep_ok_segment(X, z, offset, n, n1, threshold_t):
    K = X.shape[1]
    n0 = n - n1
    sx = np.zeros(K)
    sx2 = np.zeros(K)
    total = np.zeros(K)
    total2 = np.zeros(K)
    for ii in range(n):
        i = offset + ii
        for a in range(K):
            x = X[i, a]
            total[a] += x
            total2[a] += x * x
            if z[i] == 1:
                sx[a] += x
                sx2[a] += x * x
    for a in range(K):
        sx0 = total[a] - sx[a]
        sx20 = total2[a] - sx2[a]
        d = sx[a] / n1 - sx0 / n0
        ss1 = sx2[a] - sx[a] * sx[a] / n1
        ss0 = sx20 - sx0 * sx0 / n0
        den = np.sqrt(max(ss1, 0.0) / (n1 * n1) + max(ss0, 0.0) / (n0 * n0))
        if den > 0 and abs(d) / den > threshold_t:
            return False
    return True


@njit
def local_rep_t_segment(X, z, offset, n, n1):
    K = X.shape[1]
    n0 = n - n1
    sx = np.zeros(K)
    sx2 = np.zeros(K)
    total = np.zeros(K)
    total2 = np.zeros(K)
    for ii in range(n):
        i = offset + ii
        for a in range(K):
            x = X[i, a]
            total[a] += x
            total2[a] += x * x
            if z[i] == 1:
                sx[a] += x
                sx2[a] += x * x
    max_t = 0.0
    for a in range(K):
        sx0 = total[a] - sx[a]
        sx20 = total2[a] - sx2[a]
        d = sx[a] / n1 - sx0 / n0
        ss1 = sx2[a] - sx[a] * sx[a] / n1
        ss0 = sx20 - sx0 * sx0 / n0
        den = np.sqrt(max(ss1, 0.0) / (n1 * n1) + max(ss0, 0.0) / (n0 * n0))
        if den > 0:
            t = abs(d) / den
            if t > max_t:
                max_t = t
    return max_t


@njit
def profile_rem_segment_once(X, offset, n, n1, invcov, threshold, seed, max_attempts):
    N = X.shape[0]
    z = np.zeros(N, dtype=np.uint8)
    index = np.empty(n, dtype=np.int64)
    state = np.uint64(seed | np.uint64(1))
    for attempts in range(1, max_attempts + 1):
        state = sim.draw_segment(z, offset, n, n1, index, state)
        if sim.rem_ok_segment(X, z, offset, n, n1, invcov, threshold):
            return attempts, True
    return max_attempts, False


@njit
def profile_rep_segment_once(X, offset, n, n1, threshold, seed, max_attempts):
    N = X.shape[0]
    z = np.zeros(N, dtype=np.uint8)
    index = np.empty(n, dtype=np.int64)
    state = np.uint64(seed | np.uint64(1))
    for attempts in range(1, max_attempts + 1):
        state = sim.draw_segment(z, offset, n, n1, index, state)
        if local_rep_ok_segment(X, z, offset, n, n1, threshold):
            return attempts, True
    return max_attempts, False


@njit
def profile_rep_mn_once(X, nh, nh1, pi, starts, threshold, seed, max_attempts):
    N = X.shape[0]
    H = nh.shape[0]
    maxn = np.max(nh)
    z = np.zeros(N, dtype=np.uint8)
    index = np.empty(maxn, dtype=np.int64)
    state = np.uint64(seed | np.uint64(1))
    for attempts in range(1, max_attempts + 1):
        for h in range(H):
            state = sim.draw_segment(z, starts[h], nh[h], nh1[h], index, state)
        if sim.t_value_mn_max(X, z, nh, nh1, pi, starts) <= threshold:
            return attempts, True
    return max_attempts, False


@njit
def profile_rem_full_once(X, nh, nh1, starts, invcovs, thresholds, seed, max_attempts):
    H = nh.shape[0]
    N = X.shape[0]
    maxn = np.max(nh)
    z = np.zeros(N, dtype=np.uint8)
    index = np.empty(maxn, dtype=np.int64)
    attempts_by_h = np.zeros(H, dtype=np.int64)
    ok_by_h = np.zeros(H, dtype=np.uint8)
    state = np.uint64(seed | np.uint64(1))
    for h in range(H):
        ok = False
        for attempts in range(1, max_attempts + 1):
            state = sim.draw_segment(z, starts[h], nh[h], nh1[h], index, state)
            if sim.rem_ok_segment(
                X, z, starts[h], nh[h], nh1[h], invcovs[h], thresholds[h]
            ):
                attempts_by_h[h] = attempts
                ok_by_h[h] = 1
                ok = True
                break
        if not ok:
            attempts_by_h[h] = max_attempts
            return attempts_by_h, ok_by_h
    return attempts_by_h, ok_by_h


@njit
def profile_rep_full_once(X, nh, nh1, starts, thresholds, seed, max_attempts):
    H = nh.shape[0]
    N = X.shape[0]
    maxn = np.max(nh)
    z = np.zeros(N, dtype=np.uint8)
    index = np.empty(maxn, dtype=np.int64)
    attempts_by_h = np.zeros(H, dtype=np.int64)
    ok_by_h = np.zeros(H, dtype=np.uint8)
    state = np.uint64(seed | np.uint64(1))
    for h in range(H):
        ok = False
        for attempts in range(1, max_attempts + 1):
            state = sim.draw_segment(z, starts[h], nh[h], nh1[h], index, state)
            if local_rep_ok_segment(X, z, starts[h], nh[h], nh1[h], thresholds[h]):
                attempts_by_h[h] = attempts
                ok_by_h[h] = 1
                ok = True
                break
        if not ok:
            attempts_by_h[h] = max_attempts
            return attempts_by_h, ok_by_h
    return attempts_by_h, ok_by_h


def main():
    p_re = float(os.environ.get("P_RE", "0.001"))
    draws = int(os.environ.get("NUM_THRESHOLD_DRAWS", "1000000"))
    threshold_seed = int(os.environ.get("THRESHOLD_SEED", "1002"))
    trials = int(os.environ.get("PROFILE_TRIALS", "3"))
    max_attempts = int(os.environ.get("MAX_ATTEMPTS", "200000"))
    num_threads = int(os.environ.get("NUM_THREADS", "8"))
    data_path = os.environ.get("DATA_PATH", "SRE_real_data_sharp_null.Rdata")
    threshold_path = os.environ.get(
        "THRESHOLD_CACHE",
        f"sre_thresholds_p{p_re:g}_draws{draws}_seed{threshold_seed}.npz",
    )

    set_num_threads(num_threads)
    Y0, Y1, X, B, K, H, nh, nh1, pi, starts = sim.load_data(data_path)
    cache = np.load(threshold_path)
    a_rem = cache["a_rem_ss"]
    t_rep_ss = cache["t_threshold_rep_ss"]
    t_rep_mn = float(cache["t_threshold_rep_mn"][0])

    invcovs = np.zeros((H, K, K), dtype=np.float64)
    for h in range(H):
        s0 = starts[h]
        invcovs[h] = np.linalg.pinv(
            np.cov(X[s0 : s0 + nh[h]], rowvar=False, ddof=1),
            rcond=1e-12,
        )

    print(
        f"Profiling p={p_re:g}, trials={trials}, max_attempts={max_attempts}, "
        f"threads={num_threads}",
        flush=True,
    )
    print(f"threshold cache: {threshold_path}", flush=True)

    print("Compiling profiling kernels...", flush=True)
    profile_rem_segment_once(
        X, starts[0], nh[0], nh1[0], invcovs[0], a_rem[0], np.uint64(11), 1
    )
    profile_rep_segment_once(
        X, starts[0], nh[0], nh1[0], t_rep_ss[0], np.uint64(13), 1
    )
    profile_rep_mn_once(X, nh, nh1, pi, starts, t_rep_mn, np.uint64(17), 1)
    profile_rem_full_once(
        X, nh, nh1, starts, invcovs, a_rem, np.uint64(19), 1
    )
    profile_rep_full_once(X, nh, nh1, starts, t_rep_ss, np.uint64(23), 1)
    print("Compilation done.\n", flush=True)

    rng = np.random.default_rng(20260713)

    print("SS-ReM by stratum", flush=True)
    for h in range(H):
        rows = []
        for _ in range(trials):
            seed = np.uint64(rng.integers(1, 2**63 - 1, dtype=np.uint64))
            t0 = time.perf_counter()
            attempts, ok = profile_rem_segment_once(
                X,
                starts[h],
                nh[h],
                nh1[h],
                invcovs[h],
                a_rem[h],
                seed,
                max_attempts,
            )
            rows.append((attempts, ok, time.perf_counter() - t0))
        print(
            f"h={h+1}, n={nh[h]}, n1={nh1[h]}, threshold={a_rem[h]:.6g}, "
            f"attempts={[int(r[0]) for r in rows]}, "
            f"ok={[bool(r[1]) for r in rows]}, "
            f"seconds={[round(r[2], 3) for r in rows]}",
            flush=True,
        )

    print("\nSS-ReP by stratum", flush=True)
    for h in range(H):
        rows = []
        for _ in range(trials):
            seed = np.uint64(rng.integers(1, 2**63 - 1, dtype=np.uint64))
            t0 = time.perf_counter()
            attempts, ok = profile_rep_segment_once(
                X,
                starts[h],
                nh[h],
                nh1[h],
                t_rep_ss[h],
                seed,
                max_attempts,
            )
            rows.append((attempts, ok, time.perf_counter() - t0))
        print(
            f"h={h+1}, n={nh[h]}, n1={nh1[h]}, threshold={t_rep_ss[h]:.6g}, "
            f"attempts={[int(r[0]) for r in rows]}, "
            f"ok={[bool(r[1]) for r in rows]}, "
            f"seconds={[round(r[2], 3) for r in rows]}",
            flush=True,
        )

    print("\nFE-ReP whole assignment", flush=True)
    rows = []
    for _ in range(trials):
        seed = np.uint64(rng.integers(1, 2**63 - 1, dtype=np.uint64))
        t0 = time.perf_counter()
        attempts, ok = profile_rep_mn_once(
            X, nh, nh1, pi, starts, t_rep_mn, seed, max_attempts
        )
        rows.append((attempts, ok, time.perf_counter() - t0))
    print(
        f"threshold={t_rep_mn:.6g}, attempts={[int(r[0]) for r in rows]}, "
        f"ok={[bool(r[1]) for r in rows]}, "
        f"seconds={[round(r[2], 3) for r in rows]}",
        flush=True,
    )

    print("\nFull SS-ReM sequence, state carried across strata", flush=True)
    for j in range(trials):
        seed = np.uint64(rng.integers(1, 2**63 - 1, dtype=np.uint64))
        t0 = time.perf_counter()
        attempts, ok = profile_rem_full_once(
            X, nh, nh1, starts, invcovs, a_rem, seed, max_attempts
        )
        print(
            f"trial={j+1}, total_attempts={int(attempts.sum())}, "
            f"attempts={attempts.astype(int).tolist()}, "
            f"ok={ok.astype(int).tolist()}, seconds={time.perf_counter() - t0:.3f}",
            flush=True,
        )

    print("\nFull SS-ReP sequence, state carried across strata", flush=True)
    for j in range(trials):
        seed = np.uint64(rng.integers(1, 2**63 - 1, dtype=np.uint64))
        t0 = time.perf_counter()
        attempts, ok = profile_rep_full_once(
            X, nh, nh1, starts, t_rep_ss, seed, max_attempts
        )
        print(
            f"trial={j+1}, total_attempts={int(attempts.sum())}, "
            f"attempts={attempts.astype(int).tolist()}, "
            f"ok={ok.astype(int).tolist()}, seconds={time.perf_counter() - t0:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
