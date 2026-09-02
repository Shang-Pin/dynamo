# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregated (chunked prefill + decode) engine performance model.

Regression:  wall_time = f(sum_prefill_tokens, sum_decode_kv_tokens)
"""

import logging
import math
import statistics
from collections import deque
from typing import Optional

import numpy as np

from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.planner.core.perf_model.base import (
    _BaseRegressionModel,
    _clamp_kv_hit_rate,
    _MovingAverage,
)

logger = logging.getLogger(__name__)


class AggRegressionModel(_BaseRegressionModel):
    """2D regression for aggregated (chunked prefill + decode) engines.

    DEEPINFRA: ``wall_time`` on an agg engine is a bimodal mixture. An
    iteration that carries a chunked-prefill payload costs several times a
    pure decode step, so a single fit over both regimes predicts the mixture
    mean and is unusable for ITL. Measured on DeepSeek-V4-Flash (1588 live
    FPM samples, 1xB300, dspark nextn=7):

      * pure-decode iterations (``sum_prefill_tokens == 0``): median 0.060s
      * iterations with a prefill chunk:                       median 0.228s
      * the shipped ``f(sum_prefill_tokens, sum_decode_kv_tokens)`` fit:
        R^2=0.595 with **109% mean absolute percentage error**

    and every bit of that fit's signal is on the prefill axis:
    ``sum_decode_kv_tokens`` correlates r=-0.011 with wall_time over the
    whole mixture (R^2=0.000 on its own). So an ITL nominally derived from
    resident KV was in practice driven by prefill work.

    That near-zero raw correlation is a measurement artifact, not physics --
    see below; the KV trend is real, just small relative to the tail.

    The fix is regime separation: decode-side quantities (ITL and the decode
    egress rate) come from pure-decode observations only, while the mixed fit
    is kept for prefill-side work (TTFT and prefill admission) -- which is
    what its dominant axis actually measures.

    Within the decode regime KV *does* still matter, but the trend is small
    and buried under a heavy tail, so it has to be fitted robustly. On the
    pure-decode subset:

      * raw least squares sees nothing (R^2=0.002) and, fitted directly,
        predicts 288ms per forward against 20.2ms measured -- the tail
        (median 0.060s, mean 0.122s, max 0.925s) captures the fit;
      * medians within equal-count KV bins recover a clear trend:
        r=+0.720, R^2=0.518, slope ~2.4e-9 s/token, intercept ~0.056s,
        stable across 4/6/8 bins;
      * holding ``num_decode_requests`` near its mode gives R^2=0.629 with
        slope 1.85e-9, confirming the effect is resident KV rather than
        batch size (the two correlate only r=-0.069).

    So the step cost is ``intercept + slope * decode_kv`` fitted over bin
    medians. KV accounts for ~12% of the step across the observed range
    (57ms at 471k tokens to 64ms at 3.25M), which matters: it is the
    difference between an ITL estimate that tracks load and one that is
    blind to it.

    The slope's *sign* is dependable (75% of bootstrap resamples positive,
    and every robust estimator agrees) but its *magnitude* is not -- the
    bootstrap 95% CI spans [-1.5e-8, +7.6e-8] and half-sample splits give
    3.4e-8/5.8e-8. An ill-determined slope is exactly how the 288ms
    prediction happened, so predictions are clamped to measured territory:
    ``decode_kv`` is clipped to the observed range and the result to the
    p10..p90 of observed pure-decode wall_time. Both historical failures
    here were extrapolations -- down to 0.75ms, up to 288ms -- and the clamp
    is what makes either impossible.

    Beyond the observed KV range the estimate therefore flattens rather than
    extrapolating; the KV-saturation thresholds
    (``decode_kv_scale_up_threshold``) are what cover a cache actually
    filling up.
    """

    def __init__(
        self,
        max_num_fpm_samples: int,
        min_observations: int = 5,
        bucket_count: int = 16,
    ):
        super().__init__(
            max_num_fpm_samples, min_observations, ndim=2, bucket_count=bucket_count
        )
        self._avg_isl = _MovingAverage(max_num_fpm_samples)
        self._avg_decode_len = _MovingAverage(max_num_fpm_samples)
        self._avg_prefill_tokens = _MovingAverage(max_num_fpm_samples)
        self._avg_num_prefill = _MovingAverage(max_num_fpm_samples)
        self._avg_num_decode = _MovingAverage(max_num_fpm_samples)
        # Pure-decode (decode_kv, wall_time) pairs, newest last.
        self._decode_steps: deque[tuple[float, float]] = deque(
            maxlen=max_num_fpm_samples
        )
        self._min_decode_steps = min_observations
        self._decode_fit: Optional[tuple[Optional[float], float]] = None
        self._decode_bounds: Optional[tuple[float, float, float, float]] = None

    # Bucket medians are fitted across this many equal-count KV bins. Chosen
    # from the slope-stability sweep on the live samples: 4/6/8 bins all
    # recover slope ~2.4e-9, while >=10 bins leave too few samples per bin
    # for the median to settle and the slope jumps 10x (see class docstring).
    _DECODE_FIT_BINS = 6

    def add_observation(self, fpm: ForwardPassMetrics) -> None:
        super().add_observation(fpm)
        # A pure-decode iteration is the only sample that says what a decode
        # step costs; anything carrying a prefill chunk belongs to the other
        # regime and would bias the estimate upward. wall_time == 0 is an idle
        # heartbeat, not a measurement.
        sched = fpm.scheduled_requests
        if (
            fpm.wall_time > 0.0
            and sched.sum_prefill_tokens == 0
            and sched.num_decode_requests > 0
        ):
            self._decode_steps.append(
                (float(sched.sum_decode_kv_tokens), float(fpm.wall_time))
            )
            self._decode_fit = None
            self._decode_bounds = None

    @property
    def num_decode_step_observations(self) -> int:
        return len(self._decode_steps)

    def _ensure_decode_fit(self) -> None:
        """Robust least squares over per-bin medians of the decode regime.

        Raw least squares cannot be used here: pure-decode wall_time is
        heavy-tailed, and fitting it directly scored R^2=0.002 while
        predicting 288ms per forward against 20.2ms measured. Taking the
        median within equal-count KV bins first suppresses the tail and
        recovers the underlying trend (r=+0.720, R^2=0.518 across bins; and
        R^2=0.629 once ``num_decode_requests`` is held near its mode, which
        confirms the effect is KV and not batch size -- the two correlate
        only r=-0.069).

        A non-positive slope is unphysical (more resident KV cannot make a
        decode step cheaper) and is treated as "no usable trend", falling
        back to a flat level.
        """
        if self._decode_fit is not None or not self._decode_steps:
            return
        xs = np.array([kv for kv, _ in self._decode_steps], dtype=float)
        ys = np.array([wt for _, wt in self._decode_steps], dtype=float)
        # Extrapolation guards: never predict outside measured territory.
        # Both historical failure modes here were extrapolations -- down to
        # 0.75ms from the mixed fit, up to 288ms from raw least squares.
        # Clipping decode_kv to the observed range is the real guard; the
        # wall-time bound is an absolute backstop against a pathological
        # slope, so it uses observed extremes rather than quantiles -- inner
        # quantiles would compress predictions inside the operating range.
        self._decode_bounds = (
            float(xs.min()),
            float(xs.max()),
            float(ys.min()),
            float(ys.max()),
        )
        level = float(statistics.median(ys))

        edges = np.quantile(xs, np.linspace(0.0, 1.0, self._DECODE_FIT_BINS + 1))
        bin_x: list[float] = []
        bin_y: list[float] = []
        for i in range(self._DECODE_FIT_BINS):
            hi = edges[i + 1]
            in_bin = (
                (xs >= edges[i]) & (xs <= hi)
                if i == self._DECODE_FIT_BINS - 1
                else (xs >= edges[i]) & (xs < hi)
            )
            if in_bin.sum() >= 3:
                bin_x.append(float(np.median(xs[in_bin])))
                bin_y.append(float(np.median(ys[in_bin])))

        if len(bin_x) < 3 or len(set(bin_x)) < 3:
            self._decode_fit = (None, level)
            return
        slope, intercept = np.polyfit(np.array(bin_x), np.array(bin_y), 1)
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.0:
            self._decode_fit = (None, level)
            return
        self._decode_fit = (float(slope), float(intercept))

    def _decode_step_seconds(
        self, decode_kv_tokens: Optional[float] = None
    ) -> Optional[float]:
        """Pure-decode step time at ``decode_kv_tokens``.

        Returns ``None`` before enough pure-decode samples have arrived, so
        callers fall back to the mixed fit and a cold start degrades to the
        previous behaviour rather than failing.
        """
        if len(self._decode_steps) < self._min_decode_steps:
            return None
        self._ensure_decode_fit()
        if self._decode_fit is None or self._decode_bounds is None:
            return None
        slope, intercept = self._decode_fit
        if slope is None or decode_kv_tokens is None:
            return intercept
        kv_lo, kv_hi, wt_lo, wt_hi = self._decode_bounds
        kv = min(max(float(decode_kv_tokens), kv_lo), kv_hi)
        return min(max(intercept + slope * kv, wt_lo), wt_hi)

    def _extract_x(self, fpm: ForwardPassMetrics) -> list[float]:
        sched = fpm.scheduled_requests
        return [float(sched.sum_prefill_tokens), float(sched.sum_decode_kv_tokens)]

    def _update_moving_averages(self, fpm: ForwardPassMetrics) -> None:
        sched = fpm.scheduled_requests
        if sched.num_prefill_requests > 0:
            self._avg_isl.add(sched.sum_prefill_tokens / sched.num_prefill_requests)
        if sched.num_decode_requests > 0:
            self._avg_decode_len.add(
                sched.sum_decode_kv_tokens / sched.num_decode_requests
            )
        self._avg_prefill_tokens.add(float(sched.sum_prefill_tokens))
        self._avg_num_prefill.add(float(sched.num_prefill_requests))
        self._avg_num_decode.add(float(sched.num_decode_requests))

    @property
    def avg_isl(self) -> float:
        return self._avg_isl.value

    @property
    def avg_decode_length(self) -> float:
        return self._avg_decode_len.value

    @property
    def avg_prefill_tokens(self) -> float:
        return self._avg_prefill_tokens.value

    def _predict_2d(self, prefill_tokens: float, decode_kv_tokens: float) -> float:
        return max(
            1e-6,
            float(
                self._model.predict(np.array([[prefill_tokens, decode_kv_tokens]]))[0]
            ),
        )

    def estimate_next_ttft(
        self,
        queued_prefill_tokens: int,
        max_num_batched_tokens: int,
        current_decode_kv: int,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[float]:
        """Simulate prefill scheduling with piggybacked decode.

        ``kv_hit_rate`` (0.0-1.0) discounts the aggregate work ahead --
        both the queue backlog and the hypothetical next request's ISL --
        because a new arrival will benefit from the same prefix-cache hit
        rate as the current workload. See ``PrefillRegressionModel``.

        Returns estimated TTFT in seconds, or None if the model is not ready.
        """
        if not self._ensure_fitted() or max_num_batched_tokens <= 0:
            return None

        scale = 1.0 - _clamp_kv_hit_rate(kv_hit_rate)
        total_tokens = (queued_prefill_tokens + self._avg_isl.value) * scale
        if total_tokens <= 0:
            return 0.0

        num_iterations = math.ceil(total_tokens / max_num_batched_tokens)
        total_time = 0.0
        remaining = total_tokens
        for _ in range(num_iterations):
            chunk = min(remaining, max_num_batched_tokens)
            total_time += self._predict_2d(chunk, float(current_decode_kv))
            remaining -= chunk
        return total_time

    def estimate_next_itl(
        self,
        scheduled_decode_kv: int,
        queued_decode_kv: int,
    ) -> Optional[float]:
        """Estimate the next decode iteration time in seconds.

        Served by the decode-regime fit (see class docstring); falls back to
        the mixed fit until enough pure-decode samples have arrived. Returns
        ``None`` if neither estimator is ready.
        """
        total_kv = scheduled_decode_kv + queued_decode_kv + self._avg_decode_len.value
        step = self._decode_step_seconds(total_kv)
        if step is not None:
            return step
        if not self._ensure_fitted():
            return None
        return self._predict_2d(self._avg_prefill_tokens.value, total_kv)

    def find_best_engine_agg_rps(
        self,
        isl: float,
        osl: float,
        max_num_batched_tokens: int,
        ttft_sla: float,
        itl_sla: float,
        max_kv_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        kv_hit_rate: Optional[float] = None,
        accept_length: float = 1.0,
    ) -> tuple[float, float, float]:
        """Find the maximum agg engine request rate under both SLA targets.

        Sweeps over batch_size to find the largest decode concurrency
        where both ITL and TTFT remain within their targets.  Warns if
        even batch_size=1 violates either SLA.

        Decode egress rate is derived via Little's law, with speculative
        accept length reducing the number of decode forwards needed per raw
        output length:
        ``decode_rps = best_batch_size * accept_length / (osl * wall_time)``.
        The final engine RPS is capped by prefill admission capacity so
        aggregated engines do not claim decode-side speedup that prefill cannot
        sustain.

        ``kv_hit_rate`` discounts only the prefill portion of each
        iteration; decode KV residency uses the full ISL because cache
        hits reduce prefill compute but do not shrink the KV footprint
        used during decode.

        The upper bound for the batch-size sweep is the smallest of:
          1. KV cache capacity: ``max_kv_tokens / (isl + osl/2)``
          2. ``max_num_seqs`` (engine concurrency limit)
          3. The prefill/decode rate-balance point (steady state).  For a
             batch of size ``x``:
               - Decode egress rate: ``x / osl`` requests finish per iter
                 (x concurrent streams, each taking osl decode iters).
               - Prefill admission rate: ``(max_num_batched_tokens - x) / isl``
                 requests admitted per iter (the budget left after decode
                 takes one slot per in-flight request, divided by isl tokens
                 per new request).
             Steady state requires admission >= egress:
               ``(max_num_batched_tokens - x) / isl >= x / osl``,
             which simplifies to
               ``isl / (max_num_batched_tokens - x) <= osl``
             (the check implemented below), or equivalently
               ``x <= osl * max_num_batched_tokens / (isl + osl)``.
             Above this, prefill becomes the bottleneck and TTFT grows
             unbounded.

        The caller guarantees ``osl > 0`` and ``max_num_batched_tokens > 0``
        via the early-return validation above.
        """
        if (
            not self._ensure_fitted()
            or isl <= 0
            or osl <= 0
            or max_num_batched_tokens <= 0
        ):
            return (0.0, 0.0, 0.0)
        accept_length = max(1.0, float(accept_length))

        prefill_scale = 1.0 - _clamp_kv_hit_rate(kv_hit_rate)
        effective_isl = isl * prefill_scale

        # DEEPINFRA: KV residency per decode slot, measured.
        #
        # ``isl + osl/2`` is arrival-weighted -- every request counts once
        # when it shows up -- but a decode slot is held for the whole
        # generation, so the resident population is length-biased: long
        # requests carry more KV *and* occupy their slot longer. Sampling by
        # residency yields E[L^2]/E[L], not E[L], and on a skewed workload
        # those differ badly. Measured on DeepSeek-V4-Flash (ISL p50=801,
        # p90=8.6k, p99=115k, mean 5.6k): arrival-weighted avg_ctx said 5,621
        # tokens/slot while observed sum_decode_kv_tokens/num_decode_requests
        # was 19,426 -- a 3.5x understatement (implied CV~1.58), which put
        # every probe of the regression 6-32x below the operating range.
        #
        # ``_avg_decode_len`` is exactly that residency-weighted mean and is
        # already maintained from every FPM sample. Prefer it; fall back to
        # the arrival-weighted proxy only before any decode sample has
        # arrived. The bias factor scales with workload skew, so no constant
        # correction to the proxy would work across models.
        avg_ctx = self._avg_decode_len.value or (isl + osl / 2.0)

        # KV cache cap
        kv_cap = (
            max(1, int(max_kv_tokens / max(1.0, avg_ctx)))
            if max_kv_tokens and max_kv_tokens > 0
            else 1024  # large fallback when capability not known
        )
        # Concurrency cap
        seq_cap = max_num_seqs if max_num_seqs and max_num_seqs > 0 else kv_cap

        # Prefill/decode balance cap via binary search within [1, min(kv_cap, seq_cap)].
        # For each candidate x, check that the per-forward prefill work needed
        # to admit requests at the speculative decode egress rate fits in the
        # remaining token budget:
        #   x * accept_length * effective_isl / osl <= max_num_batched_tokens - x
        # Uses ``effective_isl`` (post-cache) because cache reuse shrinks the
        # prefill tokens each new request consumes from the per-iteration
        # budget, raising the admissible batch size.
        hard_cap = min(kv_cap, seq_cap, max_num_batched_tokens - 1)

        def _prefill_balanced(x: int) -> bool:
            prefill_budget = max_num_batched_tokens - x
            if prefill_budget <= 0:
                return False
            required_prefill = x * accept_length * effective_isl / max(1.0, osl)
            return required_prefill <= prefill_budget

        lo, hi = 1, max(1, hard_cap)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _prefill_balanced(mid):
                lo = mid
            else:
                hi = mid - 1
        max_bs = lo

        best_rps = 0.0
        best_ttft_ms = 0.0
        best_itl_ms = 0.0

        for bs in range(1, max_bs + 1):
            decode_kv = bs * avg_ctx
            # Discounted prefill per iter feeds the wall-time regression: the
            # engine actually computes ``effective_isl`` tokens per request
            # because the cached prefix is skipped.
            prefill_per_iter = min(
                bs * accept_length * effective_isl / max(1.0, osl),
                max_num_batched_tokens,
            )
            wt = self._predict_2d(prefill_per_iter, decode_kv)
            # Decode-side quantities (ITL, decode egress) come from the
            # decode-regime fit; ``wt`` is a mixed-iteration cost dominated by
            # prefill work and only describes prefill admission. Falls back to
            # ``wt`` before the regime fit is ready.
            decode_wt = self._decode_step_seconds(decode_kv) or wt
            itl_ms = decode_wt * 1000.0 / accept_length

            # ``estimate_next_ttft`` applies the same discount internally to
            # both the queued portion and the avg_isl portion. To keep the
            # discount uniform, we pass the *raw* prefill_per_iter as the
            # queued contribution and forward ``kv_hit_rate`` so the
            # function's own ``(1 - clamp(kv_hit_rate))`` factor scales
            # both sides consistently.
            raw_prefill_per_iter = min(
                bs * accept_length * isl / max(1.0, osl),
                max_num_batched_tokens,
            )
            est_ttft = self.estimate_next_ttft(
                queued_prefill_tokens=int(raw_prefill_per_iter),
                max_num_batched_tokens=max_num_batched_tokens,
                current_decode_kv=int(decode_kv),
                kv_hit_rate=kv_hit_rate,
            )
            ttft_ms = est_ttft * 1000.0 if est_ttft is not None else 0.0

            if itl_ms > itl_sla or ttft_ms > ttft_sla:
                if bs == 1:
                    logger.warning(
                        f"Agg SLA unreachable at batch_size=1: "
                        f"TTFT={ttft_ms:.1f}ms (target {ttft_sla:.1f}ms), "
                        f"ITL={itl_ms:.1f}ms (target {itl_sla:.1f}ms)"
                    )
                    decode_rps = accept_length / (osl * decode_wt)
                    prefill_budget = max(0.0, float(max_num_batched_tokens - 1))
                    prefill_rps = (
                        math.inf
                        if effective_isl <= 0.0
                        else prefill_budget / (effective_isl * wt)
                    )
                    best_rps = min(decode_rps, prefill_rps)
                    best_ttft_ms = ttft_ms
                    best_itl_ms = itl_ms
                break

            decode_rps = bs * accept_length / (osl * decode_wt)
            prefill_budget = max(0.0, float(max_num_batched_tokens - bs))
            prefill_rps = (
                math.inf
                if effective_isl <= 0.0
                else prefill_budget / (effective_isl * wt)
            )
            best_rps = min(decode_rps, prefill_rps)
            best_ttft_ms = ttft_ms
            best_itl_ms = itl_ms

        return (best_rps, best_ttft_ms, best_itl_ms)
