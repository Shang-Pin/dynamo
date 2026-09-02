# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression model unit tests.

These test the perf_model classes directly (PrefillRegressionModel,
DecodeRegressionModel, AggRegressionModel) without any planner adapter.

FPM-driven scaling integration tests live in the plugin/orchestrator test suite.
"""

import os
from unittest.mock import Mock, patch

import pytest

try:
    import msgspec  # noqa: F401
except ImportError:
    pytest.skip("msgspec required for FPM tests", allow_module_level=True)

try:
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )
except ImportError:
    pytest.skip("forward_pass_metrics not available", allow_module_level=True)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.core.perf_model import (
    AggRegressionModel,
    DecodeRegressionModel,
    PrefillRegressionModel,
)
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _make_fpm(
    *,
    sum_prefill_tokens: int = 0,
    num_prefill_requests: int = 0,
    sum_decode_kv_tokens: int = 0,
    num_decode_requests: int = 0,
    queued_prefill_tokens: int = 0,
    queued_decode_kv_tokens: int = 0,
    wall_time: float = 0.01,
    worker_id: str = "w1",
    dp_rank: int = 0,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=sum_prefill_tokens,
            num_prefill_requests=num_prefill_requests,
            sum_decode_kv_tokens=sum_decode_kv_tokens,
            num_decode_requests=num_decode_requests,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=queued_prefill_tokens,
            sum_decode_kv_tokens=queued_decode_kv_tokens,
        ),
    )


# ── PrefillRegressionModel tests ─────────────────────────────────────


class TestPrefillRegressionModel:
    def test_insufficient_data(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        assert not model.has_sufficient_data()
        assert model.estimate_next_ttft(0, 2048) is None

    def test_heartbeat_skipped(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpm = _make_fpm(wall_time=0.0, sum_prefill_tokens=100, num_prefill_requests=1)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_basic_regression_and_ttft_estimate(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500, 2000, 2500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens + 0.002,
            )
            model.add_observation(fpm)

        assert model.has_sufficient_data()

        est = model.estimate_next_ttft(
            queued_prefill_tokens=0, max_num_batched_tokens=2048
        )
        assert est is not None
        assert est > 0

    def test_chunked_ttft_simulation(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [100, 200, 300, 400, 500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            model.add_observation(fpm)

        est = model.estimate_next_ttft(
            queued_prefill_tokens=5000, max_num_batched_tokens=2048
        )
        assert est is not None
        assert est > 0.003  # at least 3 iterations worth

    def test_avg_isl_tracking(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for isl in [1000, 2000, 3000]:
            fpm = _make_fpm(
                sum_prefill_tokens=isl, num_prefill_requests=1, wall_time=0.01
            )
            model.add_observation(fpm)
        assert abs(model.avg_isl - 2000.0) < 1.0

    def test_idle_fpms_decay_avg_isl_after_traffic(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=3, min_observations=3, bucket_count=16
        )
        model.add_observation(_make_fpm(wall_time=0.0))
        assert model.avg_isl == 0.0

        for isl in [1000, 2000, 3000]:
            model.add_observation(
                _make_fpm(
                    sum_prefill_tokens=isl,
                    num_prefill_requests=1,
                    wall_time=0.01,
                )
            )
        assert abs(model.avg_isl - 2000.0) < 1.0

        for _ in range(3):
            model.add_observation(_make_fpm(wall_time=0.0))
        assert model.avg_isl == 0.0

    def test_find_best_engine_prefill_rps(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500, 2000, 2500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens + 0.002,
            )
            model.add_observation(fpm)

        rps, actual_ttft_ms = model.find_best_engine_prefill_rps(
            ttft_sla=2000.0, isl=1000.0
        )
        assert rps > 0
        assert 0.5 < rps < 2.0
        assert actual_ttft_ms > 0
        assert 1000 < actual_ttft_ms < 2000

    def test_find_best_engine_prefill_rps_zero_isl(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            model.add_observation(fpm)
        rps, _ = model.find_best_engine_prefill_rps(ttft_sla=1000.0, isl=0.0)
        assert rps == 0.0

    def test_load_benchmark_fpms(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpms = [
            _make_fpm(sum_prefill_tokens=t, num_prefill_requests=1, wall_time=0.001 * t)
            for t in [500, 1000, 1500, 2000, 2500]
        ]
        model.load_benchmark_fpms(fpms)
        assert model.num_observations == 5
        assert model.has_sufficient_data()
        est = model.estimate_next_ttft(
            queued_prefill_tokens=0, max_num_batched_tokens=2048
        )
        assert est is not None

    def test_kv_hit_rate_none_equals_zero(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500, 2000, 2500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens + 0.002,
            )
            model.add_observation(fpm)

        none_est = model.estimate_next_ttft(
            queued_prefill_tokens=3000,
            max_num_batched_tokens=2048,
            kv_hit_rate=None,
        )
        zero_est = model.estimate_next_ttft(
            queued_prefill_tokens=3000,
            max_num_batched_tokens=2048,
            kv_hit_rate=0.0,
        )
        assert none_est == zero_est

    def test_kv_hit_rate_discounts_queued_and_avg_isl(self):
        """A hit rate of 0.5 should halve the simulated work, roughly halving TTFT."""
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        # Fit on several points so the regression is stable and ~linear in tokens.
        for tokens in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            model.add_observation(fpm)

        max_batched = 100_000  # single-iteration regime, no chunking rounding
        est_full = model.estimate_next_ttft(
            queued_prefill_tokens=4000,
            max_num_batched_tokens=max_batched,
            kv_hit_rate=0.0,
        )
        est_half = model.estimate_next_ttft(
            queued_prefill_tokens=4000,
            max_num_batched_tokens=max_batched,
            kv_hit_rate=0.5,
        )
        assert est_full is not None and est_half is not None
        # With a ~linear regression and no chunking rounding, 0.5 discount
        # should produce roughly half the TTFT (within 20% tolerance for
        # linearly-fitted intercept noise).
        assert est_half < est_full
        assert est_half / est_full < 0.75

    def test_kv_hit_rate_clamped(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for tokens in [500, 1000, 1500, 2000, 2500]:
            fpm = _make_fpm(
                sum_prefill_tokens=tokens,
                num_prefill_requests=1,
                wall_time=0.001 * tokens,
            )
            model.add_observation(fpm)

        # kv_hit_rate > 1.0 should clamp to 0.95 (not 1.0) so queued/avg don't
        # fully zero out.
        est_above = model.estimate_next_ttft(
            queued_prefill_tokens=2000,
            max_num_batched_tokens=100_000,
            kv_hit_rate=1.5,
        )
        est_cap = model.estimate_next_ttft(
            queued_prefill_tokens=2000,
            max_num_batched_tokens=100_000,
            kv_hit_rate=0.95,
        )
        assert est_above == est_cap

        # Negative values clamp to 0.0 (no discount).
        est_negative = model.estimate_next_ttft(
            queued_prefill_tokens=2000,
            max_num_batched_tokens=100_000,
            kv_hit_rate=-0.3,
        )
        est_zero = model.estimate_next_ttft(
            queued_prefill_tokens=2000,
            max_num_batched_tokens=100_000,
            kv_hit_rate=0.0,
        )
        assert est_negative == est_zero

        # NaN falls back to 0.0.
        est_nan = model.estimate_next_ttft(
            queued_prefill_tokens=2000,
            max_num_batched_tokens=100_000,
            kv_hit_rate=float("nan"),
        )
        assert est_nan == est_zero


# ── Bucketed retirement tests ─────────────────────────────────────────


class TestBucketedRetirement:
    def test_total_capped_at_max(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=10, min_observations=3, bucket_count=4
        )
        for i in range(20):
            fpm = _make_fpm(
                sum_prefill_tokens=100 * (i + 1),
                num_prefill_requests=1,
                wall_time=0.01 * (i + 1),
            )
            model.add_observation(fpm)
        assert model.num_observations == 10

    def test_most_populated_bucket_loses_oldest(self):
        model = PrefillRegressionModel(
            max_num_fpm_samples=6, min_observations=1, bucket_count=4
        )
        for i in range(3):
            fpm = _make_fpm(
                sum_prefill_tokens=10 + i,
                num_prefill_requests=1,
                wall_time=0.001 * (10 + i),
            )
            model.add_observation(fpm)
        for i in range(3):
            fpm = _make_fpm(
                sum_prefill_tokens=1000 + i * 100,
                num_prefill_requests=1,
                wall_time=0.001 * (1000 + i * 100),
            )
            model.add_observation(fpm)
        assert model.num_observations == 6
        fpm = _make_fpm(sum_prefill_tokens=15, num_prefill_requests=1, wall_time=0.015)
        model.add_observation(fpm)
        assert model.num_observations == 6

    def test_uniform_distribution_preserved(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=10, min_observations=3, bucket_count=16
        )
        for _ in range(15):
            fpm = _make_fpm(
                num_decode_requests=32, sum_decode_kv_tokens=32000, wall_time=0.01
            )
            model.add_observation(fpm)
        assert model.num_observations == 10
        fpm = _make_fpm(
            num_decode_requests=4, sum_decode_kv_tokens=4000, wall_time=0.005
        )
        model.add_observation(fpm)
        assert model.num_observations == 10

    def test_2d_bucketed_retirement(self):
        model = AggRegressionModel(
            max_num_fpm_samples=8, min_observations=1, bucket_count=16
        )
        for p, d in [(100, 500), (200, 1000), (300, 1500), (400, 2000)]:
            fpm = _make_fpm(
                sum_prefill_tokens=p,
                num_prefill_requests=1,
                sum_decode_kv_tokens=d,
                num_decode_requests=5,
                wall_time=0.001 * p + 0.0001 * d,
            )
            model.add_observation(fpm)
        for _ in range(4):
            fpm = _make_fpm(
                sum_prefill_tokens=100,
                num_prefill_requests=1,
                sum_decode_kv_tokens=500,
                num_decode_requests=5,
                wall_time=0.15,
            )
            model.add_observation(fpm)
        assert model.num_observations == 8
        fpm = _make_fpm(
            sum_prefill_tokens=350,
            num_prefill_requests=1,
            sum_decode_kv_tokens=1800,
            num_decode_requests=5,
            wall_time=0.5,
        )
        model.add_observation(fpm)
        assert model.num_observations == 8


# ── DecodeRegressionModel tests ──────────────────────────────────────


class TestDecodeRegressionModel:
    def _train_2d(self, model):
        for n_req, kv in [(5, 1000), (10, 2000), (15, 3000), (20, 4000), (25, 5000)]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=kv,
                num_decode_requests=n_req,
                wall_time=0.0001 * kv + 0.0005 * n_req + 0.001,
            )
            model.add_observation(fpm)

    def test_insufficient_data(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        assert not model.has_sufficient_data()
        assert model.estimate_next_itl(0, 0) is None

    def test_heartbeat_skipped(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpm = _make_fpm(wall_time=0.0, sum_decode_kv_tokens=100, num_decode_requests=1)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_basic_itl_estimate(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_2d(model)
        assert model.has_sufficient_data()
        est = model.estimate_next_itl(scheduled_decode_kv=3000, queued_decode_kv=0)
        assert est is not None and est > 0

    def test_avg_decode_length_tracking(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        for total_kv, num_req in [(1000, 10), (2000, 10), (3000, 10)]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=total_kv,
                num_decode_requests=num_req,
                wall_time=0.01,
            )
            model.add_observation(fpm)
        assert abs(model.avg_decode_length - 200.0) < 1.0

    def _train_thpt_model(self, model):
        for n_req, kv in [
            (5, 5000),
            (10, 10000),
            (20, 20000),
            (30, 30000),
            (40, 40000),
        ]:
            fpm = _make_fpm(
                sum_decode_kv_tokens=kv,
                num_decode_requests=n_req,
                wall_time=0.00001 * kv + 0.001,
            )
            model.add_observation(fpm)

    def test_find_best_engine_decode_rps(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_thpt_model(model)
        rps, actual_itl = model.find_best_engine_decode_rps(
            itl=50.0, context_length=1000.0, osl=150.0
        )
        assert rps > 0 and actual_itl > 0 and actual_itl <= 50.0

    def test_find_best_engine_decode_rps_discounts_itl(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_thpt_model(model)
        base_rps, base_itl = model.find_best_engine_decode_rps(
            itl=50.0,
            context_length=1000.0,
            osl=150.0,
            accept_length=1.0,
            max_num_seqs=1,
        )
        spec_rps, spec_itl = model.find_best_engine_decode_rps(
            itl=50.0,
            context_length=1000.0,
            osl=150.0,
            accept_length=2.0,
            max_num_seqs=1,
        )

        assert spec_itl == pytest.approx(base_itl / 2)
        assert spec_rps == pytest.approx(base_rps * 2)

    def test_find_best_engine_decode_rps_zero_context(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_2d(model)
        rps, itl_ms = model.find_best_engine_decode_rps(
            itl=50.0, context_length=0.0, osl=150.0
        )
        assert rps == 0.0 and itl_ms == 0.0

    def test_load_benchmark_fpms(self):
        model = DecodeRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpms = [
            _make_fpm(
                num_decode_requests=n,
                sum_decode_kv_tokens=n * 1000,
                wall_time=0.001 * n,
            )
            for n in [5, 10, 15, 20, 25]
        ]
        model.load_benchmark_fpms(fpms)
        assert model.num_observations == 5 and model.has_sufficient_data()


# ── AggRegressionModel tests ─────────────────────────────────────────


class TestAggRegressionModel:
    def _train_agg(self, model):
        for p, d in [(100, 1000), (200, 2000), (300, 3000), (400, 4000), (500, 5000)]:
            fpm = _make_fpm(
                sum_prefill_tokens=p,
                num_prefill_requests=1,
                sum_decode_kv_tokens=d,
                num_decode_requests=10,
                wall_time=0.001 * p + 0.0001 * d + 0.001,
            )
            model.add_observation(fpm)

    def test_insufficient_data(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        assert not model.has_sufficient_data()
        assert model.estimate_next_ttft(0, 2048, 0) is None
        assert model.estimate_next_itl(0, 0) is None

    def test_heartbeat_skipped(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        fpm = _make_fpm(wall_time=0.0, sum_prefill_tokens=100, sum_decode_kv_tokens=200)
        model.add_observation(fpm)
        assert model.num_observations == 0

    def test_2d_regression(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        assert model.has_sufficient_data()
        ttft = model.estimate_next_ttft(
            queued_prefill_tokens=0, max_num_batched_tokens=2048, current_decode_kv=3000
        )
        assert ttft is not None and ttft > 0
        itl = model.estimate_next_itl(scheduled_decode_kv=3000, queued_decode_kv=0)
        assert itl is not None and itl > 0

    def test_find_best_engine_agg_rps(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        thpt, actual_ttft, actual_itl = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
        )
        assert thpt > 0 and actual_ttft >= 0 and actual_itl >= 0

    def test_find_best_engine_agg_rps_discounts_itl(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        base_rps, _, base_itl = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
            accept_length=1.0,
        )
        spec_rps, _, spec_itl = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
            accept_length=2.0,
        )

        assert spec_itl < base_itl
        assert spec_rps > base_rps

    def test_find_best_engine_agg_rps_caps_spec_decode_by_prefill_admission(
        self, monkeypatch
    ):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        monkeypatch.setattr(model, "_ensure_fitted", lambda: True)
        monkeypatch.setattr(model, "_predict_2d", lambda _prefill, _decode: 0.05)
        monkeypatch.setattr(
            model,
            "estimate_next_ttft",
            lambda **_kwargs: 0.0,
        )

        rps, _ttft, itl = model.find_best_engine_agg_rps(
            isl=100.0,
            osl=100.0,
            max_num_batched_tokens=1000,
            ttft_sla=10_000.0,
            itl_sla=10_000.0,
            accept_length=2.0,
        )

        # Without the stricter agg cap, the old raw balance could choose
        # batch=500 and report 200 rps. Spec decode doubles decode egress, so
        # steady-state prefill admission requires a smaller sustainable batch.
        assert rps == pytest.approx(133.2)
        assert itl == pytest.approx(25.0)

    def test_find_best_engine_agg_rps_insufficient_data(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=5, bucket_count=16
        )
        thpt, _, _ = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
        )
        assert thpt == 0.0

    def test_agg_kv_hit_rate_none_equals_zero(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        none_est = model.estimate_next_ttft(
            queued_prefill_tokens=3000,
            max_num_batched_tokens=2048,
            current_decode_kv=1000,
            kv_hit_rate=None,
        )
        zero_est = model.estimate_next_ttft(
            queued_prefill_tokens=3000,
            max_num_batched_tokens=2048,
            current_decode_kv=1000,
            kv_hit_rate=0.0,
        )
        assert none_est == zero_est

    def test_agg_kv_hit_rate_discounts_prefill(self):
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        est_full = model.estimate_next_ttft(
            queued_prefill_tokens=3000,
            max_num_batched_tokens=100_000,
            current_decode_kv=1000,
            kv_hit_rate=0.0,
        )
        est_half = model.estimate_next_ttft(
            queued_prefill_tokens=3000,
            max_num_batched_tokens=100_000,
            current_decode_kv=1000,
            kv_hit_rate=0.5,
        )
        assert est_full is not None and est_half is not None
        assert est_half < est_full

    def test_agg_find_best_engine_rps_hit_rate_increases_throughput(self):
        """find_best_engine_agg_rps should discount only prefill work,
        leaving decode KV at full context; higher hit rate should yield
        greater-or-equal engine rps."""
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        rps_base, _, _ = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
            kv_hit_rate=0.0,
        )
        rps_hit, _, _ = model.find_best_engine_agg_rps(
            isl=2048.0,
            osl=150.0,
            max_num_batched_tokens=4096,
            ttft_sla=500.0,
            itl_sla=50.0,
            kv_hit_rate=0.6,
        )
        assert rps_hit >= rps_base

    def test_agg_find_best_engine_rps_uniform_discount_in_ttft_estimate(self):
        """``find_best_engine_agg_rps`` must apply the kv_hit_rate discount
        uniformly to BOTH the per-iter prefill and the avg_isl portion of
        the TTFT simulation. Regression for the bug where the function
        passed already-discounted prefill_per_iter to estimate_next_ttft
        without forwarding kv_hit_rate, leaving avg_isl at full size and
        inflating the predicted TTFT (= over-provisioning replicas)."""
        model = AggRegressionModel(
            max_num_fpm_samples=50, min_observations=3, bucket_count=16
        )
        self._train_agg(model)
        # With a permissive ITL/TTFT SLA, the only difference in engine_rps
        # at high hit rate vs zero hit rate should come from the prefill
        # discount. If the bug recurs the high-hit-rate path will under-
        # estimate capacity (smaller batch sweep) and produce strictly less
        # rps growth than the discount factor warrants.
        rps_zero, _, _ = model.find_best_engine_agg_rps(
            isl=4000.0,
            osl=200.0,
            max_num_batched_tokens=8192,
            ttft_sla=10_000.0,
            itl_sla=10_000.0,
            kv_hit_rate=0.0,
        )
        rps_high, _, _ = model.find_best_engine_agg_rps(
            isl=4000.0,
            osl=200.0,
            max_num_batched_tokens=8192,
            ttft_sla=10_000.0,
            itl_sla=10_000.0,
            kv_hit_rate=0.8,
        )
        # Strictly greater capacity at 80% hit rate (not just >=).
        assert rps_high > rps_zero


# ── Decode/prefill regime separation (DEEPINFRA) ─────────────────────
#
# Shapes below mirror measured DeepSeek-V4-Flash traffic (1xB300, dspark
# nextn=7, accept_length~3.14): pure-decode iterations land near 0.060s
# while iterations carrying a prefill chunk land near 0.228s, and resident
# KV per decode slot is ~19.4k rather than the arrival-weighted isl+osl/2.


class TestAggRegimeSeparation:
    ISL = 5461.0
    OSL = 320.0
    KV_PER_SLOT = 19_426
    # Generative model fitted to the measured samples: a small positive KV
    # slope plus a prefill term an order of magnitude more expensive per
    # token. At 72 slots this yields 0.060s pure / 0.230s with an 8k prefill
    # chunk, matching the observed medians. Coefficients must stay positive
    # or _BaseRegressionModel._fit rejects the fit outright.
    BASE_WT = 0.040
    KV_COEF = 1.4e-8
    PREFILL_COEF = 2.08e-5

    def _wt(self, prefill_tok, kv):
        return self.BASE_WT + self.KV_COEF * kv + self.PREFILL_COEF * prefill_tok

    @property
    def pure_decode_wt(self):
        return self._wt(0, 72 * self.KV_PER_SLOT)

    @property
    def mixed_wt(self):
        return self._wt(8192, 72 * self.KV_PER_SLOT)

    def _train_bimodal(self, model, *, pure=True):
        """Feed the two regimes across the batch-size range.

        Enough distinct KV values to fill ``_DECODE_FIT_BINS`` bins with at
        least 3 samples each -- below that the model correctly declines to
        fit a trend and falls back to a level.

        ``pure=False`` trains only the prefill-bearing regime, which is what
        the pre-existing fixtures do and exercises the fallback path.
        """
        prefill_choices = (4096, 6144, 8192, 10240, 12288)
        for i, nreq in enumerate(range(40, 73, 2)):
            kv = nreq * self.KV_PER_SLOT
            prefill_tok = prefill_choices[i % len(prefill_choices)]
            for rep in range(3):
                # Slight jitter so the per-bin median is doing real work
                # rather than reading off a single exact value.
                jitter = 1.0 + 0.01 * (rep - 1)
                if pure:
                    model.add_observation(
                        _make_fpm(
                            sum_prefill_tokens=0,
                            num_prefill_requests=0,
                            sum_decode_kv_tokens=kv,
                            num_decode_requests=nreq,
                            wall_time=self._wt(0, kv) * jitter,
                        )
                    )
                model.add_observation(
                    _make_fpm(
                        sum_prefill_tokens=prefill_tok,
                        num_prefill_requests=2,
                        sum_decode_kv_tokens=kv,
                        num_decode_requests=nreq,
                        wall_time=self._wt(prefill_tok, kv) * jitter,
                    )
                )

    def test_itl_tracks_decode_regime_not_the_mixture(self):
        """ITL must reflect a pure decode step, not the bimodal mean.

        The mixed fit sees both regimes and lands between them; the decode
        regime is the one that actually governs inter-token latency.
        """
        model = AggRegressionModel(
            max_num_fpm_samples=200, min_observations=3, bucket_count=16
        )
        self._train_bimodal(model)

        itl = model.estimate_next_itl(
            scheduled_decode_kv=72 * self.KV_PER_SLOT, queued_decode_kv=0
        )
        assert itl is not None
        # Within 25% of the pure-decode step, and nowhere near the 0.228s
        # prefill-bearing regime or the ~0.14s mixture mean.
        assert itl == pytest.approx(self.pure_decode_wt, rel=0.25)
        assert itl < 0.5 * self.mixed_wt

    def test_itl_falls_back_to_mixed_fit_before_regime_ready(self):
        """No pure-decode samples yet -> previous behaviour, not None."""
        model = AggRegressionModel(
            max_num_fpm_samples=200, min_observations=3, bucket_count=16
        )
        self._train_bimodal(model, pure=False)

        itl = model.estimate_next_itl(
            scheduled_decode_kv=72 * self.KV_PER_SLOT, queued_decode_kv=0
        )
        assert itl is not None and itl > 0

    def test_capacity_itl_is_physical(self):
        """Regression guard for the sub-millisecond-ITL failure.

        Previously the sweep derived ITL from the prefill-dominated mixed fit
        probed at ``bs * (isl + osl/2)``, far below the observed KV range;
        that reported itl_ms=0.239 and ~196 rps on a single B300 against ~6
        rps of real per-engine demand. ITL must stay in a physical band and
        capacity must stay within an order of magnitude of reality.
        """
        model = AggRegressionModel(
            max_num_fpm_samples=200, min_observations=3, bucket_count=16
        )
        self._train_bimodal(model)

        rps, ttft_ms, itl_ms = model.find_best_engine_agg_rps(
            isl=self.ISL,
            osl=self.OSL,
            max_num_batched_tokens=16384,
            ttft_sla=574.0,
            itl_sla=46.0,
            max_kv_tokens=12_700_000,
            max_num_seqs=72,
            kv_hit_rate=0.165,
            accept_length=3.14,
        )

        # A forward pass on this engine cannot be sub-millisecond.
        assert itl_ms > 1.0, f"unphysical ITL {itl_ms}ms"
        # Expected ~19ms: 0.060s pure-decode step / 3.14 accept length.
        assert itl_ms == pytest.approx(
            1000.0 * self.pure_decode_wt / 3.14, rel=0.35
        )
        assert 0.0 < rps < 100.0, f"implausible engine capacity {rps} rps"
        assert ttft_ms >= 0.0

    def test_itl_rises_with_resident_kv(self):
        """The KV trend is small but real and must not be discarded.

        Bin medians on live data give slope ~2.4e-9 s/token (r=+0.720),
        ~12% of the step across the observed range. A flat level would
        leave ITL blind to decode load.
        """
        model = AggRegressionModel(
            max_num_fpm_samples=200, min_observations=3, bucket_count=16
        )
        self._train_bimodal(model)

        lo = model.estimate_next_itl(
            scheduled_decode_kv=40 * self.KV_PER_SLOT, queued_decode_kv=0
        )
        hi = model.estimate_next_itl(
            scheduled_decode_kv=72 * self.KV_PER_SLOT, queued_decode_kv=0
        )
        assert lo is not None and hi is not None
        assert hi > lo, f"ITL is flat in resident KV ({lo} -> {hi})"
        # Queued work must also register as pressure.
        queued = model.estimate_next_itl(
            scheduled_decode_kv=40 * self.KV_PER_SLOT,
            queued_decode_kv=20 * self.KV_PER_SLOT,
        )
        assert queued > lo

    def test_itl_never_extrapolates_past_measured_range(self):
        """Clamp guard: both historical failures were extrapolations.

        The mixed fit extrapolated down to a 0.75ms forward pass; raw least
        squares on the decode subset extrapolated up to 288ms. Neither may
        be reachable regardless of the KV asked about.
        """
        model = AggRegressionModel(
            max_num_fpm_samples=200, min_observations=3, bucket_count=16
        )
        self._train_bimodal(model)

        at_max = model.estimate_next_itl(
            scheduled_decode_kv=72 * self.KV_PER_SLOT, queued_decode_kv=0
        )
        for absurd_kv in (50_000_000, 500_000_000):
            est = model.estimate_next_itl(
                scheduled_decode_kv=absurd_kv, queued_decode_kv=0
            )
            assert est == pytest.approx(at_max, rel=1e-6), (
                f"extrapolated to {est}s at kv={absurd_kv}"
            )
        # And nothing collapses toward zero at implausibly low KV.
        tiny = model.estimate_next_itl(scheduled_decode_kv=1, queued_decode_kv=0)
        assert tiny > 0.5 * self._wt(0, 40 * self.KV_PER_SLOT)

    def test_avg_ctx_uses_measured_residency(self):
        """KV per slot comes from observations, not isl + osl/2.

        The arrival-weighted proxy understated residency 3.5x on skewed
        traffic, so the KV-capacity cap was correspondingly overstated.
        """
        model = AggRegressionModel(
            max_num_fpm_samples=200, min_observations=3, bucket_count=16
        )
        self._train_bimodal(model)
        assert model.avg_decode_length == pytest.approx(self.KV_PER_SLOT, rel=0.01)
        # Measured residency exceeds the arrival-weighted proxy it replaces.
        assert model.avg_decode_length > self.ISL + self.OSL / 2.0


# ── Connector-driven refresh tests ──────────────────────────────────


class TestRefreshWorkerInfoFromConnector:
    """Tests for NativePlannerBase._refresh_worker_info_from_connector.

    The tick-loop refresh delegates to the connector's ``get_worker_info``,
    which is where each connector implements its own MDC source (K8s CRDs
    for KubernetesConnector, discovery watch for VirtualConnector).  These
    tests exercise the shared refresh plumbing with a mock connector.
    """

    def _make_planner(self, require_prefill=False, require_decode=True):
        """Build a minimal NativePlannerBase with no_operation=True."""
        # Bypass Prometheus registration (Gauge+Enum double-register across
        # tests). KubernetesConnector.__init__ loads ~/.kube/config and reads
        # DYN_PARENT_DGD_K8S_NAME; stub both so this runs in plain pytest envs.
        with (
            patch("dynamo.planner.core.base.PlannerPrometheusMetrics") as mock_metrics,
            patch("dynamo.planner.connectors.kubernetes.KubernetesAPI"),
            patch.dict(os.environ, {"DYN_PARENT_DGD_K8S_NAME": "test-graph"}),
        ):
            mock_metrics.return_value = Mock()
            config = PlannerConfig.model_construct(
                throughput_adjustment_interval_seconds=60,
                prefill_engine_num_gpu=1,
                decode_engine_num_gpu=1,
                min_endpoint=1,
                max_gpu_budget=-1,
                ttft_ms=500.0,
                itl_ms=50.0,
                backend="vllm",
                no_operation=True,
                metric_pulling_prometheus_endpoint="http://localhost:9090",
                metric_reporting_prometheus_port=0,
                load_predictor="constant",
                environment="kubernetes",
                namespace="test-namespace",
                mode="agg",
                enable_load_scaling=True,
                enable_throughput_scaling=True,
                load_adjustment_interval_seconds=5,
                max_num_fpm_samples=50,
                fpm_sample_bucket_size=16,
                load_scaling_down_sensitivity=80,
                load_min_observations=5,
            )
            planner = NativePlannerBase(None, config)
        planner.require_prefill = require_prefill
        planner.require_decode = require_decode
        planner.prefill_worker_info = WorkerInfo()
        planner.decode_worker_info = WorkerInfo()
        return planner

    def _install_mock_connector(self, planner, **fresh_info_kwargs):
        """Replace planner.connector with a Mock returning a fresh WorkerInfo."""
        fresh = WorkerInfo(**fresh_info_kwargs)
        mock_connector = Mock()
        mock_connector.get_worker_info.return_value = fresh
        planner.connector = mock_connector
        return mock_connector

    def test_refresh_populates_missing_fields(self):
        """Connector returns a populated WorkerInfo; missing fields backfill."""
        planner = self._make_planner()
        assert planner.decode_worker_info.max_num_batched_tokens is None

        self._install_mock_connector(
            planner,
            max_num_batched_tokens=8192,
            total_kv_blocks=1024,
            max_num_seqs=256,
            kv_cache_block_size=16,
            context_length=4096,
        )

        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens == 8192
        assert planner.decode_worker_info.total_kv_blocks == 1024
        assert planner.decode_worker_info.max_num_seqs == 256
        assert planner.decode_worker_info.kv_cache_block_size == 16
        assert planner.decode_worker_info.context_length == 4096

    def test_noop_when_already_set(self):
        """Does not re-query once max_num_batched_tokens is populated."""
        planner = self._make_planner()
        planner.decode_worker_info = WorkerInfo(max_num_batched_tokens=2048)

        mock_connector = self._install_mock_connector(
            planner, max_num_batched_tokens=8192
        )

        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens == 2048
        mock_connector.get_worker_info.assert_not_called()

    def test_noop_when_connector_lacks_get_worker_info(self):
        """Silently does nothing if the connector does not implement get_worker_info."""
        planner = self._make_planner()

        class _StubConnector:
            pass

        planner.connector = _StubConnector()
        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens is None

    def test_noop_when_connector_returns_none_fields(self):
        """Fresh WorkerInfo with None everywhere does not overwrite anything."""
        planner = self._make_planner()
        self._install_mock_connector(planner)  # All Nones
        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens is None

    def test_exception_does_not_propagate(self):
        """If connector.get_worker_info throws, refresh is a no-op."""
        planner = self._make_planner()
        mock_connector = Mock()
        mock_connector.get_worker_info.side_effect = RuntimeError("boom")
        planner.connector = mock_connector

        planner._refresh_worker_info_from_connector()  # must not raise
        assert planner.decode_worker_info.max_num_batched_tokens is None

    def test_updates_engine_capabilities(self):
        """Builtin engine capabilities are refreshed after worker-info discovery."""
        planner = self._make_planner()
        engine = planner._ensure_engine()

        self._install_mock_connector(planner, max_num_batched_tokens=4096)

        planner._refresh_worker_info_from_connector()
        assert planner.decode_worker_info.max_num_batched_tokens == 4096
        assert engine._scaling_state._capabilities.decode.max_num_batched_tokens == 4096

    def test_refresh_skips_unneeded_sub_component(self):
        """Only sub-components with require_* True are refreshed."""
        planner = self._make_planner(require_prefill=False, require_decode=True)

        def _side_effect(sub_type, backend):
            # Should only be called for DECODE.
            assert sub_type.value == "decode"
            return WorkerInfo(max_num_batched_tokens=4096)

        mock_connector = Mock()
        mock_connector.get_worker_info.side_effect = _side_effect
        planner.connector = mock_connector

        planner._refresh_worker_info_from_connector()
        assert planner.prefill_worker_info.max_num_batched_tokens is None
        assert planner.decode_worker_info.max_num_batched_tokens == 4096

    async def test_load_only_accept_length_missing_returns_observation(self):
        planner = self._make_planner(require_prefill=False, require_decode=True)
        planner.model_name = "Qwen/Qwen3"
        planner.decode_worker_info = WorkerInfo(component_name="backend")
        planner.prometheus_traffic_client = Mock()
        planner.prometheus_traffic_client.get_avg_kv_hit_rate.return_value = None
        planner.prometheus_traffic_client.get_avg_spec_decode_accept_length.return_value = (
            None
        )

        obs = await planner._collect_kv_hit_rate_observation(duration_s=5)

        assert obs is not None
        assert obs.kv_hit_rate is None
        assert obs.accept_length is None
