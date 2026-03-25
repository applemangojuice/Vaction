"""Tests for pixel-time metrics computation."""

from __future__ import annotations

from vaction.metrics import compute_episode_metrics, compute_season_summary, format_timestamp


class TestMetrics:
    def test_episode_metrics_basic(self, populated_db):
        results = compute_episode_metrics(populated_db, "person", series_id=1)
        assert len(results) == 2  # Both episodes have "person"

        for r in results:
            assert r.pixel_time > 0
            assert r.episode_share > 0
            assert r.episode_share_pct > 0
            assert r.episode_share_pct <= 100.0

    def test_pixel_time_calculation(self, populated_db):
        results = compute_episode_metrics(
            populated_db, "person", series_id=1, episode_number=1
        )
        assert len(results) == 1

        r = results[0]
        # Episode 1 has person in 5 frames, each with pixel_area=80000, delta_t=0.5
        expected_pixel_time = 5 * 80000 * 0.5
        assert r.pixel_time == expected_pixel_time

    def test_episode_share_calculation(self, populated_db):
        results = compute_episode_metrics(
            populated_db, "person", series_id=1, episode_number=1
        )
        r = results[0]

        # Total budget: 5400 frames * 1920 * 1080 = 11,197,440,000
        total_budget = 5400 * 1920 * 1080
        # Matched: 5 frames * 80000 pixels
        matched_sum = 5 * 80000
        expected_share = matched_sum / total_budget

        assert abs(r.episode_share - expected_share) < 1e-10

    def test_and_query_metrics(self, populated_db):
        results = compute_episode_metrics(
            populated_db, "person AND car", series_id=1
        )
        # Only episode 1 has both
        assert len(results) >= 1
        ep1 = [r for r in results if r.episode_number == 1]
        assert len(ep1) == 1
        assert ep1[0].pixel_time > 0

    def test_season_summary(self, populated_db):
        summary = compute_season_summary(populated_db, "person", series_id=1)
        assert summary["num_matching_episodes"] == 2
        assert summary["total_pixel_time"] > 0
        # Share is very small due to test data (small detection area vs full frame budget)
        assert summary["average_episode_share_pct"] >= 0

    def test_strongest_windows(self, populated_db):
        results = compute_episode_metrics(
            populated_db, "person", series_id=1, episode_number=1
        )
        r = results[0]
        assert len(r.strongest_windows) > 0
        # Windows should be (start, end) tuples
        for start, end in r.strongest_windows:
            assert end >= start

    def test_no_results(self, populated_db):
        results = compute_episode_metrics(populated_db, "spaceship", series_id=1)
        assert len(results) == 0


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00:00"

    def test_minutes(self):
        assert format_timestamp(125) == "00:02:05"

    def test_hours(self):
        assert format_timestamp(3661) == "01:01:01"
