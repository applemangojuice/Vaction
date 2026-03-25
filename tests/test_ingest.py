"""Tests for ingestion and parsing."""

from __future__ import annotations

from pathlib import Path

from vaction.ingest import parse_episode_from_filename


class TestEpisodeParsing:
    def test_standard_format(self):
        p = Path("/shows/Severance/Season 1/S01E03 - In Perpetuity.mkv")
        result = parse_episode_from_filename(p)
        assert result.season == 1
        assert result.episode == 3
        assert "In Perpetuity" in result.title

    def test_lowercase(self):
        p = Path("/shows/s02e05.mp4")
        result = parse_episode_from_filename(p)
        assert result.season == 2
        assert result.episode == 5

    def test_x_format(self):
        p = Path("/shows/1x04 - Title.mkv")
        result = parse_episode_from_filename(p)
        assert result.season == 1
        assert result.episode == 4

    def test_season_word_format(self):
        p = Path("/shows/Season 3/Episode 7 - Something.mp4")
        result = parse_episode_from_filename(p)
        assert result.season == 3
        assert result.episode == 7

    def test_fallback_parsing(self):
        p = Path("/shows/Season 2/05 - Title.mp4")
        result = parse_episode_from_filename(p)
        assert result.season == 2
        assert result.episode == 5
