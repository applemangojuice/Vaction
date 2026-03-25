"""Tests for query parsing and search execution."""

from __future__ import annotations

from vaction.search import Op, execute_search, parse_query


class TestQueryParser:
    def test_simple_term(self):
        node = parse_query("car")
        assert node.op == Op.TERM
        assert node.value == "car"

    def test_and_query(self):
        node = parse_query("person AND car")
        assert node.op == Op.AND
        assert len(node.children) == 2
        assert node.children[0].value == "person"
        assert node.children[1].value == "car"

    def test_or_query(self):
        node = parse_query("helly OR irving")
        assert node.op == Op.OR
        assert len(node.children) == 2
        assert node.children[0].value == "helly"
        assert node.children[1].value == "irving"

    def test_not_query(self):
        node = parse_query("NOT car")
        assert node.op == Op.NOT
        assert node.children[0].value == "car"

    def test_complex_query(self):
        node = parse_query("person AND NOT car")
        assert node.op == Op.AND
        assert node.children[0].value == "person"
        assert node.children[1].op == Op.NOT
        assert node.children[1].children[0].value == "car"

    def test_case_insensitive_operators(self):
        node = parse_query("person and car")
        assert node.op == Op.AND

    def test_parentheses(self):
        node = parse_query("(mark OR helly) AND hallway")
        assert node.op == Op.AND
        assert node.children[0].op == Op.OR
        assert node.children[1].value == "hallway"


class TestSearchExecution:
    def test_simple_search(self, populated_db):
        results = execute_search(populated_db, "person", series_id=1)
        assert len(results) > 0
        # Should find person in both episodes
        episode_ids = {r["episode_id"] for r in results}
        assert 1 in episode_ids
        assert 2 in episode_ids

    def test_and_search(self, populated_db):
        results = execute_search(populated_db, "person AND car", series_id=1)
        # person in frames 1-5, car in frames 3-7 -> overlap in frames 3-5
        assert len(results) > 0
        # Should only match episode 1 (episode 2 has no car)
        episode_ids = {r["episode_id"] for r in results}
        assert 1 in episode_ids

    def test_or_search(self, populated_db):
        results = execute_search(populated_db, "hallway OR office", series_id=1)
        assert len(results) > 0
        episode_ids = {r["episode_id"] for r in results}
        assert 1 in episode_ids
        assert 2 in episode_ids

    def test_no_results(self, populated_db):
        results = execute_search(populated_db, "spaceship", series_id=1)
        assert len(results) == 0

    def test_season_filter(self, populated_db):
        results = execute_search(populated_db, "person", series_id=1, season_number=1)
        assert len(results) > 0

    def test_episode_filter(self, populated_db):
        results = execute_search(
            populated_db, "person", series_id=1, episode_number=1
        )
        episode_ids = {r["episode_id"] for r in results}
        assert episode_ids == {1}
