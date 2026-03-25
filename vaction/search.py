"""Query parsing and search execution over the detection index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vaction.db import get_aliases


class Op(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    TERM = "TERM"


@dataclass
class QueryNode:
    """AST node for a search query."""
    op: Op
    value: str | None = None  # For TERM nodes
    children: list["QueryNode"] | None = None  # For AND/OR/NOT nodes


def parse_query(query_str: str) -> QueryNode:
    """Parse a boolean query string into an AST.

    Supports:
        - Simple terms: "car", "hallway"
        - AND: "mark AND hallway"
        - OR: "helly OR irving"
        - NOT: "vehicle NOT office"
        - Parentheses: "(mark OR helly) AND hallway"
    """
    tokens = _tokenize(query_str)
    node, _ = _parse_or(tokens, 0)
    return node


def _tokenize(query_str: str) -> list[str]:
    """Split query into tokens, preserving AND/OR/NOT and parentheses."""
    # Split on whitespace but keep quoted strings together
    pattern = r'"[^"]*"|\(|\)|AND|OR|NOT|[^\s()"]+'
    tokens = re.findall(pattern, query_str, re.IGNORECASE)
    return tokens


def _parse_or(tokens: list[str], pos: int) -> tuple[QueryNode, int]:
    """Parse OR expressions (lowest precedence)."""
    left, pos = _parse_and(tokens, pos)

    while pos < len(tokens) and tokens[pos].upper() == "OR":
        pos += 1  # skip OR
        right, pos = _parse_and(tokens, pos)
        left = QueryNode(op=Op.OR, children=[left, right])

    return left, pos


def _parse_and(tokens: list[str], pos: int) -> tuple[QueryNode, int]:
    """Parse AND expressions."""
    left, pos = _parse_not(tokens, pos)

    while pos < len(tokens) and tokens[pos].upper() == "AND":
        pos += 1  # skip AND
        right, pos = _parse_not(tokens, pos)
        left = QueryNode(op=Op.AND, children=[left, right])

    return left, pos


def _parse_not(tokens: list[str], pos: int) -> tuple[QueryNode, int]:
    """Parse NOT expressions."""
    if pos < len(tokens) and tokens[pos].upper() == "NOT":
        pos += 1  # skip NOT
        child, pos = _parse_primary(tokens, pos)
        return QueryNode(op=Op.NOT, children=[child]), pos

    return _parse_primary(tokens, pos)


def _parse_primary(tokens: list[str], pos: int) -> tuple[QueryNode, int]:
    """Parse primary expressions: terms or parenthesized expressions."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of query")

    token = tokens[pos]

    if token == "(":
        pos += 1  # skip (
        node, pos = _parse_or(tokens, pos)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1  # skip )
        return node, pos

    # It's a term (strip quotes if present)
    term = token.strip('"').lower()
    return QueryNode(op=Op.TERM, value=term), pos + 1


def build_search_sql(
    node: QueryNode,
    conn=None,
    confidence_threshold: float = 0.25,
    series_id: int | None = None,
    season_number: int | None = None,
    episode_number: int | None = None,
) -> tuple[str, list[Any]]:
    """Build SQL query from an AST node.

    Returns (sql_string, params) for frame-level matching.
    The query returns frame IDs and their matched pixel areas.
    """
    where_clause, params = _node_to_sql(node, conn, confidence_threshold)

    # Base query: get matched frames with pixel-time data
    sql = """
    SELECT
        f.id AS frame_id,
        f.episode_id,
        f.timestamp,
        f.delta_t,
        COALESCE(matched.total_pixel_area, 0) AS matched_pixel_area,
        e.width,
        e.height,
        e.num_frames,
        e.duration_secs,
        e.number AS episode_number,
        s.number AS season_number,
        sr.name AS series_name,
        e.title AS episode_title
    FROM frame f
    JOIN episode e ON f.episode_id = e.id
    JOIN season s ON e.season_id = s.id
    JOIN series sr ON s.series_id = sr.id
    LEFT JOIN (
        SELECT frame_id, SUM(pixel_area) AS total_pixel_area
        FROM detection
        WHERE {where_clause}
        GROUP BY frame_id
    ) matched ON matched.frame_id = f.id
    WHERE matched.total_pixel_area > 0
    """.format(where_clause=where_clause)

    filter_params = []
    if series_id is not None:
        sql += " AND sr.id = ?"
        filter_params.append(series_id)
    if season_number is not None:
        sql += " AND s.number = ?"
        filter_params.append(season_number)
    if episode_number is not None:
        sql += " AND e.number = ?"
        filter_params.append(episode_number)

    sql += " ORDER BY s.number, e.number, f.timestamp"

    return sql, params + filter_params


def _node_to_sql(
    node: QueryNode, conn=None, confidence_threshold: float = 0.25
) -> tuple[str, list[Any]]:
    """Convert a query AST node to a SQL WHERE clause fragment."""
    if node.op == Op.TERM:
        labels = [node.value]
        # Expand aliases
        if conn is not None:
            aliases = get_aliases(conn, node.value)
            if aliases:
                labels = aliases

        if len(labels) == 1:
            return "label = ? AND confidence >= ?", [labels[0], confidence_threshold]
        else:
            placeholders = ", ".join("?" * len(labels))
            return (
                f"label IN ({placeholders}) AND confidence >= ?",
                labels + [confidence_threshold],
            )

    elif node.op == Op.AND:
        # Both conditions must be true in the same frame
        left_sql, left_params = _node_to_sql(node.children[0], conn, confidence_threshold)
        right_sql, right_params = _node_to_sql(node.children[1], conn, confidence_threshold)
        return (
            f"frame_id IN (SELECT frame_id FROM detection WHERE {left_sql}) "
            f"AND frame_id IN (SELECT frame_id FROM detection WHERE {right_sql})",
            left_params + right_params,
        )

    elif node.op == Op.OR:
        left_sql, left_params = _node_to_sql(node.children[0], conn, confidence_threshold)
        right_sql, right_params = _node_to_sql(node.children[1], conn, confidence_threshold)
        return f"(({left_sql}) OR ({right_sql}))", left_params + right_params

    elif node.op == Op.NOT:
        child_sql, child_params = _node_to_sql(node.children[0], conn, confidence_threshold)
        # NOT: exclude frames where the child condition is true
        # This needs special handling - we match all frames NOT containing the term
        # In practice, NOT is used with AND: "person AND NOT car"
        return (
            f"frame_id NOT IN (SELECT frame_id FROM detection WHERE {child_sql})",
            child_params,
        )

    raise ValueError(f"Unknown operation: {node.op}")


def execute_search(
    conn,
    query_str: str,
    confidence_threshold: float = 0.25,
    series_id: int | None = None,
    season_number: int | None = None,
    episode_number: int | None = None,
) -> list[dict]:
    """Execute a search query and return raw frame-level results."""
    ast = parse_query(query_str)
    sql, params = build_search_sql(
        ast, conn, confidence_threshold,
        series_id=series_id,
        season_number=season_number,
        episode_number=episode_number,
    )
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]
