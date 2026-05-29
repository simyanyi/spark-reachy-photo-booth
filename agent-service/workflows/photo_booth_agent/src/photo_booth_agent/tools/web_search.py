# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.common import TypedBaseModel
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from photo_booth_agent.constant import SERVICE_NAME

logger = logging.getLogger(SERVICE_NAME)


class WebSearchToolConfig(FunctionBaseConfig, name="web_search"):
    timeout_seconds: float = Field(
        default=8.0,
        ge=1.0,
        le=30.0,
        description="Request timeout in seconds.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search snippets to return.",
    )


class WebSearchInput(TypedBaseModel, name="web_search"):
    query: str = Field(
        description="The web search query.",
        min_length=2,
        max_length=300,
    )


def _flatten_related_topics(
    related_topics: list[dict[str, Any]],
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for item in related_topics:
        if "Topics" in item and isinstance(item["Topics"], list):
            nested = item["Topics"]
            for nested_item in nested:
                text = nested_item.get("Text")
                url = nested_item.get("FirstURL")
                if isinstance(text, str) and isinstance(url, str):
                    results.append({"text": text, "url": url})
        else:
            text = item.get("Text")
            url = item.get("FirstURL")
            if isinstance(text, str) and isinstance(url, str):
                results.append({"text": text, "url": url})
    return results


def _search_duckduckgo(query: str, timeout_seconds: float, max_results: int) -> str:
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    url = f"https://api.duckduckgo.com/?{urlencode(params)}"
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "reachy-photo-booth-web-search/1.0",
        },
    )

    with urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    abstract = payload.get("AbstractText") or ""
    answer = payload.get("Answer") or ""
    heading = payload.get("Heading") or ""
    abstract_url = payload.get("AbstractURL") or ""

    snippets: list[str] = []
    if isinstance(answer, str) and answer.strip():
        snippets.append(f"Answer: {answer.strip()}")
    if isinstance(abstract, str) and abstract.strip():
        if isinstance(heading, str) and heading.strip():
            snippets.append(f"{heading.strip()}: {abstract.strip()}")
        else:
            snippets.append(abstract.strip())
    if isinstance(abstract_url, str) and abstract_url.strip():
        snippets.append(f"Source: {abstract_url.strip()}")

    related = payload.get("RelatedTopics")
    if isinstance(related, list):
        related_items = _flatten_related_topics(related)
        for item in related_items[:max_results]:
            snippets.append(f"- {item['text']} ({item['url']})")

    if not snippets:
        return (
            "No direct web results found for this query. "
            "Try refining the query with more specific terms."
        )

    return "\n".join(snippets)


@register_function(config_type=WebSearchToolConfig)
async def web_search_tool(config: WebSearchToolConfig, _: Builder):
    async def _inner(input: WebSearchInput) -> str:
        query = input.query.strip()
        logger.info("web_search called with query=%r", query)
        try:
            return await asyncio.to_thread(
                _search_duckduckgo,
                query,
                config.timeout_seconds,
                config.max_results,
            )
        except Exception as ex:
            logger.exception("web_search failed: %s", ex)
            return (
                "Web search failed due to a network or parsing error. "
                "Please answer using existing knowledge and mention uncertainty."
            )

    yield FunctionInfo.create(
        single_fn=_inner,
        input_schema=WebSearchInput,
        description=(
            "Search the web for up-to-date or factual information and return concise "
            "snippets with source URLs."
        ),
    )
