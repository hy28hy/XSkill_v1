"""
WebSearch tool - use DuckDuckGo (via `ddgs` Python library) for web search.
No API key required. Free, unlimited (subject to DDG rate limiting).

Compatible drop-in replacement for the original Serper.dev-based implementation:
- Same class name `WebSearch` and registered tool name `web_search`
- Same call signature and return format (markdown numbered list)
"""

import os
import json
import time
from tools.base import BaseTool
from tools.tool_registry import register_tool


@register_tool("web_search")
class WebSearch(BaseTool):
    name = "web_search"
    description = (
        "Search the web for information online. Use when you need to find information, "
        "facts, or current events. Returns web search results with titles, URLs, and "
        "text snippets. You only have limited search times, so please use it wisely."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10)",
                "default": 10
            }
        },
        "required": ["query"]
    }

    def __init__(self, config=None):
        super().__init__(config)

        # Lazy import so unrelated environments without `ddgs` don't break.
        # `ddgs` is the new package name; `duckduckgo_search` is the legacy name.
        self._ddgs_cls = None
        try:
            from ddgs import DDGS  # newer package
            self._ddgs_cls = DDGS
        except Exception:
            try:
                from duckduckgo_search import DDGS  # legacy fallback
                self._ddgs_cls = DDGS
            except Exception as e:
                raise ImportError(
                    "Neither `ddgs` nor `duckduckgo_search` is installed. "
                    "Run: pip install ddgs"
                ) from e

        self.max_results_default = config.get('max_results', 10) if config else 10
        self.timeout = config.get('timeout', 30) if config else 30
        # `region` and `safesearch` are DDG-specific knobs, expose via config.
        self.region = (config.get('region', 'wt-wt') if config else 'wt-wt')
        self.safesearch = (config.get('safesearch', 'off') if config else 'off')

        # Optional proxy (DDGS supports `proxy=` arg)
        self.proxy = (
            os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        )
        if self.proxy:
            print(f"[WebSearch] Using proxy: {self.proxy}")

    def call(self, params, **kwargs):
        """
        Execute web search via DuckDuckGo.

        Args:
            params: dict with `query` and optional `max_results`, or a raw query string.

        Returns:
            Markdown numbered list of search results, identical format to Serper backend.
        """
        # Parse parameters
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {"query": params}

        query = params.get("query", "")
        max_results = int(params.get("max_results", self.max_results_default) or self.max_results_default)
        if max_results <= 0:
            max_results = self.max_results_default

        if not query:
            return "Error: No search query provided"

        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                print(f"[WebSearch] Searching for: {query} (attempt {attempt + 1}/{max_retries})")

                ddgs_kwargs = {}
                if self.proxy:
                    ddgs_kwargs["proxy"] = self.proxy
                if self.timeout:
                    ddgs_kwargs["timeout"] = int(self.timeout)

                with self._ddgs_cls(**ddgs_kwargs) as ddgs:
                    raw_results = list(
                        ddgs.text(
                            query,
                            region=self.region,
                            safesearch=self.safesearch,
                            max_results=max_results,
                        )
                    )

                if not raw_results:
                    return f"No results found for query: '{query}'"

                # DDGS schema: dict with keys `title`, `href`, `body`
                # (some versions also use `link` instead of `href`).
                formatted_results = []
                for i, r in enumerate(raw_results[:max_results], 1):
                    title = r.get("title") or "No title"
                    link = r.get("href") or r.get("link") or r.get("url") or "No URL"
                    snippet = r.get("body") or r.get("snippet") or r.get("description") or "No description"
                    formatted_results.append(
                        f"{i}. [{title}]({link})\n"
                        f"   {snippet}"
                    )

                output = f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
                print(f"[WebSearch] Found {len(raw_results)} results")
                return output

            except Exception as e:
                err = str(e)
                if attempt < max_retries - 1:
                    print(f"[WebSearch] Search failed: {err}; retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                error_msg = f"Unexpected error during DuckDuckGo search: {err}"
                print(f"[WebSearch] {error_msg}")
                import traceback
                traceback.print_exc()
                return f"Error: {error_msg}"

        return "Error: All retry attempts failed"
