"""Bounded public web search and page retrieval for model function calls."""

from __future__ import annotations

import base64
import hashlib
import html
import ipaddress
import json
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit

import requests

SEARCH_TIMEOUT = (3, 8)
FETCH_TIMEOUT = (5, 12)
MAX_SEARCH_BYTES = 1_000_000
MAX_FETCH_BYTES = 2_000_000
MAX_PAGE_CHARS = 16_000
MAX_RESULTS = 5
PAGE_CACHE_SIZE = 32
PAGE_CACHE_TTL_SECONDS = 300
USER_AGENT = "gaia-course-agent/0.1 (+public research)"
SEARCH_STOPWORDS = {
    "about", "from", "with", "what", "where", "when", "which", "whose", "their",
    "that", "this", "image", "photo", "picture", "official", "site",
}
_page_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_page_cache_lock = threading.Lock()


class _PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"}:
            self.skip_depth += 1
        elif not self.skip_depth and tag in {"p", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def _html_text(value: str) -> str:
    parser = _PageText()
    parser.feed(value)
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())


def _read_limited(response: requests.Response, limit: int) -> tuple[bytes, bool]:
    chunks = []
    size = 0
    for chunk in response.iter_content(chunk_size=32_768):
        if not chunk:
            continue
        remaining = limit + 1 - size
        chunks.append(chunk[:remaining])
        size += min(len(chunk), remaining)
        if size > limit:
            break
    return b"".join(chunks)[:limit], size > limit


def _public_url(url: str, *, resolve: bool) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP(S) URLs are allowed")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Nonstandard URL ports are not allowed")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local URLs are not allowed")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        addresses = []
    if resolve and not addresses:
        try:
            addresses = [ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, None)]
        except socket.gaierror as exc:
            raise ValueError("URL host could not be resolved") from exc
        if not addresses:
            raise ValueError("URL host could not be resolved")
    if any(not address.is_global for address in addresses):
        raise ValueError("Nonpublic URLs are not allowed")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    query = urlencode(sorted(
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ))
    return urlunsplit((parsed.scheme.lower(), host, parsed.path.rstrip("/") or "/", query, ""))


def _bing_target(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname not in {"bing.com", "www.bing.com"} or not parsed.path.startswith("/ck/"):
        return url
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if encoded.startswith("a1"):
        encoded = encoded[2:]
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return url
    return decoded if decoded.startswith(("http://", "https://")) else url


class _BingWebResults(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self.item: dict[str, str] | None = None
        self.in_title = False
        self.in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "li" and "b_algo" in (attributes.get("class") or "").split():
            self.item = {"title": "", "url": "", "snippet": ""}
        elif self.item is not None:
            if tag == "h2":
                self.in_title = True
            elif tag == "a" and self.in_title and not self.item["url"]:
                self.item["url"] = _bing_target(attributes.get("href") or "")
            elif tag == "p":
                self.in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self.in_title = False
        elif tag == "p":
            self.in_snippet = False
        elif tag == "li" and self.item is not None:
            if self.item["url"] and self.item["title"]:
                self.results.append(self.item)
            self.item = None

    def handle_data(self, data: str) -> None:
        if self.item is not None:
            if self.in_title:
                self.item["title"] += data
            elif self.in_snippet:
                self.item["snippet"] += data


class _BingImageResults(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag != "a" or "iusc" not in (attributes.get("class") or "").split():
            return
        try:
            item = json.loads(attributes.get("m") or "")
        except json.JSONDecodeError:
            return
        if isinstance(item, dict):
            self.results.append({
                "title": str(item.get("t") or ""), "url": str(item.get("purl") or ""),
                "snippet": "Image search metadata only; verify the source page. " + str(item.get("desc") or item.get("t") or ""),
            })


def _search_get(url: str, params: dict[str, str | int]) -> bytes:
    with requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT},
        timeout=SEARCH_TIMEOUT, stream=True,
    ) as response:
        response.raise_for_status()
        body, truncated = _read_limited(response, MAX_SEARCH_BYTES)
    if truncated:
        raise RuntimeError("Search response exceeded size limit")
    return body


def _search_tavily(query: str, key: str) -> list[dict[str, str]]:
    with requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": MAX_RESULTS, "search_depth": "basic"},
        timeout=SEARCH_TIMEOUT, stream=True,
    ) as response:
        response.raise_for_status()
        body, truncated = _read_limited(response, MAX_SEARCH_BYTES)
    if truncated:
        raise RuntimeError("Search response exceeded size limit")
    data = json.loads(body)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Search response has no result list")
    return [
        {"title": str(item.get("title") or ""), "url": str(item.get("url") or ""),
         "snippet": str(item.get("content") or "")}
        for item in data["results"] if isinstance(item, dict)
    ]


def _search_bing_rss(query: str) -> list[dict[str, str]]:
    root = ET.fromstring(_search_get("https://www.bing.com/search", {"q": query, "format": "rss"}))
    return [
        {"title": item.findtext("title") or "", "url": item.findtext("link") or "",
         "snippet": _html_text(html.unescape(item.findtext("description") or ""))}
        for item in root.findall("./channel/item")
    ]


def _search_bing_html(query: str) -> list[dict[str, str]]:
    parser = _BingWebResults()
    parser.feed(_search_get("https://www.bing.com/search", {"q": query, "count": 10}).decode("utf-8", errors="replace"))
    return parser.results


def _search_bing_images(query: str) -> list[dict[str, str]]:
    parser = _BingImageResults()
    parser.feed(_search_get("https://www.bing.com/images/search", {"q": query}).decode("utf-8", errors="replace"))
    return parser.results


def _search_wikipedia(query: str) -> list[dict[str, str]]:
    data = json.loads(_search_get("https://en.wikipedia.org/w/rest.php/v1/search/page", {"q": query, "limit": 10}))
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise ValueError("Wikipedia search response has no pages")
    results = []
    for page in data["pages"]:
        if not isinstance(page, dict) or not isinstance(page.get("key"), str):
            continue
        results.append({
            "title": str(page.get("title") or page["key"]),
            "url": "https://en.wikipedia.org/wiki/" + quote(page["key"], safe="()_"),
            "snippet": str(page.get("description") or "") + " " + str(page.get("excerpt") or ""),
        })
    return results


def _usable_results(candidates: list[dict[str, str]], *, query: str = "") -> list[dict[str, str]]:
    ranked: list[tuple[int, int, dict[str, str]]] = []
    seen: set[str] = set()
    terms = {term for term in re.findall(r"[a-z0-9]{3,}", query.lower()) if term not in SEARCH_STOPWORDS}
    for index, item in enumerate(candidates):
        try:
            url = _public_url(item["url"], resolve=False)
        except ValueError:
            continue
        canonical = _canonical_url(url)
        if canonical in seen:
            continue
        title = _html_text(item["title"])[:200]
        snippet = _html_text(item["snippet"])[:1200]
        if not title and not snippet:
            continue
        relevance = 0
        if len(terms) >= 3:
            words = set(re.findall(r"[a-z0-9]+", (title + " " + snippet).lower()))
            relevance = sum(
                term in words or (len(term) >= 4 and any(word.startswith(term) for word in words))
                for term in terms
            )
            if relevance < max(2, (len(terms) * 3 + 4) // 5):
                continue
        ranked.append((relevance, -index, {"title": title, "url": url, "snippet": snippet}))
        seen.add(canonical)
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    results = []
    hosts: dict[str, int] = {}
    for _, _, item in ranked:
        host = urlsplit(item["url"]).hostname or ""
        if hosts.get(host, 0) >= 2:
            continue
        results.append(item)
        hosts[host] = hosts.get(host, 0) + 1
        if len(results) == MAX_RESULTS:
            break
    return results


def search_web(query: str, *, tavily_api_key: str = "") -> dict:
    """Aggregate independent public indexes so one weak result cannot end a search."""
    query = query.strip()
    if not query or len(query) > 300:
        raise ValueError("Search query must contain 1 to 300 characters")
    backends = ([ ("tavily", lambda: _search_tavily(query, tavily_api_key)) ] if tavily_api_key else [])
    backends.extend([
        ("bing_rss", lambda: _search_bing_rss(query)),
        ("wikipedia", lambda: _search_wikipedia(query)),
    ])
    available: dict[str, list[dict[str, str]]] = {}
    failures: dict[str, str] = {}

    def collect(items: list[tuple[str, object]]) -> None:
        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = {name: executor.submit(search) for name, search in items}
            for name, future in futures.items():
                try:
                    candidates = future.result()
                    usable = _usable_results(candidates, query=query)
                    if usable:
                        available[name] = usable
                    else:
                        failures[name] = "no_results"
                except (requests.RequestException, ET.ParseError, ValueError, RuntimeError) as exc:
                    failures[name] = type(exc).__name__

    collect(backends)
    if len(available) < 2:
        collect([("bing_html", lambda: _search_bing_html(query))])
    image_query = bool(re.search(r"\b(image|photo|picture|photograph|visual)\b", query, re.I))
    if not available or image_query and len(available) < 2:
        collect([("bing_images", lambda: _search_bing_images(query))])

    candidates = [item for items in available.values() for item in items]
    ranked = _usable_results(candidates, query=query)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    hosts: dict[str, int] = {}

    def add(item: dict[str, str]) -> None:
        key = _canonical_url(item["url"])
        host = urlsplit(item["url"]).hostname or ""
        if key in seen or hosts.get(host, 0) >= 2 or len(selected) >= MAX_RESULTS:
            return
        selected.append(item)
        seen.add(key)
        hosts[host] = hosts.get(host, 0) + 1

    for items in available.values():
        add(items[0])
    for item in ranked:
        add(item)
    if not selected:
        reason = ", ".join(f"{name}:{failure}" for name, failure in failures.items()) or "no_results"
        return {
            "provider": "none", "query": query, "results": [], "sources": {},
            "untrusted_content": True, "result_kind": "none", "degraded_search": True,
            "warning": f"Search coverage degraded: no relevant public results; backends={reason}",
            "backend_failures": failures,
        }

    origins = {
        _canonical_url(item["url"]): [
            name for name, items in available.items()
            if any(_canonical_url(candidate["url"]) == _canonical_url(item["url"]) for candidate in items)
        ]
        for item in selected
    }
    providers = list(dict.fromkeys(name for item in selected for name in origins[_canonical_url(item["url"])]))
    image_only = all(set(names) == {"bing_images"} for names in origins.values())
    domains = {urlsplit(item["url"]).hostname.removeprefix("www.") for item in selected}
    wikipedia_only = domains == {"en.wikipedia.org"}
    degraded_reason = (
        "image metadata only" if image_only else
        "Wikipedia-only coverage" if wikipedia_only else
        "only one relevant source domain" if len(domains) < 2 else ""
    )
    return {
        "provider": "+".join(providers), "query": query, "results": selected,
        "sources": {item["url"]: origins[_canonical_url(item["url"])] for item in selected},
        "untrusted_content": True,
        "result_kind": "image_metadata" if image_only else "web_pages",
        "degraded_search": bool(degraded_reason),
        "warning": f"Search coverage degraded: {degraded_reason}; configure a reliable search API for broad web results" if degraded_reason else None,
        "backend_failures": failures,
    }


def preflight_web_search(query: str, *, tavily_api_key: str = "") -> dict:
    """Require relevant results from at least two independent source domains."""
    result = search_web(query, tavily_api_key=tavily_api_key)
    if result["degraded_search"]:
        raise RuntimeError(result["warning"])
    return result


def _cache_get(url: str) -> dict | None:
    with _page_cache_lock:
        entry = _page_cache.get(url)
        if entry is None:
            return None
        timestamp, result = entry
        if time.monotonic() - timestamp > PAGE_CACHE_TTL_SECONDS:
            del _page_cache[url]
            return None
        _page_cache.move_to_end(url)
        return {**result, "cache_hit": True}


def _cache_put(url: str, result: dict) -> None:
    with _page_cache_lock:
        _page_cache[url] = (time.monotonic(), result.copy())
        _page_cache.move_to_end(url)
        while len(_page_cache) > PAGE_CACHE_SIZE:
            _page_cache.popitem(last=False)


def _excerpt_quality(text: str, truncated: bool) -> str:
    if not text.strip():
        return "empty"
    if truncated:
        return "truncated"
    if len(text.strip()) < 200:
        return "thin"
    return "substantive"


def fetch_url(url: str) -> dict:
    """Fetch a public HTML, text or PDF page with redirect and size limits."""
    requested = _public_url(url, resolve=True)
    cached = _cache_get(requested)
    if cached is not None:
        return cached
    current = requested
    for redirect_count in range(4):
        if redirect_count:
            current = _public_url(current, resolve=True)
        try:
            with requests.get(
                current, headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT,
                stream=True, allow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise RuntimeError("Page redirect omitted Location")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    return {"url": current, "error": f"HTTP {response.status_code}"}
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if not (content_type.startswith("text/") or content_type in {"application/pdf", "application/json", "application/xml"}):
                    raise RuntimeError(f"Unsupported page content type: {content_type or 'unknown'}")
                body, byte_truncated = _read_limited(response, MAX_FETCH_BYTES)
                encoding = response.encoding or "utf-8"
        except requests.RequestException as exc:
            raise RuntimeError(f"Page fetch failed ({type(exc).__name__})") from exc
        if content_type == "application/pdf":
            import fitz

            try:
                with fitz.open(stream=body, filetype="pdf") as document:
                    text = "\n".join(document[index].get_text() for index in range(min(document.page_count, 20)))
                    page_truncated = document.page_count > 20
            except Exception as exc:
                raise RuntimeError(f"PDF extraction failed ({type(exc).__name__})") from exc
        else:
            decoded = body.decode(encoding, errors="replace")
            text = _html_text(decoded) if content_type == "text/html" else decoded
            page_truncated = False
        truncated = byte_truncated or page_truncated or len(text) > MAX_PAGE_CHARS
        result = {
            "url": current, "content_type": content_type, "text": text[:MAX_PAGE_CHARS],
            "truncated": truncated, "excerpt_quality": _excerpt_quality(text, truncated),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "sha256": hashlib.sha256(text[:MAX_PAGE_CHARS].encode("utf-8")).hexdigest(),
            "cache_hit": False,
        }
        _cache_put(requested, result)
        return result
    raise RuntimeError("Page exceeded redirect limit")
