"""Minimal async client for G2Bulk game top-up endpoints."""

from dataclasses import dataclass
from typing import Optional

import httpx

from config import DEFAULT_CATALOGUE_BY_GAME, G2BULK_API_KEY, G2BULK_BASE_URL

GAME_CODE_MAP = {
    "MLBB": "mlbb",
    "PUBG": "pubgm",
}


@dataclass
class TopupResult:
    ok: bool
    provider_status: str
    message: str
    provider_order_id: Optional[str] = None


def _parse_player_id(raw_player_id: str) -> tuple[str, Optional[str]]:
    value = raw_player_id.strip()
    for sep in ("|", ":", " "):
        if sep in value:
            left, right = value.split(sep, 1)
            if left.strip() and right.strip():
                return left.strip(), right.strip()
    return value, None


async def create_game_order(game: str, player_id_raw: str) -> TopupResult:
    if not G2BULK_API_KEY:
        return TopupResult(
            ok=False,
            provider_status="NOT_CONFIGURED",
            message="Missing G2BULK_API_KEY in environment.",
        )

    game_upper = game.upper()
    game_code = GAME_CODE_MAP.get(game)
    if not game_code:
        if "MLBB" in game_upper:
            game_code = "mlbb"
        elif "PUBG" in game_upper:
            game_code = "pubgm"
    if not game_code:
        return TopupResult(
            ok=False,
            provider_status="UNSUPPORTED_GAME",
            message=f"Game '{game}' is not mapped to a provider code.",
        )

    catalogue_name = DEFAULT_CATALOGUE_BY_GAME.get(game)
    if not catalogue_name:
        if game_code == "mlbb":
            catalogue_name = DEFAULT_CATALOGUE_BY_GAME.get("MLBB")
        elif game_code == "pubgm":
            catalogue_name = DEFAULT_CATALOGUE_BY_GAME.get("PUBG")
    if not catalogue_name:
        return TopupResult(
            ok=False,
            provider_status="CATALOGUE_MISSING",
            message=f"No default catalogue configured for game '{game}'.",
        )

    user_id, server_id = _parse_player_id(player_id_raw)
    payload = {
        "catalogue_name": catalogue_name,
        "player_id": user_id,
    }
    if server_id:
        payload["server_id"] = server_id

    headers = {
        "X-API-Key": G2BULK_API_KEY,
        "Content-Type": "application/json",
    }
    url = f"{G2BULK_BASE_URL}/games/{game_code}/order"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers=headers)
        data = response.json()
    except httpx.HTTPError as exc:
        return TopupResult(
            ok=False,
            provider_status="NETWORK_ERROR",
            message=str(exc),
        )
    except ValueError:
        return TopupResult(
            ok=False,
            provider_status="INVALID_RESPONSE",
            message=f"Provider returned non-JSON response (status {response.status_code}).",
        )

    if response.status_code >= 400 or not data.get("success", False):
        message = data.get("message", f"HTTP {response.status_code}")
        return TopupResult(
            ok=False,
            provider_status="FAILED",
            message=message,
        )

    order_data = data.get("order", {})
    provider_order_id = order_data.get("order_id")
    status = order_data.get("status", "PENDING")
    message = data.get("message", "Order created successfully.")

    return TopupResult(
        ok=True,
        provider_status=str(status),
        message=message,
        provider_order_id=str(provider_order_id) if provider_order_id is not None else None,
    )


async def fetch_public_products() -> list[dict]:
    """Fetch public product catalogue from G2Bulk."""
    url = f"{G2BULK_BASE_URL}/products"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return []

    products = data.get("products", [])
    if not isinstance(products, list):
        return []

    cleaned: list[dict] = []
    for item in products:
        if not isinstance(item, dict):
            continue
        product_id = item.get("id")
        title = str(item.get("title", "")).strip()
        if product_id is None or not title:
            continue
        cleaned.append(
            {
                "id": str(product_id),
                "title": title,
                "unit_price": item.get("unit_price"),
            }
        )
    return cleaned
