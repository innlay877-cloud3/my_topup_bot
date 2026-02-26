from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from config import ADMIN_ID, BOT_TOKEN, DB_PATH, G2BULK_API_KEY, G2BULK_BASE_URL

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_USD_TO_MMK = 4050
GAME_BUTTONS_PER_PAGE = 10
ITEMS_PER_PAGE = 8
CACHE_TTL_SECONDS = 120

# Anti-abuse: rate limit (requests per user per window), block bots, input/amount limits
RATE_LIMIT_MAX = 40
RATE_LIMIT_WINDOW_SEC = 60
MAX_INPUT_FIELD_LEN = 64
MAX_TOPUP_MMK = 5_000_000
MAX_ORDER_MMK = 2_000_000
_rate_limit: dict[int, list[float]] = {}


def _is_rate_limited(user_id: int) -> bool:
    if user_id == ADMIN_ID:
        return False
    now = time.time()
    if user_id not in _rate_limit:
        _rate_limit[user_id] = []
    times = _rate_limit[user_id]
    times.append(now)
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    _rate_limit[user_id] = [t for t in times if t > cutoff]
    return len(_rate_limit[user_id]) > RATE_LIMIT_MAX


def _check_abuse(update: Update) -> tuple[bool, str | None]:
    """Return (blocked, message). If blocked, caller should reply message and return."""
    user = update.effective_user
    if not user:
        return True, "Invalid request."
    if getattr(user, "is_bot", False):
        return True, None
    if _is_rate_limited(user.id):
        return True, "Too many requests. Please try again in a minute."
    return False, None


STATE_IDLE = "IDLE"
STATE_WAIT_INPUT = "WAIT_INPUT"
STATE_WAIT_RECEIPT = "WAIT_RECEIPT"
STATE_WAIT_TOPUP_AMOUNT = "WAIT_TOPUP_AMOUNT"

STATE_KEY = "state"
ORDER_KEY = "active_order_id"
SELECTED_GAME_KEY = "selected_game"
SELECTED_ITEM_KEY = "selected_item"
REQUIRED_FIELDS_KEY = "required_fields"
TOPUP_AMOUNT_KEY = "topup_amount"

GAME_ITEM_BUTTON = "Game item"
MAIN_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton(GAME_ITEM_BUTTON), KeyboardButton("💳 My Wallet")],
        [KeyboardButton("📦 My Orders")],
    ],
    resize_keyboard=True,
)

CACHE: dict[str, Any] = {
    "games": [],
    "games_updated": 0.0,
    "catalogues": {},
    "catalogues_updated": {},
}

PUBG_GROUP_UC = "UC"
PUBG_GROUP_PASS = "PASS"


@dataclass
class ProviderResult:
    ok: bool
    status: str
    message: str
    provider_order_id: str | None = None


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_local_db() -> None:
    with closing(_db()) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wallets (
                    user_id INTEGER PRIMARY KEY,
                    balance_mmk INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES ('usd_to_mmk', ?)",
                (str(DEFAULT_USD_TO_MMK),),
            )


def get_usd_to_mmk_rate() -> int:
    with closing(_db()) as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'usd_to_mmk'"
        ).fetchone()
    if not row:
        return DEFAULT_USD_TO_MMK
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return DEFAULT_USD_TO_MMK


def set_usd_to_mmk_rate(rate: int) -> None:
    with closing(_db()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO settings (key, value)
                VALUES ('usd_to_mmk', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(rate),),
            )


def ensure_wallet(user_id: int) -> None:
    with closing(_db()) as conn:
        with conn:
            conn.execute("INSERT OR IGNORE INTO wallets (user_id, balance_mmk) VALUES (?, 0)", (user_id,))


def get_wallet_balance(user_id: int) -> int:
    ensure_wallet(user_id)
    with closing(_db()) as conn:
        row = conn.execute("SELECT balance_mmk FROM wallets WHERE user_id = ?", (user_id,)).fetchone()
    return int(row["balance_mmk"]) if row else 0


def change_wallet_balance(user_id: int, delta_mmk: int) -> int:
    ensure_wallet(user_id)
    with closing(_db()) as conn:
        with conn:
            conn.execute("UPDATE wallets SET balance_mmk = balance_mmk + ? WHERE user_id = ?", (delta_mmk, user_id))
            row = conn.execute("SELECT balance_mmk FROM wallets WHERE user_id = ?", (user_id,)).fetchone()
    return int(row["balance_mmk"]) if row else 0


def reserve_wallet(user_id: int, amount_mmk: int) -> bool:
    ensure_wallet(user_id)
    with closing(_db()) as conn:
        with conn:
            row = conn.execute("SELECT balance_mmk FROM wallets WHERE user_id = ?", (user_id,)).fetchone()
            current = int(row["balance_mmk"]) if row else 0
            if current < amount_mmk:
                return False
            conn.execute("UPDATE wallets SET balance_mmk = balance_mmk - ? WHERE user_id = ?", (amount_mmk, user_id))
            return True


def create_order(
    *,
    user_id: int,
    username: str | None,
    game_code: str,
    game_name: str,
    item_id: str,
    item_name: str,
    usd_price: float,
    mmk_price: int,
    required_fields: list[str],
    input_values: dict[str, str],
) -> int:
    with closing(_db()) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO topup_orders (
                    user_id, username, game_code, game_name, item_id, item_name,
                    usd_price, mmk_price, required_fields_json, input_values_json, reserved_amount_mmk
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    username,
                    game_code,
                    game_name,
                    item_id,
                    item_name,
                    usd_price,
                    mmk_price,
                    json.dumps(required_fields),
                    json.dumps(input_values),
                    mmk_price,
                ),
            )
            return int(cursor.lastrowid)


def get_order(order_id: int) -> sqlite3.Row | None:
    with closing(_db()) as conn:
        return conn.execute("SELECT * FROM topup_orders WHERE id = ?", (order_id,)).fetchone()


def set_receipt_and_admin_msg(order_id: int, receipt_file_id: str, chat_id: int, message_id: int) -> None:
    with closing(_db()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE topup_orders
                SET receipt_file_id = ?, admin_message_chat_id = ?, admin_message_id = ?
                WHERE id = ?
                """,
                (receipt_file_id, chat_id, message_id, order_id),
            )


def update_order_result(
    order_id: int,
    *,
    status: str,
    provider_status: str,
    provider_message: str,
    provider_order_id: str | None = None,
) -> None:
    with closing(_db()) as conn:
        with conn:
            conn.execute(
                """
                UPDATE topup_orders
                SET status = ?, provider_status = ?, provider_message = ?, provider_order_id = ?
                WHERE id = ?
                """,
                (status, provider_status, provider_message, provider_order_id, order_id),
            )


def list_recent_orders(user_id: int, limit: int = 5) -> list[sqlite3.Row]:
    with closing(_db()) as conn:
        return conn.execute(
            """
            SELECT id, game_name, item_name, mmk_price, status
            FROM topup_orders
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()


async def _api_get(path: str, auth: bool = False) -> dict:
    headers = {"X-API-Key": G2BULK_API_KEY} if auth and G2BULK_API_KEY else {}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{G2BULK_BASE_URL}{path}", headers=headers)
        response.raise_for_status()
        return response.json()


async def _api_post(path: str, payload: dict, auth: bool = False) -> dict:
    headers = {"X-API-Key": G2BULK_API_KEY} if auth and G2BULK_API_KEY else {}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{G2BULK_BASE_URL}{path}", json=payload, headers=headers)
        try:
            data = response.json()
        except ValueError:
            data = {"success": False, "message": f"HTTP {response.status_code}"}
        if response.status_code >= 400:
            return {"success": False, "message": data.get("message", f"HTTP {response.status_code}")}
        return data


def _cache_fresh(updated_at: float) -> bool:
    return (time.time() - updated_at) <= CACHE_TTL_SECONDS


async def fetch_games(force_refresh: bool = False) -> list[dict]:
    if not force_refresh and _cache_fresh(float(CACHE["games_updated"])):
        return CACHE["games"]
    try:
        data = await _api_get("/games")
        games = []
        for row in data.get("games", []):
            code = str(row.get("code", "")).strip()
            name = str(row.get("name", "")).strip()
            if code and name:
                games.append({"code": code, "name": name})
        CACHE["games"] = games
        CACHE["games_updated"] = time.time()
        return games
    except httpx.HTTPError:
        return CACHE["games"]


async def fetch_catalogue(game_code: str, force_refresh: bool = False) -> list[dict]:
    cat_cache: dict = CACHE["catalogues"]
    ts_cache: dict = CACHE["catalogues_updated"]
    if not force_refresh and _cache_fresh(float(ts_cache.get(game_code, 0.0))):
        return cat_cache.get(game_code, [])
    try:
        data = await _api_get(f"/games/{game_code}/catalogue")
        items = []
        for row in data.get("catalogues", []):
            item_id = row.get("id")
            name = str(row.get("name", "")).strip()
            try:
                usd = float(row.get("amount"))
            except (TypeError, ValueError):
                continue
            if item_id is None or not name:
                continue
            items.append({"id": str(item_id), "name": name, "usd": usd})
        cat_cache[game_code] = items
        ts_cache[game_code] = time.time()
        CACHE["catalogues"] = cat_cache
        CACHE["catalogues_updated"] = ts_cache
        return items
    except httpx.HTTPError:
        return cat_cache.get(game_code, [])


async def check_player_account(game_code: str, values: dict[str, str], required_fields: list[str]) -> tuple[bool, str, str]:
    """Validate player/account via G2Bulk checkPlayerId. Returns (ok, message, player_name)."""
    user_id = (values.get("user_id") or values.get("player_id") or "").strip()
    if not user_id:
        return False, "User ID / Player ID is required.", ""
    payload = {"game": game_code.lower(), "user_id": user_id}
    server_id = (values.get("zone_id") or values.get("server_id") or "").strip()
    if server_id:
        payload["server_id"] = server_id
    try:
        data = await _api_post("/games/checkPlayerId", payload, auth=False)
    except Exception:
        return True, "", ""  # On API error, skip validation and allow order to proceed
    if data.get("valid") == "valid":
        return True, "", (data.get("name") or "").strip()
    return False, data.get("message") or "Incorrect account. Please check your User ID and Zone ID.", ""


async def fetch_api_balance() -> tuple[float | None, str | None]:
    if not G2BULK_API_KEY:
        return None, "API key not configured"
    try:
        data = await _api_get("/getMe", auth=True)
    except httpx.HTTPError:
        return None, "API request failed"
    if not data.get("success"):
        return None, str(data.get("message", "API error"))
    raw_balance = data.get("balance")
    try:
        return float(raw_balance), None
    except (TypeError, ValueError):
        return None, "Invalid balance response"


def _mmk(usd: float) -> int:
    raw_mmk = usd * get_usd_to_mmk_rate()
    # Round to market-friendly price steps:
    # - smaller amounts: nearest 50 MMK
    # - larger amounts: nearest 100 MMK
    step = 50 if raw_mmk < 10000 else 100
    return int(round(raw_mmk / step) * step)


# 2x diamond display names: hide raw amounts, show as bonus pairs (e.g. 55 -> 50+50).
DIAMOND_DISPLAY_NAMES = {"55": "50+50", "165": "150+150", "275": "250+250", "565": "500+500"}


def _item_display_name(name: str) -> str:
    """Display name override: e.g. 'Weekly' -> 'WEEKLY PASS'; 55 -> 50+50, etc."""
    s = (name or "").strip()
    if s.lower() == "weekly":
        return "WEEKLY PASS"
    if s in DIAMOND_DISPLAY_NAMES:
        return DIAMOND_DISPLAY_NAMES[s]
    return name or ""


def _is_mobile_legends(game: dict) -> bool:
    low_name = game["name"].lower()
    low_code = game["code"].lower()
    return "mobile legends" in low_name or "mlbb" in low_name or "mlbb" in low_code


def _is_pubg(game: dict) -> bool:
    low_name = game["name"].lower()
    low_code = game["code"].lower()
    return "pubg" in low_name or "pubg" in low_code


def _is_pubg_uc_category(game: dict) -> bool:
    name = game["name"].lower()
    code = game["code"].lower()
    return ("pubg" in name or "pubg" in code) and "uc" in name


def _is_pubg_pass_category(game: dict) -> bool:
    name = game["name"].lower()
    code = game["code"].lower()
    pass_words = ("pass", "prime", "pack", "royale")
    return ("pubg" in name or "pubg" in code) and any(word in name for word in pass_words)


def _is_honor_of_kings(game: dict) -> bool:
    low_name = game["name"].lower()
    low_code = game["code"].lower()
    return "honor of kings" in low_name or low_code in {"hok", "honor_of_kings"}


def _is_heartopia(game: dict) -> bool:
    low_name = game["name"].lower()
    low_code = game["code"].lower()
    return "heartopia" in low_name or "heartopia" in low_code


def _is_where_winds_meet(game: dict) -> bool:
    low_name = game["name"].lower()
    low_code = game["code"].lower()
    return "where winds meet" in low_name or "where_winds_meet" in low_code


def _is_excluded_mlbb_category(game: dict) -> bool:
    name = game["name"].lower()
    blocked = (
        "mobile legends limited promo",
        "mobile legends brazil",
        "mobile legends adventure",
    )
    return any(x in name for x in blocked)


def _chunk_games(games: list[dict], page_size: int) -> list[list[dict]]:
    if not games:
        return []
    return [games[i : i + page_size] for i in range(0, len(games), page_size)]


def _build_game_pages(all_games: list[dict]) -> list[list[dict]]:
    # Keep MLBB categories on first page block, then all other games on later pages.
    mlbb_games = [
        g
        for g in all_games
        if _is_mobile_legends(g) and not _is_excluded_mlbb_category(g)
    ]
    other_games = [g for g in all_games if not _is_mobile_legends(g)]

    priority_games: list[dict] = []
    used_codes: set[str] = set()

    def add_first_match(matcher) -> bool:
        game = next((g for g in other_games if matcher(g) and g["code"] not in used_codes), None)
        if not game:
            return False
        priority_games.append(game)
        used_codes.add(game["code"])
        return True

    # Prefer PUBG UC and PUBG PASS categories when they exist.
    found_pubg_uc = add_first_match(_is_pubg_uc_category)
    found_pubg_pass = add_first_match(_is_pubg_pass_category)

    # Fallback: if neither specific PUBG category exists, include generic PUBG.
    if not found_pubg_uc and not found_pubg_pass:
        add_first_match(_is_pubg)

    # Then add the rest of required page-2 priorities.
    add_first_match(_is_honor_of_kings)
    add_first_match(_is_heartopia)
    add_first_match(_is_where_winds_meet)

    remaining_other_games = [g for g in other_games if g["code"] not in used_codes]
    ordered_other_games = priority_games + remaining_other_games

    pages = _chunk_games(mlbb_games, GAME_BUTTONS_PER_PAGE) + _chunk_games(ordered_other_games, GAME_BUTTONS_PER_PAGE)
    if not pages:
        return [[]]
    return pages


def _game_page_info(all_games: list[dict], page: int) -> tuple[int, int]:
    """Return (current_page_1based, total_pages) for game category list."""
    pages = _build_game_pages(all_games)
    if not pages:
        return 1, 1
    max_page = len(pages) - 1
    current = max(0, min(page, max_page))
    return current + 1, len(pages)


def _game_menu_keyboard(all_games: list[dict], page: int) -> InlineKeyboardMarkup:
    pages = _build_game_pages(all_games)
    max_page = len(pages) - 1
    current_page = max(0, min(page, max_page))
    current_rows = pages[current_page]
    rows = []
    for game in current_rows:
        rows.append([InlineKeyboardButton(game["name"], callback_data=f"game:{game['code']}")])
    nav = []
    if current_page > 0:
        nav.append(InlineKeyboardButton("⬅️ Back", callback_data=f"gamepage:{current_page - 1}"))
    if current_page < max_page:
        nav.append(InlineKeyboardButton("Next ➡️", callback_data=f"gamepage:{current_page + 1}"))
    if nav:
        rows.append(nav)
    # Bottom row: Page 1, Page 2, ... (up to 5 per row)
    if max_page >= 0:
        page_buttons = [
            InlineKeyboardButton(f"Page {i + 1}", callback_data=f"gamepage:{i}")
            for i in range(max_page + 1)
        ]
        for i in range(0, len(page_buttons), 5):
            rows.append(page_buttons[i : i + 5])
    if not rows:
        rows = [[InlineKeyboardButton("No games available", callback_data="noop:games")]]
    return InlineKeyboardMarkup(rows)


def _slice_page(rows: list[dict], page: int) -> tuple[list[dict], int, int]:
    if not rows:
        return [], 0, 0
    max_page = (len(rows) - 1) // ITEMS_PER_PAGE
    current = max(0, min(page, max_page))
    start = current * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    return rows[start:end], current, max_page


def _sort_items_word_then_number_asc(items: list[dict]) -> list[dict]:
    """Sort: items starting with a word first (price asc), then items starting with a number (price asc)."""
    word_items = [i for i in items if i.get("name") and str(i["name"]).strip() and not str(i["name"]).strip()[0].isdigit()]
    number_items = [i for i in items if i.get("name") and str(i["name"]).strip() and str(i["name"]).strip()[0].isdigit()]
    word_sorted = sorted(word_items, key=lambda x: float(x.get("usd", 0)))
    number_sorted = sorted(number_items, key=lambda x: float(x.get("usd", 0)))
    return word_sorted + number_sorted


def _items_keyboard(game_code: str, items: list[dict], page: int) -> InlineKeyboardMarkup:
    ordered = _sort_items_word_then_number_asc(items)
    current_rows, current_page, max_page = _slice_page(ordered, page)
    rows = []
    for item in current_rows:
        label = f"{_item_display_name(item['name'])} | {_mmk(float(item['usd'])):,} MMK"
        rows.append([InlineKeyboardButton(label, callback_data=f"item:{game_code}:{item['id']}")])
    nav = []
    if current_page > 0:
        nav.append(InlineKeyboardButton("⬅️ Back", callback_data=f"itempage:{game_code}:{current_page - 1}"))
    if current_page < max_page:
        nav.append(InlineKeyboardButton("Next ➡️", callback_data=f"itempage:{game_code}:{current_page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("⬅️ Back to Games", callback_data="gamepage:0")])
    if not current_rows:
        rows = [[InlineKeyboardButton("No items available", callback_data="noop:items")]]
    return InlineKeyboardMarkup(rows)


def _pubg_group_keyboard(game_code: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("PUBG UC TOPUP", callback_data=f"pubggroup:{game_code}:{PUBG_GROUP_UC}:0")],
            [InlineKeyboardButton("PUBG MOBILE PASS", callback_data=f"pubggroup:{game_code}:{PUBG_GROUP_PASS}:0")],
            [InlineKeyboardButton("⬅️ Back to Games", callback_data="gamepage:0")],
        ]
    )


def _is_pubg_uc_item(item: dict) -> bool:
    name = str(item.get("name", "")).strip()
    # For PUBG UC TOPUP, match items that start with a number.
    return bool(name and name[0].isdigit())


def _is_pubg_pass_item(item: dict) -> bool:
    low = str(item.get("name", "")).lower()
    return any(k in low for k in ("pass", "prime", "pack", "royale"))


def _filter_pubg_items(items: list[dict], group: str) -> list[dict]:
    if group == PUBG_GROUP_UC:
        # Strict mode: show only UC items that start with number.
        uc_items = [item for item in items if _is_pubg_uc_item(item)]
        return sorted(uc_items, key=lambda item: float(item.get("usd", 0)))

    # Passes group
    pass_items = [item for item in items if _is_pubg_pass_item(item)]
    if pass_items:
        return pass_items
    # fallback: items that are not UC-like
    non_uc = [item for item in items if "uc" not in str(item.get("name", "")).lower()]
    if non_uc:
        return non_uc
    return items


def _pubg_items_keyboard(game_code: str, group: str, items: list[dict], page: int) -> InlineKeyboardMarkup:
    ordered = _sort_items_word_then_number_asc(items)
    current_rows, current_page, max_page = _slice_page(ordered, page)
    rows = []
    for item in current_rows:
        label = f"{_item_display_name(item['name'])} | {_mmk(float(item['usd'])):,} MMK"
        rows.append([InlineKeyboardButton(label, callback_data=f"item:{game_code}:{item['id']}")])
    nav = []
    if current_page > 0:
        nav.append(
            InlineKeyboardButton(
                "⬅️ Back",
                callback_data=f"pubggroup:{game_code}:{group}:{current_page - 1}",
            )
        )
    if current_page < max_page:
        nav.append(
            InlineKeyboardButton(
                "Next ➡️",
                callback_data=f"pubggroup:{game_code}:{group}:{current_page + 1}",
            )
        )
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("⬅️ Back to PUBG Menu", callback_data=f"game:{game_code}")])
    if not current_rows:
        rows = [[InlineKeyboardButton("No items available", callback_data="noop:items")]]
    return InlineKeyboardMarkup(rows)


def _required_fields_for_game(game: dict) -> list[str]:
    if _is_mobile_legends(game):
        return ["user_id", "zone_id"]
    return ["player_id"]


def _field_label(field: str) -> str:
    mapping = {"user_id": "User ID", "zone_id": "Zone ID", "player_id": "Player ID"}
    return mapping.get(field, field.replace("_", " ").title())


def _verified_id_lines(values: dict[str, str], required_fields: list[str]) -> str:
    """Build verified block lines: User ID with zone on one line (no Region label)."""
    emoji = {"user_id": "🆔", "player_id": "🎮"}
    labels = []
    for f in required_fields:
        if f in ("zone_id", "server_id"):
            continue
        v = (values.get(f) or "").strip()
        if not v:
            continue
        if f == "user_id" and ("zone_id" in required_fields or "server_id" in required_fields):
            extra = (values.get("zone_id") or values.get("server_id") or "").strip()
            if extra:
                v = f"{v}  {extra}"
        lbl = _field_label(f)
        e = emoji.get(f, "")
        labels.append(f"{e} {lbl}: {v}" if e else f"{lbl}: {v}")
    return "\n".join(labels) if labels else ""


def _parse_inputs(raw: str, fields: list[str]) -> dict[str, str] | None:
    text = (raw or "").strip()
    if len(text) > 200:
        return None
    parts = [x.strip() for x in text.replace("|", ",").split(",") if x.strip()]
    # If one token but two fields, try splitting on parentheses: value1(value2)
    if len(parts) == 1 and len(fields) == 2 and "(" in parts[0] and ")" in parts[0]:
        left, _, rest = parts[0].partition("(")
        right = rest.rstrip(")").strip() if rest else ""
        if left.strip() and right:
            parts = [left.strip(), right]
    if len(parts) != len(fields):
        return None
    result = {field: parts[i] for i, field in enumerate(fields)}
    for v in result.values():
        if len(v) > MAX_INPUT_FIELD_LEN:
            return None
    return result


def _reset_flow(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data[STATE_KEY] = STATE_IDLE
    for key in (ORDER_KEY, SELECTED_GAME_KEY, SELECTED_ITEM_KEY, REQUIRED_FIELDS_KEY, TOPUP_AMOUNT_KEY):
        context.user_data.pop(key, None)


def _admin_buttons(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("Approve", callback_data=f"admin:approve:{order_id}"),
            InlineKeyboardButton("Reject", callback_data=f"admin:reject:{order_id}"),
        ]]
    )


def _admin_caption(order: sqlite3.Row) -> str:
    return (
        "New Top-up Receipt\n\n"
        f"Order ID: {order['id']}\n"
        f"User ID: {order['user_id']}\n"
        f"Username: @{order['username'] or '(none)'}\n"
        f"Game: {order['game_name']} ({order['game_code']})\n"
        f"Item: {order['item_name']} (ID: {order['item_id']})\n"
        f"Price: {order['mmk_price']:,} MMK\n"
        f"Status: {order['status']}"
    )


def _append_decision(caption: str | None, decision: str) -> str:
    base = caption or "Order update"
    if "Decision:" in base:
        return base
    return f"{base}\n\nDecision: {decision}"


# Set to False to show KBZ Pay / Wave Pay and payment instructions again.
PAYMENT_TEMPORARILY_REMOVED = True

PAYMENT_INSTRUCTIONS = (
    "FOR PAYMENT\n\n"
    "KBZ Pay: 09740938277 (Name: Naw Phyu Shin Thant)\n"
    "WAVE PAY: 09740938277 (Name: Naw Phyu Shin Thant)\n\n"
    "Note မှာ Shop လိုရေးပေးပါရှင့်✔️\n\n"
    "📢အခြားငွေလွဲနံပါတ် လွှမိ့ပါက တာဝန်မယူပါ📌"
)


def _wallet_keyboard(is_admin: bool) -> InlineKeyboardMarkup:
    rows = []
    if not PAYMENT_TEMPORARILY_REMOVED:
        rows.append([InlineKeyboardButton("➕ Top up wallet", callback_data="wallet:topup")])
        if not is_admin:
            rows.append([
                InlineKeyboardButton("KBZ Pay", callback_data="wallet:kbzpay"),
                InlineKeyboardButton("Wave Pay", callback_data="wallet:wavepay"),
            ])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="wallet:back")])
    return InlineKeyboardMarkup(rows)


async def submit_provider_order(order: sqlite3.Row) -> ProviderResult:
    if not G2BULK_API_KEY:
        return ProviderResult(False, "NO_API_KEY", "Missing G2BULK_API_KEY")
    try:
        values = json.loads(order["input_values_json"])
    except (TypeError, ValueError):
        values = {}
    payload = {
        "catalogue_name": order["item_name"],
        "denom_id": order["item_id"],
        "player_id": values.get("player_id") or values.get("user_id", ""),
    }
    if values.get("zone_id"):
        payload["server_id"] = values["zone_id"]
    data = await _api_post(f"/games/{order['game_code']}/order", payload, auth=True)
    if not data.get("success"):
        message = str(data.get("message", "Provider error"))
        low = message.lower()
        if "stock" in low and ("out" in low or "empty" in low):
            return ProviderResult(False, "OUT_OF_STOCK", message)
        return ProviderResult(False, "FAILED", message)
    order_data = data.get("order", {})
    provider_order_id = order_data.get("order_id")
    status = str(order_data.get("status", "PENDING"))
    return ProviderResult(True, status, str(data.get("message", "Order created")), str(provider_order_id) if provider_order_id is not None else None)


async def post_init(app: Application) -> None:
    await app.bot.delete_webhook(drop_pending_updates=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _reset_flow(context)
    if update.message:
        await update.message.reply_text(
            "Welcome to Suii Auto Topup Bot!\nTo see price list tap to \"Game item\"",
            reply_markup=MAIN_MENU,
        )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _reset_flow(context)
    if update.message:
        await update.message.reply_text("Cancelled.", reply_markup=MAIN_MENU)


async def browse(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _reset_flow(context)
    games = await fetch_games(force_refresh=True)
    cur, total = _game_page_info(games, 0)
    text = f"Choose game category: (Page {cur} of {total})"
    if update.message:
        await update.message.reply_text(text, reply_markup=_game_menu_keyboard(games, page=0))
    elif update.callback_query and update.callback_query.message:
        await update.callback_query.message.reply_text(text, reply_markup=_game_menu_keyboard(games, page=0))


async def show_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    balance = get_wallet_balance(user.id)
    is_admin = user.id == ADMIN_ID
    rate_line = ""
    api_balance_line = ""
    local_balance_line = ""
    if is_admin:
        api_balance, api_error = await fetch_api_balance()
        if api_balance is None:
            api_balance_line = f"🌐 API Balance: {api_error or 'Unavailable'}\n"
        else:
            api_balance_line = f"🌐 API Balance: {api_balance:.2f} USD\n"
        rate_line = f"💱 Rate: 1 USD = {get_usd_to_mmk_rate():,} MMK\n"
    else:
        local_balance_line = f"💰 Balance: {balance:,} MMK\n"
    need_credits = "Need credits? Contact admin." if PAYMENT_TEMPORARILY_REMOVED else "Need credits? Tap Top up wallet below."
    text = (
        "💳 My Wallet (Local)\n\n"
        f"🆔 User ID: {user.id}\n"
        f"👤 Username: @{user.username or '(none)'}\n"
        f"{local_balance_line}"
        f"{api_balance_line}"
        f"{rate_line}\n"
        f"{need_credits}"
    )
    if update.message:
        await update.message.reply_text(text, reply_markup=_wallet_keyboard(is_admin))
    elif update.callback_query and update.callback_query.message:
        await update.callback_query.message.reply_text(text, reply_markup=_wallet_keyboard(is_admin))


async def add_balance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Admin only command.")
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /addbalance <user_id> <amount_mmk>")
        return
    try:
        user_id = int(context.args[0])
        amount = int(context.args[1])
    except ValueError:
        await update.message.reply_text("Invalid user_id or amount.")
        return
    new_balance = change_wallet_balance(user_id, amount)
    await update.message.reply_text(f"User {user_id} new balance: {new_balance:,} MMK")


async def set_rate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Admin only command.")
        return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /setrate <amount>")
        return
    try:
        rate = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid rate. Example: /setrate 4100")
        return
    if rate <= 0:
        await update.message.reply_text("Rate must be greater than 0.")
        return
    set_usd_to_mmk_rate(rate)
    await update.message.reply_text(f"Rate updated to {rate} MMK successfully!")


async def my_orders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    rows = list_recent_orders(update.effective_user.id)
    if not rows:
        await update.message.reply_text("No orders yet.")
        return
    lines = ["Recent orders:"]
    for row in rows:
        lines.append(f"#{row['id']} | {row['item_name']} | {row['mmk_price']:,} MMK | {row['status']}")
    await update.message.reply_text("\n".join(lines))


async def on_game_page(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        await query.answer()
        if msg:
            await query.message.reply_text(msg)
        return
    await query.answer()
    _, raw_page = query.data.split(":")
    games = await fetch_games(force_refresh=False)
    page_num = int(raw_page)
    cur, total = _game_page_info(games, page_num)
    await query.edit_message_text(
        f"Choose game category: (Page {cur} of {total})",
        reply_markup=_game_menu_keyboard(games, page=page_num),
    )


async def on_game_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        await query.answer()
        if msg:
            await query.message.reply_text(msg)
        return
    await query.answer()
    _, game_code = query.data.split(":")
    games = await fetch_games(force_refresh=False)
    game = next((g for g in games if g["code"] == game_code), None)
    if not game:
        await query.answer("Game not found.", show_alert=True)
        return
    items = await fetch_catalogue(game_code, force_refresh=True)
    context.user_data[SELECTED_GAME_KEY] = game
    if _is_pubg(game):
        await query.edit_message_text(
            f"{game['name']} - choose item type:",
            reply_markup=_pubg_group_keyboard(game_code),
        )
        return
    await query.edit_message_text(
        f"{game['name']} - choose top-up item:",
        reply_markup=_items_keyboard(game_code, items, page=0),
    )


async def on_item_page(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        await query.answer()
        if msg:
            await query.message.reply_text(msg)
        return
    await query.answer()
    _, game_code, raw_page = query.data.split(":")
    items = await fetch_catalogue(game_code, force_refresh=False)
    await query.edit_message_text("Choose top-up item:", reply_markup=_items_keyboard(game_code, items, page=int(raw_page)))


async def on_pubg_group_page(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        await query.answer()
        if msg:
            await query.message.reply_text(msg)
        return
    await query.answer()
    _, game_code, group, raw_page = query.data.split(":")
    items = await fetch_catalogue(game_code, force_refresh=False)
    filtered_items = _filter_pubg_items(items, group)
    await query.edit_message_text(
        "Choose top-up item:",
        reply_markup=_pubg_items_keyboard(game_code, group, filtered_items, page=int(raw_page)),
    )


async def on_item_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        await query.answer()
        if msg:
            await query.message.reply_text(msg)
        return
    await query.answer()
    _, game_code, item_id = query.data.split(":")
    games = await fetch_games(force_refresh=False)
    game = next((g for g in games if g["code"] == game_code), None)
    items = await fetch_catalogue(game_code, force_refresh=False)
    item = next((x for x in items if x["id"] == item_id), None)
    if not game or not item:
        await query.answer("Item not found.", show_alert=True)
        return
    required_fields = _required_fields_for_game(game)
    context.user_data[SELECTED_GAME_KEY] = game
    context.user_data[SELECTED_ITEM_KEY] = item
    context.user_data[REQUIRED_FIELDS_KEY] = required_fields
    context.user_data[STATE_KEY] = STATE_WAIT_INPUT
    labels = ", ".join(_field_label(f) for f in required_fields)
    await query.message.reply_text(
        f"Selected: {_item_display_name(item['name'])}\n"
        f"Price: {_mmk(float(item['usd'])):,} MMK\n\n"
        f"Send {labels} in one line.\n"
        "Format: value1,value2 or value1|value2 or UserID(ZoneID) e.g. 1190574855(13837)"
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        if msg:
            await update.message.reply_text(msg)
        return
    text = update.message.text.strip()
    if text == GAME_ITEM_BUTTON:
        await browse(update, context)
        return
    if text == "💳 My Wallet":
        await show_wallet(update, context)
        return
    if text == "📦 My Orders":
        await my_orders(update, context)
        return
    state = context.user_data.get(STATE_KEY, STATE_IDLE)
    if state == STATE_WAIT_TOPUP_AMOUNT:
        normalized = text.replace(",", "").strip()
        if not normalized.isdigit():
            await update.message.reply_text("Please enter a valid MMK amount. Example: 3000")
            return
        amount = int(normalized)
        if amount <= 0:
            await update.message.reply_text("Amount must be greater than 0.")
            return
        if amount > MAX_TOPUP_MMK:
            await update.message.reply_text(f"Maximum top-up amount is {MAX_TOPUP_MMK:,} MMK.")
            return
        context.user_data[TOPUP_AMOUNT_KEY] = amount
        context.user_data[STATE_KEY] = STATE_IDLE
        topup_msg = f"Top-up request: {amount:,} MMK\n\n"
        if not PAYMENT_TEMPORARILY_REMOVED:
            topup_msg += f"{PAYMENT_INSTRUCTIONS}\n\n"
        topup_msg += "Send payment proof to admin. Admin will credit your wallet after confirmation."
        await update.message.reply_text(
            topup_msg,
            reply_markup=_wallet_keyboard(update.effective_user.id == ADMIN_ID),
        )
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=(
                "Wallet top-up request\n\n"
                f"User ID: {update.effective_user.id}\n"
                f"Username: @{update.effective_user.username or '(none)'}\n"
                f"Requested Amount: {amount:,} MMK"
            ),
        )
        return

    if state == STATE_WAIT_RECEIPT:
        await update.message.reply_text("Please send receipt photo.")
        return
    if state != STATE_WAIT_INPUT:
        await update.message.reply_text("To see price list tap to \"Game item\".")
        return

    game = context.user_data.get(SELECTED_GAME_KEY)
    item = context.user_data.get(SELECTED_ITEM_KEY)
    required_fields = context.user_data.get(REQUIRED_FIELDS_KEY, [])
    if not game or not item or not isinstance(required_fields, list):
        _reset_flow(context)
        await update.message.reply_text("Session expired. Please select product again.")
        return

    values = _parse_inputs(text, required_fields)
    if values is None:
        labels = ", ".join(_field_label(f) for f in required_fields)
        await update.message.reply_text(f"Invalid format. Send: {labels} with comma separator.")
        return

    ok, err, player_name = await check_player_account(str(game["code"]), values, required_fields)
    if not ok:
        await update.message.reply_text(f"Incorrect account.\n{err}")
        return

    id_lines = _verified_id_lines(values, required_fields)
    verified_block = "✅ Account verified.\n\n"
    if player_name:
        verified_block += f"👤 Name: {player_name}\n"
    if id_lines:
        verified_block += f"{id_lines}\n\n"

    mmk_price = _mmk(float(item["usd"]))
    if mmk_price > MAX_ORDER_MMK:
        await update.message.reply_text(f"Order amount exceeds maximum ({MAX_ORDER_MMK:,} MMK).")
        return
    if not reserve_wallet(update.effective_user.id, mmk_price):
        balance = get_wallet_balance(update.effective_user.id)
        await update.message.reply_text(
            verified_block
            + f"⚠️ Insufficient wallet balance.\n"
            f"📌 Required: {mmk_price:,} MMK\n"
            f"💰 Current: {balance:,} MMK"
        )
        return

    order_id = create_order(
        user_id=update.effective_user.id,
        username=update.effective_user.username,
        game_code=str(game["code"]),
        game_name=str(game["name"]),
        item_id=str(item["id"]),
        item_name=str(item["name"]),
        usd_price=float(item["usd"]),
        mmk_price=mmk_price,
        required_fields=required_fields,
        input_values=values,
    )
    context.user_data[ORDER_KEY] = order_id
    context.user_data[STATE_KEY] = STATE_WAIT_RECEIPT
    summary = verified_block + f"Order #{order_id} created.\nReserved: {mmk_price:,} MMK from wallet.\nNow send your receipt photo."
    await update.message.reply_text(summary)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        if msg:
            await update.message.reply_text(msg)
        return
    if context.user_data.get(STATE_KEY) != STATE_WAIT_RECEIPT:
        await update.message.reply_text("No pending receipt upload.")
        return
    order_id = context.user_data.get(ORDER_KEY)
    if not order_id:
        _reset_flow(context)
        await update.message.reply_text("Order missing. Please start again.")
        return
    order = get_order(int(order_id))
    if not order:
        _reset_flow(context)
        await update.message.reply_text("Order not found. Please start again.")
        return
    receipt_file_id = update.message.photo[-1].file_id
    admin_message = await context.bot.send_photo(
        chat_id=ADMIN_ID,
        photo=receipt_file_id,
        caption=_admin_caption(order),
        reply_markup=_admin_buttons(int(order["id"])),
    )
    set_receipt_and_admin_msg(int(order["id"]), receipt_file_id, int(admin_message.chat_id), int(admin_message.message_id))
    _reset_flow(context)
    await update.message.reply_text("Receipt submitted. Waiting for admin approval.")


async def on_admin_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message:
        return
    if query.from_user.id != ADMIN_ID:
        await query.answer("Only admin can do this.", show_alert=True)
        return
    await query.answer()
    _, action, raw_id = query.data.split(":")
    order_id = int(raw_id)
    order = get_order(order_id)
    if not order:
        await query.edit_message_caption("Order not found.")
        return
    if order["status"] in {"Done", "Rejected", "Failed"}:
        await query.answer("Order already finalized.", show_alert=True)
        return

    if action == "reject":
        change_wallet_balance(int(order["user_id"]), int(order["reserved_amount_mmk"]))
        update_order_result(order_id, status="Rejected", provider_status="REJECTED_BY_ADMIN", provider_message="Rejected by admin.")
        await context.bot.send_message(
            chat_id=int(order["user_id"]),
            text=f"Order #{order_id} rejected.\nRefunded: {int(order['reserved_amount_mmk']):,} MMK to wallet.",
        )
        await query.edit_message_caption(caption=_append_decision(query.message.caption, "Rejected"), reply_markup=None)
        return

    result = await submit_provider_order(order)
    if result.ok:
        update_order_result(
            order_id,
            status="Done",
            provider_status=result.status,
            provider_message=result.message,
            provider_order_id=result.provider_order_id,
        )
        await context.bot.send_message(chat_id=int(order["user_id"]), text=f"Top-up successful.\nOrder #{order_id}\nStatus: Done")
        await query.edit_message_caption(caption=_append_decision(query.message.caption, f"Approved ({result.status})"), reply_markup=None)
        return

    if result.status == "OUT_OF_STOCK":
        change_wallet_balance(int(order["user_id"]), int(order["reserved_amount_mmk"]))
        update_order_result(order_id, status="Failed", provider_status=result.status, provider_message=result.message)
        await context.bot.send_message(
            chat_id=int(order["user_id"]),
            text=f"Order #{order_id} failed: item out of stock.\nRefunded: {int(order['reserved_amount_mmk']):,} MMK.",
        )
        await query.edit_message_caption(caption=_append_decision(query.message.caption, "Out of Stock (Refunded)"), reply_markup=None)
        return

    update_order_result(order_id, status="Pending", provider_status=result.status, provider_message=result.message)
    await context.bot.send_message(chat_id=int(order["user_id"]), text=f"Order #{order_id} pending provider retry. Reason: {result.message}")
    await query.edit_message_caption(caption=_append_decision(query.message.caption, f"Provider Error ({result.status})"), reply_markup=None)


async def on_noop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query:
        await update.callback_query.answer("Not available.", show_alert=False)


async def on_wallet_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    blocked, msg = _check_abuse(update)
    if blocked:
        await query.answer()
        if msg:
            await query.message.reply_text(msg)
        return
    await query.answer()
    _, action = query.data.split(":")
    if action == "topup":
        if PAYMENT_TEMPORARILY_REMOVED:
            await query.message.reply_text("Top-up is temporarily unavailable.")
            return
        _reset_flow(context)
        context.user_data[STATE_KEY] = STATE_WAIT_TOPUP_AMOUNT
        await query.message.reply_text(
            "Please enter the amount (in MMK) you want to deposit.\n"
            "Example: 3000\n\n"
            "If you want to cancel, send /cancel"
        )
        return
    if action == "kbzpay" or action == "wavepay":
        if PAYMENT_TEMPORARILY_REMOVED:
            await query.message.reply_text("Payment options are temporarily unavailable.")
        else:
            await query.message.reply_text(
                "Payment details\n\n" + PAYMENT_INSTRUCTIONS + "\n\n"
                "Send payment proof to admin. Include your User ID for faster wallet credit."
            )
        return
    if action == "back":
        _reset_flow(context)
        await query.message.reply_text("Back to main menu.", reply_markup=MAIN_MENU)
        return


async def post_init(app: Application) -> None:
    await app.bot.delete_webhook(drop_pending_updates=True)


def main() -> None:
    init_local_db()
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(CommandHandler("browse", browse))
    app.add_handler(CommandHandler("wallet", show_wallet))
    app.add_handler(CommandHandler("myorders", my_orders))
    app.add_handler(CommandHandler("addbalance", add_balance))
    app.add_handler(CommandHandler("setrate", set_rate))
    app.add_handler(CallbackQueryHandler(on_game_page, pattern=r"^gamepage:"))
    app.add_handler(CallbackQueryHandler(on_game_selected, pattern=r"^game:"))
    app.add_handler(CallbackQueryHandler(on_pubg_group_page, pattern=r"^pubggroup:"))
    app.add_handler(CallbackQueryHandler(on_item_page, pattern=r"^itempage:"))
    app.add_handler(CallbackQueryHandler(on_item_selected, pattern=r"^item:"))
    app.add_handler(CallbackQueryHandler(on_admin_action, pattern=r"^admin:"))
    app.add_handler(CallbackQueryHandler(on_wallet_action, pattern=r"^wallet:"))
    app.add_handler(CallbackQueryHandler(on_noop, pattern=r"^noop:"))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    logger.info("Bot running...")
    app.run_polling()


if __name__ == "__main__":
    main()
