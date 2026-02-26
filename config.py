"""Configuration for Telegram top-up bot."""

import os

# Telegram bot token from @BotFather.
BOT_TOKEN = os.getenv("BOT_TOKEN", "8271578609:AAHQtDMnhPgD5nx6EPFiUCkJnBgTqM99n7I")

# Telegram numeric user ID for the admin account.
ADMIN_ID = int(os.getenv("ADMIN_ID", "8325479692"))

# SQLite database file path.
DB_PATH = os.getenv("DB_PATH", "topup_bot.db")

# G2Bulk API settings.
G2BULK_BASE_URL = os.getenv("G2BULK_BASE_URL", "https://api.g2bulk.com/v1")
G2BULK_API_KEY = os.getenv("G2BULK_API_KEY", "372dc4e2cd05e3f694d47f676b02a2ceeb888c286c6070ccd6e782564b6aa8e8")

# Default package/denomination by game title used in this bot.
# Update these to match exact catalogue names returned by:
# GET /v1/games/:code/catalogue
DEFAULT_CATALOGUE_BY_GAME = {
    "MLBB": os.getenv("CATALOGUE_MLBB", "86 Diamonds"),
    "PUBG": os.getenv("CATALOGUE_PUBG", "60 UC"),
}
