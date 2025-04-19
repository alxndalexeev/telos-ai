# tools/telegram_messenger.py

import requests
import sys
import argparse
import os

class TelegramMessenger:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, chat_id: str, text: str, parse_mode: str = None) -> dict:
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        response = requests.post(url, data=payload)
        response.raise_for_status()
        return response.json()

def main():
    parser = argparse.ArgumentParser(description="Send a message to a Telegram user or channel via Bot API.")
    parser.add_argument("--token", type=str, help="Telegram Bot API token. Can also be set via TELEGRAM_BOT_TOKEN env variable.")
    parser.add_argument("--chat_id", type=str, required=True, help="Target chat ID or channel username (e.g., @channelusername).")
    parser.add_argument("--message", type=str, required=True, help="Message text to send.")
    parser.add_argument("--parse_mode", type=str, choices=["Markdown", "HTML"], default=None, help="Parse mode for formatting (Markdown or HTML).")

    args = parser.parse_args()
    bot_token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("Error: Bot token must be provided via --token argument or TELEGRAM_BOT_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    messenger = TelegramMessenger(bot_token)
    try:
        result = messenger.send_message(args.chat_id, args.message, args.parse_mode)
        print("Message sent successfully:", result)
    except requests.HTTPError as e:
        print("Failed to send message:", e.response.text, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()