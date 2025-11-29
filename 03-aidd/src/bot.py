import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Bot initialized")
    print(f"Telegram token: {'SET' if os.getenv('TELEGRAM_BOT_TOKEN') else 'NOT SET'}")

if __name__ == "__main__":
    main()

