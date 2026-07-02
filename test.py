import os
import uuid
from pathlib import Path

import requests
import urllib3
from dotenv import load_dotenv

# Загружаем .env именно из папки, где лежит этот файл.
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH, override=True)

API_KEY = os.getenv("GIGA_KEY", "").strip()

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

SCOPES = [
    "GIGACHAT_API_PERS",
    "GIGACHAT_API_B2B",
    "GIGACHAT_API_CORP",
]

# Убираем предупреждения, потому что для учебного проекта
# временно используем verify=False.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if not API_KEY:
    raise ValueError(
        "GIGA_KEY не найден. Проверь, что рядом с test.py лежит файл .env"
    )

print(f"Ключ найден. Длина ключа: {len(API_KEY)}")
print()


def check_scope(scope: str):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {API_KEY}",
    }

    response = requests.post(
        OAUTH_URL,
        data={"scope": scope},
        headers=headers,
        timeout=30,
        verify=False,
    )

    return response


for scope in SCOPES:
    try:
        response = check_scope(scope)
    except requests.RequestException as error:
        print(f"{scope}: ошибка соединения — {error}")
        continue

    if response.status_code == 200:
        data = response.json()

        print(f"{scope}: КЛЮЧ РАБОТАЕТ")
        print("GigaChat успешно выдал access token.")
        print()
        print("Добавь этот scope в business-rules.py:")
        print(f'    scope="{scope}",')

        if "expires_at" in data:
            print(f"\nТокен действует до: {data['expires_at']}")

        break

    try:
        error_message = response.json().get("message", "без сообщения")
    except ValueError:
        error_message = response.text

    print(f"{scope}: {response.status_code} — {error_message}")

else:
    print()
    print("Ни один scope не подошёл.")
    print("Проверь, что в .env вставлен именно Authorization Key.")
    print("В ключе не должно быть слова Basic, кавычек или пробелов.")