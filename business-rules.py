import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat


BASE_DIR = Path(__file__).parent

# Загружаем переменные из .env, который находится рядом с этим файлом.
load_dotenv(BASE_DIR / ".env", override=True)

GIGA_KEY = os.getenv("GIGA_KEY", "").strip()
GIGA_SCOPE = os.getenv("GIGA_SCOPE", "GIGACHAT_API_PERS")
GIGA_MODEL = os.getenv("GIGA_MODEL", "GigaChat-2")

if not GIGA_KEY:
    raise ValueError("Ключ GIGA_KEY не найден в файле .env")


# URL опубликованных сервисов Loginom.
LOGINOM_BASE_URL = "https://edu.loginom.dev/lgi/rest/instacart_ws_kuzmin3"

LOGINOM_HISTORY_URL = f"{LOGINOM_BASE_URL}/GetUserHistory"
LOGINOM_FORGOTTEN_URL = f"{LOGINOM_BASE_URL}/GetForgottenProducts"
LOGINOM_RHYTHM_URL = f"{LOGINOM_BASE_URL}/GetPurchaseRhythm"


llm = GigaChat(
    credentials=GIGA_KEY,
    scope=GIGA_SCOPE,
    model=GIGA_MODEL,
    verify_ssl_certs=False,
    temperature=0.3,
    max_tokens=500,
)


def call_loginom_service(url: str, user_id: int):
    """Вызывает опубликованный сервис Loginom и возвращает строки результата."""
    payload = {
        "Variables": {
            "user_id": user_id
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Ошибка при обращении к Loginom: {error}") from error

    try:
        data = response.json()
        rows = data["DataSet"]["Rows"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(
            "Loginom вернул ответ в неожиданном формате"
        ) from error

    if not isinstance(rows, list):
        raise ValueError("Loginom не вернул список строк")

    return rows


def get_user_history(user_id: int):
    """Возвращает историю популярных товаров пользователя."""
    rows = call_loginom_service(LOGINOM_HISTORY_URL, user_id)

    if not rows:
        raise ValueError(
            "Пользователь не найден или у него нет истории покупок"
        )

    return rows


def build_products_text(rows):
    """Преобразует историю покупок в текст для GigaChat."""
    lines = []

    for index, row in enumerate(rows, start=1):
        product_name = row.get("product_name", "Неизвестный товар")

        department = row.get(
            "department_rus",
            row.get("department", "Неизвестный отдел"),
        )

        aisle = row.get("aisle", "Неизвестная категория")

        order_count = row.get(
            "order_count",
            row.get("purchase_count", "—"),
        )

        lines.append(
            f"{index}. {product_name}; "
            f"отдел: {department}; "
            f"категория: {aisle}; "
            f"заказов: {order_count}"
        )

    return "\n".join(lines)


def ask_gigachat(user_id: int, products_text: str):
    """Создаёт короткую рекомендацию по истории покупок клиента."""
    system_prompt = """
Ты персональный помощник покупателя в продуктовом онлайн-магазине.

Твоя задача: кратко и понятно объяснять покупательские привычки клиента.
Обращайся на "вы", используй дружелюбный, но не рекламный тон.
Не придумывай факты, которых нет в данных.
Пиши не больше 5–6 предложений.
""".strip()

    user_prompt = f"""
История популярных товаров пользователя с ID {user_id}:

{products_text}

Сделай краткий персональный вывод:
1. Какие предпочтения видны в покупках.
2. Какие категории или отделы пользователь выбирает чаще.
3. Один ненавязчивый совет для следующей корзины.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    recommendation = str(response.content).strip()

    return user_prompt, recommendation


def get_forgotten_products(user_id: int):
    """
    Возвращает товары, которые пользователь часто покупал,
    но не добавлял в несколько последних заказов.
    """
    return call_loginom_service(LOGINOM_FORGOTTEN_URL, user_id)


def build_forgotten_products_text(rows):
    """Преобразует забытые товары в текст для итоговой рекомендации."""
    if not rows:
        return "Явных товаров для напоминания не найдено."

    lines = []

    for row in rows:
        product_name = row.get("product_name", "Неизвестный товар")
        purchase_count = row.get("purchase_count", "—")
        missed_orders = row.get("orders_since_last_purchase", "—")

        lines.append(
            f"- {product_name}: покупали в {purchase_count} заказах, "
            f"отсутствует в последних {missed_orders} заказах"
        )

    return "\n".join(lines)


def get_purchase_rhythm(user_id: int):
    """Возвращает частоту, типичный день и время заказов пользователя."""
    rows = call_loginom_service(LOGINOM_RHYTHM_URL, user_id)

    if not rows:
        raise ValueError("Не удалось получить ритм покупок пользователя")

    rhythm = dict(rows[0])

    favorite_day = rhythm.get(
        "favorite_day",
        rhythm.get("order_dow_string", "Неизвестный день"),
    )

    favorite_hour = rhythm.get(
        "favorite_hour",
        rhythm.get("order_hour_of_day"),
    )

    rhythm["favorite_day"] = favorite_day

    try:
        rhythm["favorite_hour_text"] = f"{int(favorite_hour):02d}:00"
    except (TypeError, ValueError):
        rhythm["favorite_hour_text"] = "неизвестное время"

    return rhythm


def build_rhythm_text(rhythm):
    """Преобразует метрики ритма покупок в понятный текст."""
    total_orders = rhythm.get("total_orders", "—")
    avg_interval_days = rhythm.get("avg_interval_days")
    favorite_day = rhythm.get("favorite_day", "неизвестный день")
    favorite_hour = rhythm.get("favorite_hour_text", "неизвестное время")

    if avg_interval_days is None:
        interval_text = "недостаточно данных для расчёта интервала"
    else:
        interval_text = f"в среднем раз в {avg_interval_days} дня"

    return (
        f"Всего заказов: {total_orders}. "
        f"Пользователь делает заказ {interval_text}. "
        f"Чаще всего заказывает в {favorite_day} около {favorite_hour}."
    )


def build_final_context(history_rows, forgotten_rows, rhythm):
    """Собирает результаты всех сервисов в единый контекст для GigaChat."""
    history_text = build_products_text(history_rows)
    forgotten_text = build_forgotten_products_text(forgotten_rows)
    rhythm_text = build_rhythm_text(rhythm)

    return f"""
Популярные товары клиента:
{history_text}

Товары, которые пользователь мог забыть:
{forgotten_text}

Ритм покупок:
{rhythm_text}
""".strip()


def ask_final_advice(user_id: int, context: str):
    """Формирует 2–3 итоговых действия для пользователя."""
    system_prompt = """
Ты помощник продуктового онлайн-магазина.

На основе готовой аналитики сформируй 2–3 конкретных и полезных действия
для пользователя. Не выдумывай товары или привычки, которых нет в данных.
Не используй навязчивый рекламный тон. Пиши ясно и коротко.
""".strip()

    user_prompt = f"""
Данные по покупательскому поведению пользователя с ID {user_id}:

{context}

Сформируй блок "Что стоит сделать сейчас".

Требования:
- дай от 2 до 3 конкретных советов;
- нумеруй советы;
- объясняй совет через данные пользователя;
- если нет товаров для напоминания, не выдумывай их.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    recommendation = str(response.content).strip()

    return user_prompt, recommendation