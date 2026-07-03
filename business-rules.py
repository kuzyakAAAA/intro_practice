import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat


BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env", override=True)

GIGA_KEY = os.getenv("GIGA_KEY", "").strip()
GIGA_SCOPE = os.getenv("GIGA_SCOPE", "GIGACHAT_API_PERS")
GIGA_MODEL = os.getenv("GIGA_MODEL", "GigaChat-2")

if not GIGA_KEY:
    raise ValueError("Ключ GIGA_KEY не найден в файле .env")


LOGINOM_BASE_URL = "https://edu.loginom.dev/lgi/rest/instacart_ws_kuzmin3"

LOGINOM_HISTORY_URL = f"{LOGINOM_BASE_URL}/GetUserHistory"
LOGINOM_FORGOTTEN_URL = f"{LOGINOM_BASE_URL}/GetForgottenProducts"
LOGINOM_RHYTHM_URL = f"{LOGINOM_BASE_URL}/GetPurchaseRhythm"
LOGINOM_ORDER_TIMING_URL = f"{LOGINOM_BASE_URL}/GetOrderTiming"


# Нумерация order_dow в твоей базе Instacart.
DAYS_RU = {
    0: "Сб",
    1: "Вс",
    2: "Пн",
    3: "Вт",
    4: "Ср",
    5: "Чт",
    6: "Пт",
}


llm = GigaChat(
    credentials=GIGA_KEY,
    scope=GIGA_SCOPE,
    model=GIGA_MODEL,
    verify_ssl_certs=False,
    temperature=0.3,
    max_tokens=500,
)


def call_loginom_service(url: str, user_id: int):
    """Вызывает REST-метод Loginom и возвращает список строк."""
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
        raise RuntimeError(
            f"Ошибка при обращении к Loginom: {error}"
        ) from error

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


def to_int_or_none(value):
    """Преобразует значение Loginom в целое число."""
    if value is None:
        return None

    try:
        text = str(value).strip()

        if not text:
            return None

        return int(float(text.split()[0]))
    except (TypeError, ValueError):
        return None


def to_rounded_int_or_none(value):
    """Преобразует значение в округлённое целое число."""
    if value is None:
        return None

    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def get_day_name(day_number):
    """Преобразует номер дня недели в сокращённое название."""
    day_number = to_int_or_none(day_number)

    if day_number is None:
        return "Неизвестный день"

    return DAYS_RU.get(day_number, "Неизвестный день")


def format_hour(hour):
    """Преобразует час в формат ЧЧ:00."""
    hour = to_int_or_none(hour)

    if hour is None:
        return "Неизвестное время"

    return f"{hour:02d}:00"


def get_user_history(user_id: int):
    """Возвращает популярные товары пользователя."""
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
    """Формирует краткий вывод по истории покупок."""
    system_prompt = """
Ты персональный помощник покупателя в продуктовом онлайн-магазине.

Кратко и понятно объясняй покупательские привычки клиента.
Обращайся на "вы", используй дружелюбный, но не рекламный тон.
Не выдумывай факты, которых нет в данных.
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

    return user_prompt, str(response.content).strip()


def get_forgotten_products(user_id: int):
    """Возвращает товары, которые пользователь мог забыть купить."""
    return call_loginom_service(LOGINOM_FORGOTTEN_URL, user_id)


def build_forgotten_products_text(rows):
    """Преобразует список забытых товаров в текст."""
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
    """
    Возвращает статистику по периодичности покупок.

    Прогноз строится относительно последнего заказа в истории,
    потому что в Instacart нет реальной календарной даты заказа.
    """
    rows = call_loginom_service(LOGINOM_RHYTHM_URL, user_id)

    if not rows or not rows[0]:
        raise ValueError(
            "GetPurchaseRhythm вернул пустой результат. "
            "Проверь SQL и выходной порт в Loginom."
        )

    raw = dict(rows[0])

    avg_days = to_rounded_int_or_none(raw.get("avg_days"))
    min_days = to_rounded_int_or_none(raw.get("min_days"))
    max_days = to_rounded_int_or_none(raw.get("max_days"))

    last_order_dow = to_int_or_none(raw.get("last_order_dow"))

    expected_next_order_day = "Недостаточно данных"

    if avg_days is not None and last_order_dow is not None:
        expected_day_number = (
            last_order_dow + avg_days
        ) % 7

        expected_next_order_day = get_day_name(
            expected_day_number
        )

    return {
        "avg_days": avg_days,
        "min_days": min_days,
        "max_days": max_days,
        "orders_count": to_int_or_none(
            raw.get("orders_count")
        ),
        "last_order_number": to_int_or_none(
            raw.get("last_order_number")
        ),
        "last_order_day": get_day_name(
            raw.get("last_order_dow")
        ),
        "last_order_hour_text": format_hour(
            raw.get("last_order_hour")
        ),
        "expected_next_order_day": expected_next_order_day,
    }


def get_order_timing(user_id: int):
    """
    Возвращает распределение заказов по дням недели и часам.

    Используется для heatmap, графика по дням и поиска
    самого частого момента заказа.
    """
    rows = call_loginom_service(
        LOGINOM_ORDER_TIMING_URL,
        user_id,
    )

    if not rows:
        raise ValueError(
            "GetOrderTiming не вернул данные. "
            "Проверь SQL и выходной порт в Loginom."
        )

    result = []

    for row in rows:
        result.append(
            {
                "order_dow": to_int_or_none(
                    row.get("order_dow")
                ),
                "order_hour": to_int_or_none(
                    row.get("order_hour_of_day")
                ),
                "orders_count": to_int_or_none(
                    row.get("orders_count")
                ) or 0,
            }
        )

    return result


def build_timing_summary(timing_rows):
    """
    Формирует данные для:
    - самого частого времени заказа;
    - графика по дням недели;
    - heatmap.
    """
    weekly_counts = {
        day_number: 0
        for day_number in DAYS_RU
    }

    heatmap_data = []

    for row in timing_rows:
        day_number = row["order_dow"]
        hour = row["order_hour"]
        orders_count = row["orders_count"]

        if day_number in weekly_counts:
            weekly_counts[day_number] += orders_count

        heatmap_data.append(
            {
                "day_number": day_number,
                "day_name": get_day_name(day_number),
                "hour": hour,
                "hour_text": format_hour(hour),
                "orders_count": orders_count,
            }
        )

    weekly_activity = []

    for day_number, day_name in DAYS_RU.items():
        weekly_activity.append(
            {
                "day_number": day_number,
                "day_name": day_name,
                "orders_count": weekly_counts[day_number],
            }
        )

    peak_order_time = max(
        heatmap_data,
        key=lambda item: item["orders_count"],
        default=None,
    )

    if peak_order_time is None:
        peak_day = "Неизвестный день"
        peak_hour_text = "Неизвестное время"
        peak_orders_count = 0
    else:
        peak_day = peak_order_time["day_name"]
        peak_hour_text = peak_order_time["hour_text"]
        peak_orders_count = peak_order_time["orders_count"]

    return {
        "peak_day": peak_day,
        "peak_hour_text": peak_hour_text,
        "peak_orders_count": peak_orders_count,
        "weekly_activity": weekly_activity,
        "heatmap_data": heatmap_data,
    }


def build_rhythm_text(rhythm, timing_summary):
    """Преобразует ритм и время заказов в текст для GigaChat."""
    avg_days = rhythm.get("avg_days")

    if avg_days is None:
        forecast_text = "Недостаточно данных для прогноза."
    else:
        forecast_text = (
            f"Следующая покупка обычно происходит примерно через "
            f"{avg_days} дней после последнего заказа, "
            f"ориентировочно в {rhythm['expected_next_order_day']}."
        )

    return (
        f"Всего заказов: {rhythm.get('orders_count') or '—'}. "
        f"Средний интервал: {avg_days or '—'} дней. "
        f"Минимальный интервал: {rhythm.get('min_days') or '—'} дней. "
        f"Максимальный интервал: {rhythm.get('max_days') or '—'} дней. "
        f"Последний заказ в истории: №"
        f"{rhythm.get('last_order_number') or '—'}, "
        f"{rhythm.get('last_order_day')} "
        f"около {rhythm.get('last_order_hour_text')}. "
        f"Чаще всего пользователь оформляет заказ в "
        f"{timing_summary['peak_day']} около "
        f"{timing_summary['peak_hour_text']}. "
        f"{forecast_text}"
    )


def build_final_context(
    history_rows,
    forgotten_rows,
    rhythm,
    timing_summary,
):
    """Собирает результаты аналитики для итогового совета."""
    history_text = build_products_text(history_rows)
    forgotten_text = build_forgotten_products_text(forgotten_rows)

    rhythm_text = build_rhythm_text(
        rhythm,
        timing_summary,
    )

    return f"""
Популярные товары клиента:
{history_text}

Товары, которые пользователь мог забыть:
{forgotten_text}

Ритм и время заказов:
{rhythm_text}
""".strip()


def ask_final_advice(user_id: int, context: str):
    """Формирует 2–3 итоговых совета."""
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
- объясняй каждый совет данными пользователя;
- если нет товаров для напоминания, не выдумывай их.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return user_prompt, str(response.content).strip()