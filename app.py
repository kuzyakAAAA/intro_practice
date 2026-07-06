import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session


BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env", override=True)

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "instacart-demo-secret-key",
)


BUSINESS_RULES_MODULE = None


def load_business_rules():
    """Загружает business-rules.py как Python-модуль."""
    global BUSINESS_RULES_MODULE

    if BUSINESS_RULES_MODULE is not None:
        return BUSINESS_RULES_MODULE

    rules_path = BASE_DIR / "business-rules.py"

    if not rules_path.exists():
        raise FileNotFoundError(
            "Файл business-rules.py не найден в папке проекта"
        )

    spec = importlib.util.spec_from_file_location(
        "business_rules_module",
        rules_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Не удалось загрузить business-rules.py"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    BUSINESS_RULES_MODULE = module

    return module


def render_page(**kwargs):
    """Передаёт данные в templates/index.html."""
    context = {
        "user_id": session.get("user_id"),
        "products": None,
        "prompt_text": None,
        "recommendation": None,
        "forgotten_products": None,
        "rhythm": None,
        "timing_summary": None,
        "final_prompt_text": None,
        "final_recommendation": None,
        "error": None,
    }

    context.update(kwargs)

    return render_template("index.html", **context)


def get_current_user_id():
    """Возвращает user_id текущего пользователя из Flask-сессии."""
    return session.get("user_id")


@app.route("/", methods=["GET"])
def index():
    """Главная страница."""
    startup_error = None

    if not (BASE_DIR / "business-rules.py").exists():
        startup_error = "Файл business-rules.py не найден."

    return render_page(error=startup_error)


@app.route("/login", methods=["POST"])
def login():
    """Сохраняет user_id в сессии."""
    user_id = request.form.get("user_id", "").strip()

    if not user_id.isdigit():
        return render_page(
            error="user_id должен быть целым положительным числом"
        )

    session["user_id"] = int(user_id)

    return render_page()


@app.route("/logout", methods=["POST"])
def logout():
    """Очищает текущую сессию."""
    session.clear()

    return render_page()


@app.route("/history", methods=["POST"])
def history():
    """Показывает историю популярных товаров и вывод GigaChat."""
    user_id = get_current_user_id()

    if not user_id:
        return render_page(
            error="Сначала авторизуйтесь"
        )

    try:
        rules = load_business_rules()

        products = rules.get_user_history(user_id)

        products_text = rules.build_products_text(products)

        prompt_text, recommendation = rules.ask_gigachat(
            user_id=user_id,
            products_text=products_text,
        )

        return render_page(
            products=products,
            prompt_text=prompt_text,
            recommendation=recommendation,
        )

    except Exception as error:
        return render_page(
            error=f"Не удалось получить историю покупок: {error}"
        )


@app.route("/forgotten-products", methods=["POST"])
def forgotten_products():
    """Показывает товары, которые пользователь мог забыть купить."""
    user_id = get_current_user_id()

    if not user_id:
        return render_page(
            error="Сначала авторизуйтесь"
        )

    try:
        rules = load_business_rules()

        products = rules.get_forgotten_products(user_id)

        return render_page(
            forgotten_products=products
        )

    except Exception as error:
        return render_page(
            error=f"Не удалось найти забытые товары: {error}"
        )


@app.route("/purchase-rhythm", methods=["POST"])
def purchase_rhythm():
    """
    Показывает:
    - средний, минимальный и максимальный интервал;
    - последний заказ;
    - ожидаемый день следующего заказа;
    - пиковое время заказа;
    - данные для графика активности по дням недели.
    """
    user_id = get_current_user_id()

    if not user_id:
        return render_page(
            error="Сначала авторизуйтесь"
        )

    try:
        rules = load_business_rules()

        rhythm = rules.get_purchase_rhythm(user_id)

        timing_rows = rules.get_order_timing(user_id)

        timing_summary = rules.build_timing_summary(
            timing_rows
        )

        return render_page(
            rhythm=rhythm,
            timing_summary=timing_summary,
        )

    except Exception as error:
        return render_page(
            error=f"Не удалось получить ритм покупок: {error}"
        )


@app.route("/final-advice", methods=["POST"])
def final_advice():
    """Формирует итоговые рекомендации GigaChat."""
    user_id = get_current_user_id()

    if not user_id:
        return render_page(
            error="Сначала авторизуйтесь"
        )

    products = None
    forgotten_products_data = None
    rhythm = None
    timing_summary = None

    try:
        rules = load_business_rules()

        products = rules.get_user_history(user_id)

        forgotten_products_data = rules.get_forgotten_products(
            user_id
        )

        rhythm = rules.get_purchase_rhythm(user_id)

        timing_rows = rules.get_order_timing(user_id)

        timing_summary = rules.build_timing_summary(
            timing_rows
        )

        context = rules.build_final_context(
            history_rows=products,
            forgotten_rows=forgotten_products_data,
            rhythm=rhythm,
            timing_summary=timing_summary,
        )

        final_prompt_text, final_recommendation = (
            rules.ask_final_advice(
                user_id=user_id,
                context=context,
            )
        )

        return render_page(
            products=products,
            forgotten_products=forgotten_products_data,
            rhythm=rhythm,
            timing_summary=timing_summary,
            final_prompt_text=final_prompt_text,
            final_recommendation=final_recommendation,
        )

    except Exception as error:
        return render_page(
            products=products,
            forgotten_products=forgotten_products_data,
            rhythm=rhythm,
            timing_summary=timing_summary,
            error=f"Не удалось сформировать итоговый совет: {error}",
        )


@app.route("/api/history", methods=["POST"])
def api_history():
    """API-вариант сервиса истории покупок."""
    user_id = get_current_user_id()

    if not user_id:
        return jsonify(
            {"error": "Сначала авторизуйтесь"}
        ), 403

    try:
        rules = load_business_rules()

        products = rules.get_user_history(user_id)

        products_text = rules.build_products_text(products)

        prompt_text, recommendation = rules.ask_gigachat(
            user_id=user_id,
            products_text=products_text,
        )

        return jsonify(
            {
                "user_id": user_id,
                "products": products,
                "prompt_text": prompt_text,
                "recommendation": recommendation,
            }
        )

    except Exception as error:
        return jsonify(
            {"error": str(error)}
        ), 500


@app.route("/api/dashboard", methods=["POST"])
def api_dashboard():
    """API-вариант полного аналитического кабинета."""
    user_id = get_current_user_id()

    if not user_id:
        return jsonify(
            {"error": "Сначала авторизуйтесь"}
        ), 403

    try:
        rules = load_business_rules()

        products = rules.get_user_history(user_id)

        forgotten_products_data = rules.get_forgotten_products(
            user_id
        )

        rhythm = rules.get_purchase_rhythm(user_id)

        timing_rows = rules.get_order_timing(user_id)

        timing_summary = rules.build_timing_summary(
            timing_rows
        )

        context = rules.build_final_context(
            history_rows=products,
            forgotten_rows=forgotten_products_data,
            rhythm=rhythm,
            timing_summary=timing_summary,
        )

        prompt_text, final_recommendation = rules.ask_final_advice(
            user_id=user_id,
            context=context,
        )

        return jsonify(
            {
                "user_id": user_id,
                "products": products,
                "forgotten_products": forgotten_products_data,
                "rhythm": rhythm,
                "timing_summary": timing_summary,
                "prompt_text": prompt_text,
                "final_recommendation": final_recommendation,
            }
        )

    except Exception as error:
        return jsonify(
            {"error": str(error)}
        ), 500


if __name__ == "__main__":
    app.run(debug=True)