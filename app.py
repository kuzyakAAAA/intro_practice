from flask import Flask, request, session, jsonify, render_template
import importlib.util
from pathlib import Path

app = Flask(__name__)
app.secret_key = "instacart-demo-secret-key"
BASE_DIR = Path(__file__).parent


def load_business_rules():
    py_path = BASE_DIR / "business-rules.py"
    if not py_path.exists():
        raise FileNotFoundError(
            "Не найден business-rules.py."
        )

    spec = importlib.util.spec_from_file_location("business-rules", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.route("/", methods=["GET"])
def index():
    startup_error = None
    if not (BASE_DIR / "business-rules.py").exists():
        startup_error = (
            "Файл business-rules.py не найден. "            
        )

    return render_template(
        "index.html",
        user_id=session.get("user_id"),
        products=None,
        prompt_text=None,
        recommendation=None,
        error=startup_error,
    )


@app.route("/login", methods=["POST"])
def login():
    user_id = request.form.get("user_id", "").strip()
    if not user_id.isdigit():
        return render_template(
            "index.html",
            user_id=None,
            products=None,
            prompt_text=None,
            recommendation=None,
            error="user_id должен быть целым числом",
        )
    session["user_id"] = int(user_id)
    return render_template(
        "index.html",
        user_id=session.get("user_id"),
        products=None,
        prompt_text=None,
        recommendation=None,
        error=None,
    )


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return render_template(
        "index.html",
        user_id=None,
        products=None,
        prompt_text=None,
        recommendation=None,
        error=None,
    )


@app.route("/history", methods=["POST"])
def history():
    user_id = session.get("user_id")
    if not user_id:
        return render_template(
            "index.html",
            user_id=None,
            products=None,
            prompt_text=None,
            recommendation=None,
            error="Сначала авторизуйтесь по user_id",
        )

    try:
        rules = load_business_rules()
        rows = rules.get_user_history(user_id)

        if not rows:
            return render_template(
                "index.html",
                user_id=user_id,
                products=None,
                prompt_text=None,
                recommendation=None,
                error="Loginom вернул пустой список товаров",
            )

        products_text = rules.build_products_text(rows)
        prompt_text, recommendation = rules.ask_gigachat(user_id, products_text)

        return render_template(
            "index.html",
            user_id=user_id,
            products=rows,
            prompt_text=prompt_text,
            recommendation=recommendation,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            user_id=user_id,
            products=None,
            prompt_text=None,
            recommendation=None,
            error=f"Ошибка: {e}",
        )


@app.route("/api/history", methods=["POST"])
def api_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Сначала авторизуйтесь"}), 403

    try:
        rules = load_business_rules()
        rows = rules.get_user_history(user_id)
        products_text = rules.build_products_text(rows)
        prompt_text, recommendation = rules.ask_gigachat(user_id, products_text)
        return jsonify({
            "user_id": user_id,
            "products": rows,
            "prompt_text": prompt_text,
            "recommendation": recommendation,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
