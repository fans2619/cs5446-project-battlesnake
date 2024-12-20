import logging
import os
import typing

from dotenv import load_dotenv
from flask import Flask
from flask import request

# Load environment variables from .env file
load_dotenv()


def run_server(handlers: typing.Dict, custom_port=None):
    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        return handlers["info"]()

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        game_state = request.get_json()
        return handlers["move"](game_state)

    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        handlers["end"](game_state)
        return "ok"

    @app.after_request
    def identify_server(response):
        response.headers.set(
            "server", "battlesnake/github/starter-snake-python"
        )
        return response

    host = "0.0.0.0"
    port = custom_port or int(os.environ.get("PORT", "80"))

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    debug = os.environ.get("FLASK_ENV", "PROD") == "DEV"
    app.run(host=host, port=port, debug=debug)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
