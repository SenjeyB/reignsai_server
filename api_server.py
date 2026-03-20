from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import json
import random
import os
import sys
import logging
import concurrent.futures
from functools import wraps
import threading
import time

from event_card_generator import EventCardGenerator, DeepSeekLLMGenerator
from config import config as CONFIG_MAP

app = Flask(__name__)
CORS(app)

env = os.getenv('ENV', os.getenv('FLASK_ENV', 'default'))
ConfigClass = CONFIG_MAP.get(env, CONFIG_MAP['default'])
app.config.from_object(ConfigClass)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=getattr(ConfigClass, 'RATELIMIT_STORAGE_URL', 'memory://'),
    default_limits=[getattr(ConfigClass, 'RATELIMIT_DEFAULT', "100 per hour")]
)

POOL_SIZE = getattr(ConfigClass, 'POOL_SIZE', 100)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=POOL_SIZE)

LLM_CONCURRENCY = int(os.getenv('LLM_CONCURRENCY', 4))
LLM_QUEUE_ACQUIRE_TIMEOUT = float(os.getenv('LLM_QUEUE_ACQUIRE_TIMEOUT', 5.0))
llm_semaphore = threading.BoundedSemaphore(LLM_CONCURRENCY)

LLM_RETRIES = int(os.getenv('LLM_RETRIES', 3))
LLM_BACKOFF_BASE = float(os.getenv('LLM_BACKOFF_BASE', 1.5))
LLM_BACKOFF_JITTER = float(os.getenv('LLM_BACKOFF_JITTER', 0.3))


class OverloadedError(Exception):
    pass


DEFAULT_ATTRIBUTES = {
    "science": 50,
    "army": 50,
    "support": 50,
    "resources": 50
}

llm = DeepSeekLLMGenerator()
generator = EventCardGenerator(llm, verbose=False)

GEN_TIMEOUT = int(os.getenv('GEN_TIMEOUT', 20))
MAX_CARDS = getattr(ConfigClass, 'MAX_CARDS_PER_REQUEST', 20)


def run_generation_safe(attributes, num_cards):
    acquired = llm_semaphore.acquire(timeout=LLM_QUEUE_ACQUIRE_TIMEOUT)
    if not acquired:
        raise OverloadedError("Server busy - too many concurrent generation requests")
    try:
        attempts = 0
        while True:
            attempts += 1
            future = _executor.submit(generator.generate_cards, attributes, num_cards)
            try:
                return future.result(timeout=GEN_TIMEOUT)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError("Generation timeout")
            except Exception as e:
                status = None
                try:
                    status = getattr(getattr(e, 'response', None), 'status_code', None)
                except Exception:
                    status = None
                msg = str(e).lower()
                is_rate_limited = (status == 429) or ('too many requests' in msg) or ('429' in msg and 'too many' in msg)
                if is_rate_limited and attempts < LLM_RETRIES:
                    backoff = (LLM_BACKOFF_BASE ** attempts) + random.uniform(0, LLM_BACKOFF_JITTER)
                    time.sleep(backoff)
                    continue
                if is_rate_limited:
                    raise RuntimeError("Upstream rate limit / provider throttling") from e
                raise
    finally:
        try:
            llm_semaphore.release()
        except Exception:
            pass


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "generator": "ready"
    })


@app.route('/api/v1/cards/generate', methods=['POST'])
@limiter.limit("3 per minute")
def generate_cards():
    try:
        data = request.get_json() or {}
        count = int(data.get('count', 1))
        count = max(1, min(count, MAX_CARDS))
        attributes = data.get('attributes', DEFAULT_ATTRIBUTES)

        cards = run_generation_safe(attributes, num_cards=count)

        return jsonify({
            "success": True,
            "data": {
                "cards": cards,
                "count": len(cards)
            }
        })
    except OverloadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504
    except RuntimeError as e:
        logger.warning("Upstream rate limit: %s", e)
        return jsonify({"success": False, "error": "Upstream rate limit. Try later."}), 503
    except Exception as e:
        logger.exception("Error in /api/v1/cards/generate")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/cards/random', methods=['GET'])
@limiter.limit("5 per minute")
def get_random_card():
    try:
        try:
            count = int(request.args.get('count', 1))
        except (ValueError, TypeError):
            count = 1
        count = max(1, min(count, MAX_CARDS))

        cards = run_generation_safe(DEFAULT_ATTRIBUTES, num_cards=count)

        return jsonify({
            "success": True,
            "data": {
                "cards": cards,
                "count": len(cards)
            }
        })
    except OverloadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504
    except RuntimeError as e:
        logger.warning("Upstream rate limit: %s", e)
        return jsonify({"success": False, "error": "Upstream rate limit. Try later."}), 503
    except Exception as e:
        logger.exception("Error in /api/v1/cards/random")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/cards/batch', methods=['POST'])
@limiter.limit("2 per minute")
def get_batch_cards():
    try:
        data = request.get_json() or {}
        count = int(data.get('count', 3))
        count = max(1, min(count, MAX_CARDS))
        attributes = data.get('attributes', DEFAULT_ATTRIBUTES)

        cards = run_generation_safe(attributes, num_cards=count)

        return jsonify({
            "success": True,
            "data": {
                "cards": cards,
                "count": len(cards),
                "requested": count
            }
        })
    except OverloadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504
    except RuntimeError as e:
        logger.warning("Upstream rate limit: %s", e)
        return jsonify({"success": False, "error": "Upstream rate limit. Try later."}), 503
    except Exception as e:
        logger.exception("Error in /api/v1/cards/batch")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "generator": "active",
        "llm_provider": "deepseek"
    })


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "success": False,
        "error": "Rate limit exceeded. Please try again later.",
        "retry_after": getattr(e, 'description', None)
    }), 429


if __name__ == '__main__':
    print("Event Card Generator API Server")
    print("LLM: DeepSeek (deepseek-chat)")
    print("Ready to generate cards on-demand")
    app.run(host=ConfigClass.HOST, port=ConfigClass.PORT, debug=ConfigClass.DEBUG)
