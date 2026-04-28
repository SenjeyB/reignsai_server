from flask import Flask, jsonify, request, g, has_request_context
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
import uuid
from urllib.parse import parse_qs

from event_card_generator import EventCardGenerator, DeepSeekLLMGenerator, CardGenerationError
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
REQUEST_LOG_BODY_MAX = int(os.getenv('REQUEST_LOG_BODY_MAX', 800))
GENERATE_RATE_LIMIT = os.getenv(
    'GENERATE_RATE_LIMIT',
    '120 per minute' if ConfigClass.DEBUG else '20 per minute'
)
RANDOM_RATE_LIMIT = os.getenv(
    'RANDOM_RATE_LIMIT',
    '120 per minute' if ConfigClass.DEBUG else '30 per minute'
)
BATCH_RATE_LIMIT = os.getenv(
    'BATCH_RATE_LIMIT',
    '60 per minute' if ConfigClass.DEBUG else '10 per minute'
)
SESSION_INIT_RATE_LIMIT = os.getenv(
    'SESSION_INIT_RATE_LIMIT',
    '120 per minute' if ConfigClass.DEBUG else '30 per minute'
)
CHAT_SESSION_TTL_SECONDS = int(os.getenv('CHAT_SESSION_TTL_SECONDS', 7200))
CHAT_SESSION_MAX_ACTIVE = int(os.getenv('CHAT_SESSION_MAX_ACTIVE', 500))


class OverloadedError(Exception):
    pass


DEFAULT_ATTRIBUTES = {
    "science": 50,
    "army": 50,
    "support": 50,
    "resources": 50
}
ATTRIBUTE_ALIASES = {
    "science": ("science",),
    "army": ("army", "militia"),
    "support": ("support",),
    "resources": ("resources", "coffers"),
}

llm = DeepSeekLLMGenerator()
generator = EventCardGenerator(llm, verbose=False)

GEN_TIMEOUT = int(os.getenv('GEN_TIMEOUT', 60))
MAX_CARDS = getattr(ConfigClass, 'MAX_CARDS_PER_REQUEST', 20)
CARD_GEN_RETRIES = int(os.getenv('CARD_GEN_RETRIES', 5))


class ChatSessionStore:
    def __init__(self, ttl_seconds: int, max_active: int):
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_active = max(10, int(max_active))
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}

    def _cleanup_locked(self, now_ts: float) -> None:
        expired = [
            sid for sid, item in self._items.items()
            if (now_ts - item["last_used"]) > self.ttl_seconds
        ]
        for sid in expired:
            del self._items[sid]

        overflow = len(self._items) - self.max_active
        if overflow > 0:
            oldest = sorted(self._items.items(), key=lambda kv: kv[1]["last_used"])[:overflow]
            for sid, _ in oldest:
                del self._items[sid]

    def create(self) -> str:
        session_id = uuid.uuid4().hex
        now_ts = time.time()
        with self._lock:
            self._cleanup_locked(now_ts)
            self._items[session_id] = {
                "history": llm.create_session(),
                "lock": threading.RLock(),
                "created_at": now_ts,
                "last_used": now_ts,
            }
        return session_id

    def acquire(self, session_id: str | None):
        if not session_id:
            return None, None

        now_ts = time.time()
        with self._lock:
            self._cleanup_locked(now_ts)
            item = self._items.get(session_id)
            if item is None:
                return None, None
            item["last_used"] = now_ts
            history = item["history"]
            session_lock = item["lock"]

        session_lock.acquire()
        return history, session_lock


chat_sessions = ChatSessionStore(CHAT_SESSION_TTL_SECONDS, CHAT_SESSION_MAX_ACTIVE)


def _request_id():
    if not has_request_context():
        return "-"
    return getattr(g, "request_id", "-")


def _truncate_for_log(value, max_len=REQUEST_LOG_BODY_MAX):
    if value is None:
        return "null"
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = text.replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text


def _parse_request_json():
    data = request.get_json(silent=True)
    if data is None:
        return {}
    if isinstance(data, list):
        return {"attributes": data}
    if not isinstance(data, dict):
        logger.warning("[req:%s] Unsupported JSON body type: %s", _request_id(), type(data).__name__)
        return {}
    return data


def _normalize_attributes(raw_attributes):
    attributes = raw_attributes
    if isinstance(attributes, str):
        text = attributes.strip()
        if not text:
            attributes = {}
        else:
            if text.startswith("@ref ds_map("):
                logger.warning(
                    "[req:%s] attributes is GameMaker DS map reference (%s). "
                    "Client must send nested JSON map via ds_map_add_map/json_encode.",
                    _request_id(),
                    text,
                )
            try:
                attributes = json.loads(text)
            except json.JSONDecodeError:
                parsed_qs = parse_qs(text, keep_blank_values=False)
                if parsed_qs:
                    attributes = {k: v[-1] for k, v in parsed_qs.items()}
                else:
                    logger.warning("[req:%s] Unparseable attributes string, using defaults", _request_id())
                    attributes = {}

    if isinstance(attributes, list) and len(attributes) == 4:
        attributes = {
            "science": attributes[0],
            "army": attributes[1],
            "support": attributes[2],
            "resources": attributes[3],
        }

    if attributes is None:
        attributes = {}
    if hasattr(attributes, "items") and not isinstance(attributes, dict):
        attributes = dict(attributes.items())
    if not isinstance(attributes, dict):
        logger.warning(
            "[req:%s] Unsupported attributes type: %s, using defaults",
            _request_id(),
            type(attributes).__name__,
        )
        attributes = {}

    def _coerce_int(name, raw_value, default_value):
        if raw_value is None or raw_value == "":
            return default_value
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            try:
                return int(float(raw_value))
            except (TypeError, ValueError):
                logger.warning(
                    "[req:%s] Invalid attributes.%s=%r, using default=%d",
                    _request_id(),
                    name,
                    raw_value,
                    default_value,
                )
                return default_value

    lowered = {}
    for raw_key, raw_value in attributes.items():
        lowered[str(raw_key).strip().lower()] = raw_value

    normalized = {}
    for key, default_value in DEFAULT_ATTRIBUTES.items():
        raw_value = default_value
        for alias in ATTRIBUTE_ALIASES[key]:
            if alias in lowered:
                raw_value = lowered[alias]
                break
        normalized[key] = _coerce_int(key, raw_value, default_value)

    return normalized


def _normalize_status_name(raw_name):
    return str(raw_name).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_status_context(raw_statuses=None, raw_status_values=None):
    status_values = {}

    values_obj = raw_status_values
    if isinstance(values_obj, str):
        text = values_obj.strip()
        if text:
            try:
                values_obj = json.loads(text)
            except json.JSONDecodeError:
                parsed_qs = parse_qs(text, keep_blank_values=False)
                values_obj = {k: v[-1] for k, v in parsed_qs.items()} if parsed_qs else {}
        else:
            values_obj = {}

    if hasattr(values_obj, "items") and not isinstance(values_obj, dict):
        values_obj = dict(values_obj.items())
    if values_obj is None:
        values_obj = {}
    if isinstance(values_obj, dict):
        for raw_name, raw_value in values_obj.items():
            name = _normalize_status_name(raw_name)
            if name.startswith("status_"):
                name = name[7:]
            if not name:
                continue
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                try:
                    value = int(float(raw_value))
                except (TypeError, ValueError):
                    value = 0
            status_values[name] = value

    active_statuses = []
    seen_statuses = set()

    def add_active(raw_name):
        name = _normalize_status_name(raw_name)
        if name and name not in seen_statuses:
            active_statuses.append(name)
            seen_statuses.add(name)

    statuses_obj = raw_statuses
    if isinstance(statuses_obj, str):
        text = statuses_obj.strip()
        if text:
            try:
                statuses_obj = json.loads(text)
            except json.JSONDecodeError:
                statuses_obj = [part.strip() for part in text.split(",") if part.strip()]
        else:
            statuses_obj = []

    if isinstance(statuses_obj, dict):
        for key, value in statuses_obj.items():
            try:
                is_active = int(value) > 0
            except (TypeError, ValueError):
                is_active = bool(value)
            if is_active:
                add_active(key)
    elif isinstance(statuses_obj, (list, tuple, set)):
        for value in statuses_obj:
            add_active(value)

    for name, value in status_values.items():
        if value > 0:
            add_active(name)

    return active_statuses, status_values


def _normalize_count(raw_count, default_count):
    try:
        count = int(raw_count)
    except (TypeError, ValueError):
        logger.warning("[req:%s] Invalid count=%r, using default=%d", _request_id(), raw_count, default_count)
        count = default_count
    return max(1, min(count, MAX_CARDS))


def run_generation_safe(
    attributes,
    num_cards,
    active_statuses=None,
    status_values=None,
    chat_session=None,
    request_id="-",
):
    logger.info(
        "[req:%s] generation start count=%s attrs=%s statuses=%s status_values=%s session_ctx=%s",
        request_id,
        num_cards,
        _truncate_for_log(attributes),
        _truncate_for_log(active_statuses),
        _truncate_for_log(status_values),
        "yes" if chat_session is not None else "no",
    )
    acquired = llm_semaphore.acquire(timeout=LLM_QUEUE_ACQUIRE_TIMEOUT)
    if not acquired:
        logger.warning("[req:%s] generation rejected: semaphore acquire timeout", request_id)
        raise OverloadedError("Server busy - too many concurrent generation requests")
    try:
        attempts = 0
        while True:
            attempts += 1
            logger.info("[req:%s] provider attempt %d", request_id, attempts)
            future = _executor.submit(
                generator.generate_cards,
                attributes,
                num_cards,
                CARD_GEN_RETRIES,
                active_statuses,
                status_values,
                chat_session,
            )
            try:
                return future.result(timeout=GEN_TIMEOUT)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.warning("[req:%s] generation timeout after %ss", request_id, GEN_TIMEOUT)
                raise TimeoutError("Generation timeout")
            except Exception as e:
                logger.warning(
                    "[req:%s] provider attempt %d failed: %s: %s",
                    request_id,
                    attempts,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                status = None
                try:
                    status = getattr(getattr(e, 'response', None), 'status_code', None)
                except Exception:
                    status = None
                msg = str(e).lower()
                is_rate_limited = (status == 429) or ('too many requests' in msg) or ('429' in msg and 'too many' in msg)
                if is_rate_limited and attempts < LLM_RETRIES:
                    backoff = (LLM_BACKOFF_BASE ** attempts) + random.uniform(0, LLM_BACKOFF_JITTER)
                    logger.warning(
                        "[req:%s] upstream rate-limited, backoff %.2fs before retry",
                        request_id,
                        backoff,
                    )
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


@app.before_request
def _log_request_start():
    g.request_id = uuid.uuid4().hex[:10]
    g.request_started_at = time.perf_counter()
    body = None
    if request.method in ('POST', 'PUT', 'PATCH'):
        body = request.get_json(silent=True)
    logger.info(
        "[req:%s] -> %s %s args=%s body=%s",
        _request_id(),
        request.method,
        request.path,
        _truncate_for_log(dict(request.args)),
        _truncate_for_log(body),
    )


@app.after_request
def _log_request_end(response):
    started = getattr(g, "request_started_at", None)
    if isinstance(started, (float, int)):
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info("[req:%s] <- status=%s duration_ms=%.1f", _request_id(), response.status_code, duration_ms)
    else:
        logger.info("[req:%s] <- status=%s", _request_id(), response.status_code)
    response.headers["X-Request-Id"] = _request_id()
    return response


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "generator": "ready"
    })


@app.route('/api/v1/session/init', methods=['POST'])
@limiter.limit(SESSION_INIT_RATE_LIMIT)
def init_session():
    req_id = _request_id()
    session_id = chat_sessions.create()
    logger.info("[req:%s] session initialized session_id=%s", req_id, session_id)
    return jsonify({
        "success": True,
        "data": {
            "session_id": session_id
        }
    })


@app.route('/api/v1/cards/generate', methods=['POST'])
@limiter.limit(GENERATE_RATE_LIMIT)
def generate_cards():
    req_id = _request_id()
    chat_session = None
    chat_session_lock = None
    try:
        data = _parse_request_json()
        count = _normalize_count(data.get('count', 1), 1)
        attributes_source = data.get('attributes', data)
        attributes = _normalize_attributes(attributes_source)
        session_id = str(data.get("session_id", "")).strip()
        if session_id:
            chat_session, chat_session_lock = chat_sessions.acquire(session_id)
            if chat_session is None:
                logger.warning("[req:%s] unknown/expired session_id=%s; using stateless mode", req_id, session_id)

        raw_statuses = data.get('statuses')
        raw_status_values = data.get('status_values')
        if raw_status_values is None and isinstance(raw_statuses, dict):
            raw_status_values = raw_statuses
            raw_statuses = None
        if raw_status_values is None and 'in_war' in data:
            raw_status_values = {"in_war": data.get("in_war")}

        active_statuses, status_values = _normalize_status_context(raw_statuses, raw_status_values)
        cards = run_generation_safe(
            attributes,
            num_cards=count,
            active_statuses=active_statuses,
            status_values=status_values,
            chat_session=chat_session,
            request_id=req_id,
        )

        return jsonify({
            "success": True,
            "data": {
                "cards": cards,
                "count": len(cards),
                "session_id": session_id if session_id else None,
            }
        })
    except OverloadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504
    except CardGenerationError as e:
        logger.warning("[req:%s] Card generation failed: %s", req_id, e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 502
    except RuntimeError as e:
        logger.warning("[req:%s] Upstream rate limit: %s", req_id, e, exc_info=True)
        return jsonify({"success": False, "error": "Upstream rate limit. Try later."}), 503
    except Exception as e:
        logger.exception("[req:%s] Error in /api/v1/cards/generate", req_id)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        if chat_session_lock is not None:
            chat_session_lock.release()


@app.route('/api/v1/cards/random', methods=['GET'])
@limiter.limit(RANDOM_RATE_LIMIT)
def get_random_card():
    req_id = _request_id()
    chat_session = None
    chat_session_lock = None
    try:
        query = dict(request.args.items())
        count = _normalize_count(query.get('count', 1), 1)
        attributes = _normalize_attributes(query)
        session_id = str(query.get("session_id", "")).strip()
        if session_id:
            chat_session, chat_session_lock = chat_sessions.acquire(session_id)
            if chat_session is None:
                logger.warning("[req:%s] unknown/expired session_id=%s; using stateless mode", req_id, session_id)

        query_status_values = {}
        for key, value in query.items():
            key_lower = str(key).strip().lower()
            if key_lower.startswith("status_"):
                query_status_values[key_lower] = value
        if "in_war" in query:
            query_status_values["in_war"] = query.get("in_war")

        active_statuses, status_values = _normalize_status_context(
            raw_statuses=query.get("statuses"),
            raw_status_values=query_status_values if query_status_values else None,
        )
        cards = run_generation_safe(
            attributes,
            num_cards=count,
            active_statuses=active_statuses,
            status_values=status_values,
            chat_session=chat_session,
            request_id=req_id,
        )

        return jsonify({
            "success": True,
            "data": {
                "cards": cards,
                "count": len(cards),
                "session_id": session_id if session_id else None,
            }
        })
    except OverloadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504
    except CardGenerationError as e:
        logger.warning("[req:%s] Card generation failed: %s", req_id, e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 502
    except RuntimeError as e:
        logger.warning("[req:%s] Upstream rate limit: %s", req_id, e, exc_info=True)
        return jsonify({"success": False, "error": "Upstream rate limit. Try later."}), 503
    except Exception as e:
        logger.exception("[req:%s] Error in /api/v1/cards/random", req_id)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        if chat_session_lock is not None:
            chat_session_lock.release()


@app.route('/api/v1/cards/batch', methods=['POST'])
@limiter.limit(BATCH_RATE_LIMIT)
def get_batch_cards():
    req_id = _request_id()
    chat_session = None
    chat_session_lock = None
    try:
        data = _parse_request_json()
        count = _normalize_count(data.get('count', 3), 3)
        attributes_source = data.get('attributes', data)
        attributes = _normalize_attributes(attributes_source)
        session_id = str(data.get("session_id", "")).strip()
        if session_id:
            chat_session, chat_session_lock = chat_sessions.acquire(session_id)
            if chat_session is None:
                logger.warning("[req:%s] unknown/expired session_id=%s; using stateless mode", req_id, session_id)

        raw_statuses = data.get('statuses')
        raw_status_values = data.get('status_values')
        if raw_status_values is None and isinstance(raw_statuses, dict):
            raw_status_values = raw_statuses
            raw_statuses = None
        if raw_status_values is None and 'in_war' in data:
            raw_status_values = {"in_war": data.get("in_war")}

        active_statuses, status_values = _normalize_status_context(raw_statuses, raw_status_values)
        cards = run_generation_safe(
            attributes,
            num_cards=count,
            active_statuses=active_statuses,
            status_values=status_values,
            chat_session=chat_session,
            request_id=req_id,
        )

        return jsonify({
            "success": True,
            "data": {
                "cards": cards,
                "count": len(cards),
                "requested": count,
                "session_id": session_id if session_id else None,
            }
        })
    except OverloadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except TimeoutError as e:
        return jsonify({"success": False, "error": str(e)}), 504
    except CardGenerationError as e:
        logger.warning("[req:%s] Card generation failed: %s", req_id, e, exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 502
    except RuntimeError as e:
        logger.warning("[req:%s] Upstream rate limit: %s", req_id, e, exc_info=True)
        return jsonify({"success": False, "error": "Upstream rate limit. Try later."}), 503
    except Exception as e:
        logger.exception("[req:%s] Error in /api/v1/cards/batch", req_id)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        if chat_session_lock is not None:
            chat_session_lock.release()


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
    print("LLM: DeepSeek (deepseek-v4-flash)")
    print(f"DeepSeek thinking: {llm.thinking_type}")
    print("Ready to generate cards on-demand")
    app.run(host=ConfigClass.HOST, port=ConfigClass.PORT, debug=ConfigClass.DEBUG)
