import http.client
import ast
import json
import os
import random
import re
import ssl
import threading
import logging
from collections import deque
from typing import Dict, List, Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("event_card_generator")


class CardGenerationError(Exception):
    pass


class DeepSeekLLMGenerator:

    def __init__(self, model: str = "deepseek-v4-flash"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in .env")

        self.api_key = api_key
        self.model = model
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.thinking_type = os.getenv("DEEPSEEK_THINKING_TYPE", "disabled").strip().lower()
        if self.thinking_type not in {"disabled", "enabled"}:
            logger.warning(
                "Unsupported DEEPSEEK_THINKING_TYPE=%r, falling back to 'disabled'",
                self.thinking_type,
            )
            self.thinking_type = "disabled"

        parsed = urlparse(self.base_url)
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._path = (parsed.path or "").rstrip("/") + "/chat/completions"
        self._ssl_ctx = ssl.create_default_context()
        self._tls = threading.local()

    def create_session(self) -> List[Dict[str, str]]:
        return []

    def _get_conn(self) -> http.client.HTTPSConnection:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            try:
                conn.request("HEAD", "/", headers={"Connection": "keep-alive"})
                conn.getresponse().read()
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
                self._tls.conn = None
        if conn is None:
            conn = http.client.HTTPSConnection(
                self._host, self._port, context=self._ssl_ctx, timeout=60
            )
            self._tls.conn = conn
        return conn

    def _drop_conn(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._tls.conn = None

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_format: Dict[str, Any] | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": {"type": self.thinking_type},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)
        body_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }

        for attempt in range(2):
            try:
                conn = self._get_conn()
                conn.request("POST", self._path, body=body_bytes, headers=headers)
                resp = conn.getresponse()
                raw = resp.read().decode("utf-8")

                if resp.status != 200:
                    logger.warning(
                        "DeepSeek API HTTP error status=%s body=%s",
                        resp.status,
                        raw[:500].replace("\n", "\\n"),
                    )
                    raise RuntimeError(f"DeepSeek API HTTP error {resp.status}: {raw[:500]}")

                data = json.loads(raw)
                break
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, OSError) as e:
                self._drop_conn()
                logger.warning("DeepSeek API connection error (attempt %d): %s", attempt + 1, e)
                if attempt == 1:
                    raise RuntimeError(f"DeepSeek API connection error: {e}") from e
            except json.JSONDecodeError as e:
                logger.warning("DeepSeek API returned non-JSON body: %s", e)
                raise RuntimeError(f"DeepSeek API returned non-JSON response: {e}") from e

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"DeepSeek API response missing choices: {data}")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"DeepSeek API response missing text content: {data}")

        return content.strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        session: List[Dict[str, str]] | None = None,
        response_format: Dict[str, Any] | None = None,
        extra_body: Dict[str, Any] | None = None,
    ) -> str:
        if session is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return self._call_api(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                extra_body=extra_body,
            )

        messages = [
            *session,
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        content = self._call_api(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            extra_body=extra_body,
        )
        session.append({"role": "user", "content": user_prompt})
        session.append({"role": "assistant", "content": content})
        return content

    def cleanup(self) -> None:
        self._drop_conn()


class EventCardGenerator:

    def __init__(self, llm_generator: DeepSeekLLMGenerator, verbose: bool = True, session_mode: str = "per-card"):
        self.gpt = llm_generator
        self.max_delta_sum = 40
        self.max_delta_per_stat = 20
        self.verbose = verbose
        self.session_mode = session_mode
        self.situation_repair_attempts = max(1, int(os.getenv("SITUATION_REPAIR_ATTEMPTS", 1)))
        self.options_repair_attempts = max(1, int(os.getenv("OPTIONS_REPAIR_ATTEMPTS", 1)))
        self._global_session = self.gpt.create_session() if session_mode == "global" else None
        self._theme_pool = [
            "petition", "accusation", "shortage", "discovery",
            "delegation", "ceremony", "intrigue", "loyalty",
            "inheritance", "tradition", "rumor", "rivalry",
            "arrival", "celebration", "punishment", "reform",
        ]
        self._domain_pool = [
            "village", "court", "trade", "military",
            "scholarly", "religious", "foreign", "personal",
        ]
        self._theme_bag: List[str] = []
        self._domain_bag: List[str] = []
        self._last_domain: str | None = None
        self._bag_lock = threading.Lock()
        self._recent_situations: deque[str] = deque(maxlen=10)
        self._stopwords = {
            "about", "after", "again", "against", "along", "also", "among", "another", "because",
            "before", "being", "between", "could", "first", "from", "have", "into", "king", "local",
            "market", "must", "near", "only", "over", "plain", "road", "short", "should", "small",
            "some", "that", "their", "there", "these", "they", "this", "town", "very", "village",
            "watch", "where", "which", "with", "would",
        }

    def _new_card_session(self) -> List[Dict[str, str]] | None:
        if self.session_mode == "none":
            return None
        if self.session_mode == "global":
            return self._global_session
        return self.gpt.create_session()

    def _pick_theme_domain(self) -> tuple[str, str]:
        with self._bag_lock:
            if not self._theme_bag:
                self._theme_bag = list(self._theme_pool)
                random.shuffle(self._theme_bag)
            if not self._domain_bag:
                self._domain_bag = list(self._domain_pool)
                random.shuffle(self._domain_bag)

            theme = self._theme_bag.pop()

            domain = None
            if self._last_domain is not None and len(self._domain_bag) > 1:
                for idx in range(len(self._domain_bag) - 1, -1, -1):
                    if self._domain_bag[idx] != self._last_domain:
                        domain = self._domain_bag.pop(idx)
                        break
            if domain is None:
                domain = self._domain_bag.pop()
            self._last_domain = domain

            return theme, domain

    def _token_signature(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z]{4,}", text.lower()))
        return {token for token in tokens if token not in self._stopwords}

    def _is_too_similar_to_recent(self, situation: str) -> bool:
        current = self._token_signature(situation)
        if len(current) < 4:
            return False
        for prev in self._recent_situations:
            old = self._token_signature(prev)
            if not old:
                continue
            overlap = len(current & old)
            jaccard = overlap / max(1, len(current | old))
            if overlap >= 4 or jaccard >= 0.58:
                return True
        return False

    def _compress_text(self, text: str, target_chars: int, kind: str) -> str:
        if not text or len(text) <= target_chars:
            return text
        prompt = (
            f"Rewrite the following medieval {kind} in {target_chars} characters or fewer. "
            "Keep the core meaning. Return strictly JSON: {\"text\": \"<rewritten>\"}.\n"
            f"Source:\n{text}"
        )
        try:
            response = self.gpt.generate(
                "Return strictly valid JSON only. No markdown.",
                prompt,
                max_tokens=140,
                temperature=0.2,
                session=None,
                response_format={"type": "json_object"},
            )
            extracted = self._extract_first_json_object(response) or response
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                rewritten = parsed.get("text") or parsed.get(kind) or parsed.get("situation") or parsed.get("phrase")
                if isinstance(rewritten, str):
                    rewritten = self._compact_line(rewritten).strip().strip('"').strip("'")
                    if rewritten and len(rewritten) < len(text):
                        return rewritten
        except Exception as e:
            logger.warning("Compress %s failed: %s", kind, e)
        return text

    def _truncate_at_sentence(self, text: str, max_length: int = 400) -> str:
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        sentence_ends = []
        for match in re.finditer(r'[.!?](?=\s|$)', truncated):
            sentence_ends.append(match.end())

        if sentence_ends:
            return text[:sentence_ends[-1]].strip()
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return text[:last_space].strip()
        return truncated.strip()

    def _stats_compact(self, attributes: Dict[str, int]) -> str:
        if not isinstance(attributes, dict):
            raise ValueError("attributes must be an object with science/army/support/resources")
        return (
            f"sci={int(attributes['science'])} army={int(attributes['army'])} "
            f"sup={int(attributes['support'])} res={int(attributes['resources'])}"
        )

    def _status_context_compact(
        self,
        active_statuses: List[str] | None = None,
        status_values: Dict[str, int] | None = None,
    ) -> str:
        parts: List[str] = []
        values = status_values or {}

        for name, value in values.items():
            if int(value) > 0:
                parts.append(f"{name}={int(value)}")

        for name in active_statuses or []:
            if name not in values:
                parts.append(name)

        return ", ".join(parts) if parts else "none"

    def _first_n_sentences(self, text: str, n: int) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(parts[:n]).strip()

    def _compact_line(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _build_phrase_from_situation(self, situation: str) -> str:
        cleaned = self._compact_line(situation).strip().strip('"').strip("'")
        if not cleaned:
            return "Your Majesty, we need your judgment."
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return self._truncate_at_sentence(cleaned, max_length=140)

    def _adjust_deltas(self, deltas: Dict[str, int]) -> Dict[str, int]:
        cap = self.max_delta_per_stat
        normalized = {k: max(-cap, min(cap, int(v))) for k, v in deltas.items()}
        current_sum = sum(abs(v) for v in normalized.values())

        if current_sum == 0:
            return normalized

        if current_sum <= self.max_delta_sum:
            return normalized

        scale_factor = self.max_delta_sum / current_sum

        adjusted: Dict[str, int] = {}
        for key, value in normalized.items():
            scaled = value * scale_factor
            adjusted[key] = int(round(scaled))

        if sum(abs(v) for v in adjusted.values()) == 0:
            strongest_key = max(normalized.keys(), key=lambda k: abs(normalized[k]))
            adjusted[strongest_key] = 1 if normalized[strongest_key] > 0 else -1

        while sum(abs(v) for v in adjusted.values()) > self.max_delta_sum:
            key_to_reduce = max(adjusted.keys(), key=lambda k: abs(adjusted[k]))
            if adjusted[key_to_reduce] == 0:
                break
            adjusted[key_to_reduce] -= 1 if adjusted[key_to_reduce] > 0 else -1

        return adjusted

    def _extract_first_json_object(self, text: str) -> str | None:
        if not text:
            return None

        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        start = cleaned.find("{")
        if start == -1:
            return None

        depth = 0
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : idx + 1]
        return None

    def _normalize_options_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, list) and len(payload) >= 2:
            payload = {"option_1": payload[0], "option_2": payload[1]}
        if not isinstance(payload, dict):
            raise ValueError("options payload is not a JSON object")

        option_1 = (
            payload.get("option_1")
            or payload.get("option1")
            or payload.get("first_option")
            or payload.get("first")
        )
        option_2 = (
            payload.get("option_2")
            or payload.get("option2")
            or payload.get("second_option")
            or payload.get("second")
        )

        if option_1 is None or option_2 is None:
            raise ValueError("options payload missing option_1/option_2")

        return {"option_1": option_1, "option_2": option_2}

    def _parse_options_payload(self, raw_text: str) -> Dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            raise ValueError("empty options response")

        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        candidates: List[str] = []
        extracted = self._extract_first_json_object(text)
        if extracted:
            candidates.append(extracted)
        candidates.append(text)

        tried = []
        for candidate in candidates:
            snippet = candidate[:120].replace("\n", " ")
            if snippet in tried:
                continue
            tried.append(snippet)
            try:
                parsed = json.loads(candidate)
                return self._normalize_options_payload(parsed)
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(candidate)
                return self._normalize_options_payload(parsed)
            except Exception:
                pass

        raise ValueError("unable to parse options payload into JSON")

    def _normalize_situation_payload(self, payload: Any) -> Dict[str, str]:
        if not isinstance(payload, dict):
            raise ValueError("situation payload is not a JSON object")

        situation = (
            payload.get("situation")
            or payload.get("problem")
            or payload.get("context")
            or payload.get("story")
        )
        phrase = (
            payload.get("phrase")
            or payload.get("quote")
            or payload.get("complaint")
            or payload.get("visitor_quote")
        )

        if situation is None or phrase is None:
            raise ValueError("situation payload missing situation/phrase")
        return {"situation": str(situation), "phrase": str(phrase)}

    def _parse_situation_payload(self, raw_text: str) -> Dict[str, str]:
        text = (raw_text or "").strip()
        if not text:
            raise ValueError("empty situation response")

        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        candidates: List[str] = []
        extracted = self._extract_first_json_object(text)
        if extracted:
            candidates.append(extracted)
        candidates.append(text)

        tried = []
        for candidate in candidates:
            snippet = candidate[:120].replace("\n", " ")
            if snippet in tried:
                continue
            tried.append(snippet)
            try:
                parsed = json.loads(candidate)
                return self._normalize_situation_payload(parsed)
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(candidate)
                return self._normalize_situation_payload(parsed)
            except Exception:
                pass

        raise ValueError("unable to parse situation payload into JSON")

    def validate_deltas(self, option: Dict[str, Any]) -> bool:
        cap = self.max_delta_per_stat
        for key in ("science", "army", "support", "resources"):
            if abs(int(option.get(key, 0))) > cap:
                return False
        delta_sum = (
            abs(option.get("science", 0)) +
            abs(option.get("army", 0)) +
            abs(option.get("support", 0)) +
            abs(option.get("resources", 0))
        )
        return delta_sum <= self.max_delta_sum

    def validate_card(self, card: Dict[str, Any]) -> tuple[bool, str]:
        if "situation" not in card:
            return False, "Missing 'situation' field"
        if "phrase" not in card:
            return False, "Missing 'phrase' field"
        if "option_1" not in card or "option_2" not in card:
            return False, "Missing option fields"

        for opt_name in ["option_1", "option_2"]:
            option = card[opt_name]
            if "description" not in option:
                return False, f"Missing 'description' in {opt_name}"

            for attr in ["science", "army", "support", "resources"]:
                if attr not in option:
                    return False, f"Missing '{attr}' in {opt_name}"
                if not isinstance(option[attr], int):
                    return False, f"'{attr}' in {opt_name} must be integer"

            if not self.validate_deltas(option):
                return False, f"Delta sum exceeds {self.max_delta_sum} in {opt_name}"

        return True, "Valid"

    def generate_situation(
        self,
        attributes: Dict[str, int],
        session: List[Dict[str, str]] | None = None,
        theme: str | None = None,
        domain: str | None = None,
        active_statuses: List[str] | None = None,
        status_values: Dict[str, int] | None = None,
        month: str | None = None,
    ) -> tuple[str, str]:
        stats = self._stats_compact(attributes)
        statuses = self._status_context_compact(active_statuses, status_values)
        month_line = f"Month: {month}. Use the season subtly when it fits (harvest, frost, plague, festival).\n" if month else ""
        theme_line = f"Theme: {theme}. Domain: {domain}.\n" if theme and domain else ""

        situation_prompt = (
            f"Stats: {stats}. Statuses: {statuses}.\n"
            f"{theme_line}"
            f"{month_line}"
            "Return exactly one JSON object with keys: situation, phrase.\n"
            "situation: exactly 1 short sentence describing a medieval governance problem that fits the theme and domain.\n"
            "phrase: petitioner's quote to the king, <=18 words, problem only.\n"
            "Do NOT include actions/options/alternatives in either field.\n"
            "No markdown, no extra keys, no fantasy (no magic, dragons, elves)."
        )

        situation_system_prompt = "Return strictly valid JSON only."
        response = self.gpt.generate(
            situation_system_prompt,
            situation_prompt,
            max_tokens=96,
            temperature=0.4,
            session=session,
            response_format={"type": "json_object"},
        )
        try:
            situation_data = self._parse_situation_payload(response)
        except Exception as parse_error:
            logger.warning(
                "Primary situation parse failed: %s; raw=%s",
                parse_error,
                (response or "")[:300].replace("\n", "\\n"),
            )
            repair_system_prompt = "Return strictly valid JSON only. No markdown, no comments."
            repair_prompt = (
                "Return exactly one JSON object with keys situation and phrase.\n"
                "situation must be one concise sentence describing a medieval governance problem matching the theme and domain.\n"
                "phrase must be complaint quote <=18 words, no action proposals.\n"
                f"Stats: {stats}\nStatuses: {statuses}\n{theme_line.rstrip()}\n"
                f"Source text:\n{response}"
            )
            last_error = parse_error
            situation_data = None
            for repair_idx in range(self.situation_repair_attempts):
                repaired = self.gpt.generate(
                    repair_system_prompt,
                    repair_prompt,
                    max_tokens=120,
                    temperature=0.1,
                    session=None,
                    response_format={"type": "json_object"},
                )
                try:
                    situation_data = self._parse_situation_payload(repaired)
                    break
                except Exception as repair_error:
                    last_error = repair_error
                    logger.warning(
                        "Repair situation parse failed (attempt %d): %s; repaired=%s",
                        repair_idx + 1,
                        repair_error,
                        (repaired or "")[:300].replace("\n", "\\n"),
                    )
            if situation_data is None:
                raise ValueError(
                    f"situation JSON parse failed after repair attempts: {last_error}"
                ) from last_error

        raw_situation = self._compact_line(situation_data.get("situation", ""))
        situation = self._first_n_sentences(raw_situation, 1)
        situation = re.split(r"(?i)\b(option\s*1|option\s*2|choice\s*1|choice\s*2)\b", situation, maxsplit=1)[0].strip()
        situation = re.split(r"(?i)\b(should we|choose between|the king must choose)\b", situation, maxsplit=1)[0].strip()
        if len(situation) > 140:
            situation = self._compress_text(situation, 130, "situation")
        situation = self._truncate_at_sentence(situation, max_length=180)

        raw_phrase = self._compact_line(situation_data.get("phrase", "")).strip('"').strip("'")
        raw_phrase = re.split(
            r"(?i)\b(should we|choose between|option\s*1|option\s*2|choice\s*1|choice\s*2)\b",
            raw_phrase,
            maxsplit=1,
        )[0].strip()
        words = raw_phrase.split()
        if len(words) > 18:
            raw_phrase = " ".join(words[:18])
        if len(raw_phrase) > 130:
            raw_phrase = self._compress_text(raw_phrase, 120, "petitioner quote")
        phrase = self._truncate_at_sentence(raw_phrase, max_length=180)

        if not situation:
            situation = "A visitor comes to the king with an important matter."
        if not phrase:
            phrase = self._build_phrase_from_situation(situation)

        return situation, phrase

    def generate_options(
        self,
        situation: str,
        phrase: str,
        attributes: Dict[str, int],
        session: List[Dict[str, str]] | None = None,
        active_statuses: List[str] | None = None,
        status_values: Dict[str, int] | None = None,
        month: str | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        stats = self._stats_compact(attributes)
        statuses = self._status_context_compact(active_statuses, status_values)
        month_line = f"Month: {month}\n" if month else ""

        prompt = (
            f"Stats: {stats}\nStatuses: {statuses}\n{month_line}Situation: {situation}\nQuote: {phrase}\n"
            "2 king decisions. JSON only:\n"
            '{"option_1":{"description":"<4 words","science":int,"army":int,"support":int,"resources":int},'
            '"option_2":{...}}\n'
            f"Each stat delta MUST be an integer in [-{self.max_delta_per_stat}, {self.max_delta_per_stat}]. "
            f"Total |sum| across all four stats per option <= {self.max_delta_sum}."
        )

        system_prompt = "Valid JSON only."
        response = self.gpt.generate(
            system_prompt,
            prompt,
            max_tokens=360,
            temperature=0.2,
            session=session,
            response_format={"type": "json_object"},
        )
        try:
            data = self._parse_options_payload(response)
        except Exception as parse_error:
            logger.warning(
                "Primary options parse failed: %s; raw=%s",
                parse_error,
                (response or "")[:300].replace("\n", "\\n"),
            )
            repair_system_prompt = "Return strictly valid JSON only. No markdown, no comments."
            repair_prompt = (
                "Return exactly one JSON object with keys option_1 and option_2.\n"
                "Each option must contain: description, science, army, support, resources.\n"
                "description must be <=4 words.\n"
                f"Stats: {stats}\nStatuses: {statuses}\nSituation: {situation}\nQuote: {phrase}\n"
                "If the source text is malformed, reconstruct valid options from context.\n"
                f"Source text:\n{response}"
            )
            last_error = parse_error
            data = None
            for repair_idx in range(self.options_repair_attempts):
                repaired = self.gpt.generate(
                    repair_system_prompt,
                    repair_prompt,
                    max_tokens=420,
                    temperature=0.1,
                    session=None,
                    response_format={"type": "json_object"},
                )
                try:
                    data = self._parse_options_payload(repaired)
                    break
                except Exception as repair_error:
                    last_error = repair_error
                    logger.warning(
                        "Repair options parse failed (attempt %d): %s; repaired=%s",
                        repair_idx + 1,
                        repair_error,
                        (repaired or "")[:300].replace("\n", "\\n"),
                    )
            if data is None:
                raise ValueError(
                    f"options JSON parse failed after repair attempts: {last_error}"
                ) from last_error

        def normalize_option(option_data: Any, option_number: int) -> Dict[str, Any]:
            if not isinstance(option_data, dict):
                raise ValueError(f"option_{option_number} is not an object")

            deltas = {
                "science": int(option_data.get("science", 0)),
                "army": int(option_data.get("army", 0)),
                "support": int(option_data.get("support", 0)),
                "resources": int(option_data.get("resources", 0)),
            }
            deltas = self._adjust_deltas(deltas)

            description = str(option_data.get("description", "")).strip()[:200]
            if not description:
                description = f"Option {option_number}"

            words = description.split()
            if len(words) > 4:
                description = " ".join(words[:4])

            return {
                "description": description,
                "science": deltas["science"],
                "army": deltas["army"],
                "support": deltas["support"],
                "resources": deltas["resources"],
            }

        option_1 = normalize_option(data.get("option_1"), 1)
        option_2 = normalize_option(data.get("option_2"), 2)

        if option_1["description"].strip().lower() == option_2["description"].strip().lower():
            raise ValueError("option descriptions are identical")

        return option_1, option_2

    def generate_card(
        self,
        attributes: Dict[str, int],
        max_retries: int = 3,
        active_statuses: List[str] | None = None,
        status_values: Dict[str, int] | None = None,
        session: List[Dict[str, str]] | None = None,
        month: str | None = None,
    ) -> Dict[str, Any]:
        last_error: Exception | None = None
        attempt_errors: List[str] = []
        for attempt in range(max_retries):
            try:
                if self.verbose and attempt > 0:
                    print(f"  Retry {attempt}/{max_retries-1}")

                if session is None:
                    card_session = self._new_card_session()
                else:
                    card_session = list(session)
                theme, domain = self._pick_theme_domain()

                situation, phrase = self.generate_situation(
                    attributes,
                    session=card_session,
                    theme=theme,
                    domain=domain,
                    active_statuses=active_statuses,
                    status_values=status_values,
                    month=month,
                )

                option_1, option_2 = self.generate_options(
                    situation,
                    phrase,
                    attributes,
                    session=card_session,
                    active_statuses=active_statuses,
                    status_values=status_values,
                    month=month,
                )

                card = {
                    "situation": situation,
                    "phrase": phrase,
                    "option_1": option_1,
                    "option_2": option_2
                }

                is_valid, message = self.validate_card(card)
                if is_valid:
                    if session is not None and card_session is not None:
                        session.clear()
                        session.extend(card_session)
                    self._recent_situations.append(situation)
                    return card
                last_error = ValueError(f"Validation failed: {message}")
                attempt_errors.append(
                    f"#{attempt + 1} validation failed: {message}"
                )
                logger.warning(
                    "Card attempt %d/%d validation failed: %s",
                    attempt + 1,
                    max_retries,
                    message,
                )
                if self.verbose:
                    print(f"  Validation failed: {message}")

            except Exception as e:
                last_error = e
                attempt_errors.append(
                    f"#{attempt + 1} {type(e).__name__}: {e}"
                )
                logger.warning(
                    "Card attempt %d/%d failed: %s: %s",
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    e,
                )
                if self.verbose:
                    print(f"  Error: {e}")

        if attempt_errors:
            reason = " | ".join(attempt_errors[-3:])
        else:
            reason = str(last_error) if last_error else "unknown reason"
        logger.warning("Card generation exhausted retries: %s", reason)
        raise CardGenerationError(
            f"Failed to generate valid card after {max_retries} attempts: {reason}"
        ) from last_error

    def generate_cards(
        self,
        attributes: Dict[str, int],
        num_cards: int = 1,
        max_retries: int = 3,
        active_statuses: List[str] | None = None,
        status_values: Dict[str, int] | None = None,
        session: List[Dict[str, str]] | None = None,
        month: str | None = None,
    ) -> List[Dict[str, Any]]:
        if self.verbose:
            print(f"Generating {num_cards} card(s)...", end="", flush=True)

        cards = []
        for i in range(num_cards):
            card = self.generate_card(
                attributes,
                max_retries=max_retries,
                active_statuses=active_statuses,
                status_values=status_values,
                session=session,
                month=month,
            )
            cards.append(card)
            if self.verbose:
                print(f" {i+1}", end="", flush=True)

        if self.verbose:
            print(" Done")
        return cards
