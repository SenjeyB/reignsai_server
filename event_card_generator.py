import http.client
import json
import os
import random
import re
import ssl
from collections import deque
from typing import Dict, List, Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


class DeepSeekLLMGenerator:

    def __init__(self, model: str = "deepseek-chat"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in .env")

        self.api_key = api_key
        self.model = model
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

        parsed = urlparse(self.base_url)
        self._host = parsed.hostname
        self._port = parsed.port or 443
        self._path = (parsed.path or "").rstrip("/") + "/chat/completions"
        self._ssl_ctx = ssl.create_default_context()
        self._conn: http.client.HTTPSConnection | None = None

    def create_session(self) -> List[Dict[str, str]]:
        return []

    def _get_conn(self) -> http.client.HTTPSConnection:
        if self._conn is not None:
            try:
                self._conn.request("HEAD", "/", headers={"Connection": "keep-alive"})
                self._conn.getresponse().read()
            except Exception:
                self._conn = None
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(
                self._host, self._port, context=self._ssl_ctx, timeout=30
            )
        return self._conn

    def _call_api(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
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
                    raise RuntimeError(f"DeepSeek API HTTP error {resp.status}: {raw[:500]}")

                data = json.loads(raw)
                break
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, OSError) as e:
                self._conn = None
                if attempt == 1:
                    raise RuntimeError(f"DeepSeek API connection error: {e}") from e
            except json.JSONDecodeError as e:
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
    ) -> str:
        if session is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return self._call_api(messages, max_tokens=max_tokens, temperature=temperature)

        messages = [
            *session,
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        content = self._call_api(messages, max_tokens=max_tokens, temperature=temperature)
        session.append({"role": "user", "content": user_prompt})
        session.append({"role": "assistant", "content": content})
        return content

    def cleanup(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


class EventCardGenerator:

    def __init__(self, llm_generator: DeepSeekLLMGenerator, verbose: bool = True, session_mode: str = "per-card"):
        self.gpt = llm_generator
        self.max_delta_sum = 40
        self.verbose = verbose
        self.session_mode = session_mode
        self._global_session = self.gpt.create_session() if session_mode == "global" else None
        self._local_focus_pool = [
            "market stall licenses",
            "bridge toll dispute",
            "well maintenance duty",
            "village watch staffing",
            "road repair labor",
            "grain storage spoilage",
            "fishing rights conflict",
            "mill queue scheduling",
            "guild apprentice quota",
            "harvest cart traffic",
            "town gate closing hours",
            "charcoal supply shortage",
            "canal lock repairs",
            "sheep grazing boundaries",
            "bakery flour rationing",
            "blacksmith coal allotment",
        ]
        self._recent_focuses: deque[str] = deque(maxlen=6)
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

    def _pick_focus(self, blocked_focuses: set[str] | None = None) -> str:
        blocked = blocked_focuses or set()
        candidates = [
            focus
            for focus in self._local_focus_pool
            if focus not in blocked and focus not in self._recent_focuses
        ]
        if not candidates:
            candidates = [focus for focus in self._local_focus_pool if focus not in blocked]
        if not candidates:
            candidates = self._local_focus_pool
        return random.choice(candidates)

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

    def _truncate_at_sentence(self, text: str, max_length: int = 400) -> str:
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        sentence_ends = []
        for match in re.finditer(r'[.!?](?=\s|$)', truncated):
            sentence_ends.append(match.end())

        if sentence_ends:
            return text[:sentence_ends[-1]].strip()
        else:
            last_space = truncated.rfind(' ')
            if last_space > 0:
                return text[:last_space].strip() + '...'
            return truncated.strip() + '...'

    def _stats_compact(self, attributes: Dict[str, int]) -> str:
        return (
            f"sci={attributes['science']} army={attributes['army']} "
            f"sup={attributes['support']} res={attributes['resources']}"
        )

    def _first_n_sentences(self, text: str, n: int) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(parts[:n]).strip()

    def _adjust_deltas(self, deltas: Dict[str, int]) -> Dict[str, int]:
        normalized = {k: int(v) for k, v in deltas.items()}
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

    def validate_deltas(self, option: Dict[str, Any]) -> bool:
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
        focus_hint: str | None = None,
    ) -> tuple[str, str]:
        stats = self._stats_compact(attributes)

        situation_prompt = (
            f"Stats: {stats}. Focus: {focus_hint or 'any local issue'}.\n"
            "2 short sentences: a local medieval governance problem (village/market/road/mill/guild scale).\n"
            "S1=what happened locally. S2=what the king must decide now.\n"
            "No epic scale, no wars, no prophecy."
        )

        situation_system_prompt = "2 plain sentences only. Simple, local, specific."
        raw_situation = self.gpt.generate(
            situation_system_prompt,
            situation_prompt,
            max_tokens=48,
            temperature=0.6,
            session=session,
        )

        raw_situation = raw_situation.strip()
        situation_part = raw_situation.split('\n\n')[0] if '\n\n' in raw_situation else raw_situation.split('\n')[0]
        situation = self._first_n_sentences(situation_part, 2)
        situation = self._truncate_at_sentence(situation, max_length=260)

        phrase_prompt = (
            f"Situation: {situation}\n"
            "Visitor's direct quote to the king, <=15 words. Only the quote."
        )

        phrase_system_prompt = "Quote text only."
        response = self.gpt.generate(
            phrase_system_prompt,
            phrase_prompt,
            max_tokens=30,
            temperature=1.3,
            session=session,
        )

        phrase = response.strip().strip('"').strip("'")
        phrase_part = phrase.split('\n\n')[0] if '\n\n' in phrase else phrase.split('\n')[0]
        phrase = self._truncate_at_sentence(phrase_part, max_length=150)

        if not situation:
            situation = "A visitor comes to the king with an important matter."
        if not phrase:
            phrase = "Your Majesty, I come with an important matter that requires your attention."

        return situation, phrase

    def generate_options(
        self,
        situation: str,
        phrase: str,
        attributes: Dict[str, int],
        session: List[Dict[str, str]] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        stats = self._stats_compact(attributes)

        prompt = (
            f"Stats: {stats}\nSituation: {situation}\nQuote: {phrase}\n"
            "2 king decisions. JSON only:\n"
            '{"option_1":{"description":"<4 words","science":int,"army":int,"support":int,"resources":int},'
            '"option_2":{...}}'
        )

        system_prompt = "Valid JSON only."
        response = self.gpt.generate(system_prompt, prompt, max_tokens=160, temperature=0.8, session=session)

        extracted = self._extract_first_json_object(response)
        data = json.loads(extracted if extracted else response)

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

    def generate_card(self, attributes: Dict[str, int], max_retries: int = 3) -> Dict[str, Any]:
        used_focuses: set[str] = set()
        for attempt in range(max_retries):
            try:
                if self.verbose and attempt > 0:
                    print(f"  Retry {attempt}/{max_retries-1}")

                session = self._new_card_session()
                focus = self._pick_focus(blocked_focuses=used_focuses)
                used_focuses.add(focus)

                situation, phrase = self.generate_situation(attributes, session=session, focus_hint=focus)

                if self._is_too_similar_to_recent(situation):
                    raise ValueError("situation too similar to recent cards")

                option_1, option_2 = self.generate_options(situation, phrase, attributes, session=session)

                card = {
                    "situation": situation,
                    "phrase": phrase,
                    "option_1": option_1,
                    "option_2": option_2
                }

                is_valid, message = self.validate_card(card)
                if is_valid:
                    self._recent_focuses.append(focus)
                    self._recent_situations.append(situation)
                    return card
                elif self.verbose:
                    print(f"  Validation failed: {message}")

            except Exception as e:
                if self.verbose:
                    print(f"  Error: {e}")

        raise Exception(f"Failed to generate valid card after {max_retries} attempts")

    def generate_cards(self, attributes: Dict[str, int], num_cards: int = 1) -> List[Dict[str, Any]]:
        if self.verbose:
            print(f"Generating {num_cards} card(s)...", end="", flush=True)

        cards = []
        for i in range(num_cards):
            card = self.generate_card(attributes)
            cards.append(card)
            if self.verbose:
                print(f" {i+1}", end="", flush=True)

        if self.verbose:
            print(" Done")
        return cards
