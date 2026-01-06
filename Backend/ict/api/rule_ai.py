from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional

from nltk.stem import PorterStemmer

from .ml_services import (
    get_happy_tt,
    get_beam_settings,
    get_gec_pipeline,
    get_sentence_transformer,
    get_cross_encoder,
    get_keybert,
)

STEMMER = PorterStemmer()


class Processing:
    """
    Rule-based / AI-assisted grading orchestration.

    Uses:
    - GEC transformer (pipeline)
    - HappyTransformer grammar pass
    - SentenceTransformer cosine sim
    - CrossEncoder relevance sim
    - KeyBERT keyword extraction
    """

    STOPWORDS = {
        "a","about","above","after","again","against","all","am","an","and","any","are",
        "aren't","as","at","be","because","been","before","being","below","between","both",
        "but","by","can","cannot","could","couldn't","did","didn't","do","does","doesn't",
        "doing","don't","down","during","each","few","for","from","further","had","hadn't",
        "has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here",
        "here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll",
        "i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me",
        "more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only",
        "or","other","ought","our","ours","ourselves","out","over","own","same","shan't",
        "she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that",
        "that's","the","their","theirs","them","themselves","then","there","there's","these",
        "they","they'd","they'll","they're","they've","this","those","through","to","too",
        "under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've",
        "were","weren't","what","what's","when","when's","where","where's","which","while",
        "who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd",
        "you'll","you're","you've","your","yours","yourself","yourselves"
    }

    SIGNAL_INTERPRETATION = {
        "sentence_compare": {
            "low": {"range": (0, 30), "message": "The meaning does not strongly align with the expected answer."},
            "medium": {"range": (30, 55), "message": "The answer is somewhat related, but key ideas are missing or unclear."},
            "high": {"range": (55, 80), "message": "The answer captures most of the intended meaning."},
            "awesome": {"range": (80, 100), "message": "The answer matches the expected meaning very well."},
        },
        "keyword_finder": {
            "low": {"range": (0, 40), "message": "Key terms were mostly missing."},
            "medium": {"range": (40, 70), "message": "Some important keywords were used, but coverage is incomplete."},
            "high": {"range": (70, 90), "message": "Most of the important keywords were used correctly."},
            "awesome": {"range": (90, 101), "message": "Excellent keyword usage with strong coverage of key concepts."},
        },
        "topic_sent": {
            "low": {"range": (0, 30), "message": "The response does not stay focused on the main topic."},
            "medium": {"range": (30, 55), "message": "The response is on topic but lacks clarity or precision."},
            "high": {"range": (55, 80), "message": "The response stays mostly focused on the main topic."},
            "awesome": {"range": (80, 100), "message": "The response is clearly focused and directly addresses the topic."},
        },
        "relevant_topic": {
            "low": {"range": (0, 25), "message": "The answer is related but does not closely match the expected definition."},
            "medium": {"range": (25, 55), "message": "The answer partially matches the expected idea but is incomplete."},
            "high": {"range": (55, 80), "message": "The answer closely matches the expected idea."},
            "awesome": {"range": (80, 100), "message": "The answer is nearly identical in meaning to the expected definition."},
        },
    }

    def tokenized(self, user_answer: str) -> Dict[str, Any]:
        tokens = [t for t in (user_answer or "").lower().split() if t and t not in self.STOPWORDS]
        return {"list_format": tokens, "string_format": " ".join(tokens)}

    def remove_punctuation(self, text: str) -> str:
        pattern = "[" + re.escape(string.punctuation) + "]"
        return re.sub(pattern, "", text or "")

    def gec_process(self, sentence: str) -> str:
        gec = get_gec_pipeline()
        out = gec(sentence or "", max_length=128, clean_up_tokenization_spaces=True)
        return out[0]["generated_text"] if out else (sentence or "")

    def grammar_transformer(self, sentence: str) -> str:
        happy_tt = get_happy_tt()
        beam = get_beam_settings()
        return happy_tt.generate_text(sentence or "", args=beam).text

    def sentence_similarity_percent(self, user_answer: str, official_answer: str) -> int:
        util_model = get_sentence_transformer()
        from sentence_transformers import util

        emb_official = util_model.encode(official_answer or "", convert_to_tensor=True)
        emb_user = util_model.encode(user_answer or "", convert_to_tensor=True)
        score = util.cos_sim(emb_user, emb_official).item()
        return round(score * 100)

    def cross_encoder_similarity_percent(self, reference: str, student: str) -> int:
        ce = get_cross_encoder()
        score = float(ce.predict([(reference or "", student or "")])[0])
        return round(min(score / 5.0, 1.0) * 100)

    def extract_key_points(self, official_answer: str, top_n: int = 24) -> List[str]:
        kw = get_keybert()
        keywords = kw.extract_keywords(
            official_answer or "",
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
        )

        flattened: List[str] = []
        for phrase, _score in keywords:
            flattened.extend([w.strip() for w in phrase.split() if w.strip()])

        # unique + remove punctuation-only
        cleaned = []
        seen = set()
        for w in flattened:
            w2 = w.lower().strip(string.punctuation)
            if not w2 or w2 in seen:
                continue
            seen.add(w2)
            cleaned.append(w2)

        return cleaned

    def keyword_coverage_percent(self, key_points: List[str], user_tokens: List[str]) -> int:
        def norm(token: str) -> str:
            return (token or "").lower().strip(string.punctuation)

        normalized_user = {norm(t) for t in (user_tokens or []) if norm(t)}
        normalized_keys = {norm(k) for k in (key_points or []) if norm(k)}

        if not normalized_keys:
            return 0

        matched = normalized_user.intersection(normalized_keys)
        return round(len(matched) / len(normalized_keys) * 100)

    def interpret_signal(self, score: float, signal_name: str, model: str) -> Dict[str, Any]:
        for level, data in self.SIGNAL_INTERPRETATION[signal_name].items():
            low, high = data["range"]
            if low <= score < high:
                return {"model": model, "level": level, "message": data["message"]}
        # fallback
        return {"model": model, "level": "unknown", "message": "No interpretation available."}

    def grade(self, user_answer: str, official_answer: str) -> List[Dict[str, Any]]:
        # 1) Grammar normalization passes
        gec_text = self.gec_process(user_answer)
        norm_text = self.grammar_transformer(gec_text)

        # 2) Tokenization + keyword extraction
        tokens = self.tokenized(norm_text)
        key_points = self.extract_key_points(official_answer)

        # 3) Signals
        sentence_compare = self.sentence_similarity_percent(tokens["string_format"], official_answer)
        keyword_score = self.keyword_coverage_percent(key_points, tokens["list_format"])
        topic_sent = self.sentence_similarity_percent(norm_text, official_answer)
        relevant_topic = self.cross_encoder_similarity_percent(norm_text, official_answer)

        return [
            self.interpret_signal(sentence_compare, "sentence_compare", "Answer Comparison"),
            self.interpret_signal(keyword_score, "keyword_finder", "Terminologies"),
            self.interpret_signal(topic_sent, "topic_sent", "On Topic"),
            self.interpret_signal(relevant_topic, "relevant_topic", "Relevance"),
        ]