"""Editorial policy: spoiler tiers, launch batch rules, angles, reply detection. Pure unit tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promo_finder import config as C  # noqa: E402
from promo_finder.candidates import _is_reply  # noqa: E402
from promo_finder.editorial import launch_ranking, spoiler_tier, tag_angle  # noqa: E402
from promo_finder.parser import Event  # noqa: E402


def ev(text, speaker="A", kind="dialogue", start=0.0, end=1.0, delivery=None):
    return Event("E01-001", 1, 1, 0, kind, speaker or kind.upper(), speaker if kind == "dialogue" else None,
                 delivery, start, end, None, None, text, 1)


def test_green_inside_window_even_with_payoff_wording():
    t = spoiler_tier(3, [ev("I am the rightful heir!")], window=10, early=set())
    assert t["tier"] == "green" and "inside the launch window" in t["evidence"][0]


def test_red_after_window_needs_evidence():
    red = spoiler_tier(30, [ev("You poisoned my mother.")], window=10, early=set())
    assert red["tier"] == "red" and "payoff wording" in red["evidence"][0] and red["confidence"] == "low"
    novel = spoiler_tier(30, [ev("Kneel before the duke.")], window=10, early=set())
    assert novel["tier"] == "red" and '"duke"' in novel["evidence"][0]
    heard = spoiler_tier(30, [ev("Kneel before the duke.")], window=10, early={"duke"})
    assert heard["tier"] == "amber"                     # a term the viewer already heard is not a reveal


def test_llm_can_raise_but_never_lower_the_tier():
    assert spoiler_tier(30, [ev("Where are you going?")], 10, set(), llm_risk="high")["tier"] == "red"
    assert spoiler_tier(30, [ev("You poisoned her.")], 10, set(), llm_risk="low")["tier"] == "red"
    assert spoiler_tier(3, [ev("Hello.")], 10, set(), llm_risk="medium")["tier"] == "amber"


def test_window_zero_is_content_only():
    assert spoiler_tier(45, [ev("Where are you going?")], window=0, early=set())["tier"] == "green"
    assert spoiler_tier(2, [ev("You framed me!")], window=0, early=set())["tier"] == "red"


def _promo(ep, score, tier, angle="humiliation", edit=1.0):
    return {"episodes": [ep], "score": score, "angle": angle, "features": {"edit_effort": edit},
            "spoiler": {"tier": tier}}


def test_launch_batch_rules():
    promos = ([_promo(40, 0.99, "red"), _promo(41, 0.98, "red")]
              + [_promo(20 + i, 0.95 - i / 100, "amber") for i in range(5)]
              + [_promo(i, 0.80 - i / 100, "green") for i in range(1, 9)]
              + [_promo(1, 0.79, "green")])                                   # second clip from Ep 1
    batch = launch_ranking(promos, n_episodes=50, window=10)
    tiers = [p["spoiler"]["tier"] for p in batch]
    assert "red" not in tiers and tiers.count("amber") <= C.LAUNCH_MAX_AMBER
    eps = [p["episodes"][0] for p in batch]
    assert len(eps) == len(set(eps))                                          # one clip per episode
    bar = sorted(p["score"] for p in promos)[int(len(promos) * C.LAUNCH_MIN_PCTL / 100)]
    assert all(p["score"] >= bar for p in batch)                              # no filler from the bottom half
    assert [p["launch_rank"] for p in batch] == list(range(1, len(batch) + 1))


def test_angle_repeat_penalty_breaks_near_ties_only():
    promos = [_promo(1, 0.800, "green", "humiliation"), _promo(2, 0.799, "green", "humiliation"),
              _promo(3, 0.790, "green", "romance")] + [_promo(4 + i, 0.10, "green") for i in range(4)]
    batch = launch_ranking(promos, n_episodes=10, window=10)
    assert [p["episodes"][0] for p in batch][:2] == [1, 3]                    # romance overtakes a near-tie


def test_angle_comes_with_evidence():
    angle, evidence = tag_angle([ev("You lied to me, traitor!"), ev("I trusted you.")])
    assert angle == "betrayal" and {"lied", "traitor", "trusted"} <= set(evidence)
    assert tag_angle([ev("Pass the salt.")]) == ("conflict", [])


def test_reply_opening_detection():
    q = ev("Why are you wearing my ring?", "MAYA", start=10.0, end=12.0)
    a = ev("Because he gave it to me.", "VANESSA", start=12.3, end=14.0)
    assert _is_reply(a, q, s=12.2)                                  # answer to an unheard question
    assert not _is_reply(a, ev("Hello.", "X", start=1.0, end=2.0), s=12.2)   # 10 s gap: a fresh beat
    assert not _is_reply(a, ev("Line.", "VANESSA", start=10.0, end=12.0), s=12.2)   # same speaker


def test_reply_detector_reads_the_first_clause_regression():
    """v5.0 missed a short reaction followed by more words (whole-line length was > 4 words)."""
    stmt = ev("From today you are no son of mine.", "FATHER", start=10.0, end=12.0)
    reaction = ev("You really mean that? After everything I did for you!", "SON", start=12.6, end=15.0)
    assert _is_reply(reaction, stmt, s=12.5)
    fresh = ev("I have covered your night shifts for three years.", "SON", start=12.6, end=15.0)
    assert not _is_reply(fresh, stmt, s=12.5)                     # a self-contained statement is not a reply


def test_vocative_name_is_skipped_and_answer_words_match_whole_words():
    stmt = ev("Good evening, everyone.", "HOST", start=1.0, end=2.0)
    vocative = ev("Ethan? Why is she wearing my ring tonight?", "MAYA", start=2.5, end=5.0)
    assert not _is_reply(vocative, stmt, s=2.4, names={"ethan"})   # addressing someone is not reacting
    assert _is_reply(vocative, stmt, s=2.4, names=set())            # without the name list it looks like a reaction
    nothing = ev("Nothing in this hospital happens without my signature.", "MAYA", start=2.5, end=5.0)
    assert not _is_reply(nothing, stmt, s=2.4)                      # 'no' must not match 'nothing'
    no = ev("No, you will listen to me this time.", "MAYA", start=2.5, end=5.0)
    assert _is_reply(no, stmt, s=2.4)
