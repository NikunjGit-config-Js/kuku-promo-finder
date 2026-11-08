"""All tunable numbers and word lists live here, so nothing show-specific is hard-coded in logic.

The weights are design choices (not Kuku's), chosen to mirror the brief's three tests:
hook in the first 3 s, cliffhanger at the cut, makes sense on its own.
"""

# ---- Hard rules (from the brief) -------------------------------------------------
MIN_DUR, MAX_DUR = 20.0, 60.0          # promo length, seconds
HOOK_WINDOW = 3.0                      # hook must start within the first 3 s
EPS = 0.05                             # timestamp tolerance, seconds
DUR_TOL = 1e-6                         # float noise only: 20-60 s is a hard rule, not a +/-0.05 s one

# ---- Candidate eligibility (our choices) ------------------------------------------
MIN_DIALOGUE_LINES = 3                 # a promo a stranger can follow needs some dialogue
MAX_DEAD_TAIL = 8.0                    # seconds allowed after the last line before the cut
HOOK_MIN_INTENSITY = 0.35              # weakest opening event we still call a hook
NMS_OVERLAP = 0.35                     # temporal IoU above which two windows are "the same promo"
CANDIDATES_PER_EPISODE = 6             # sent to the LLM judge
PROMOS_PER_EPISODE = 2                 # kept in the final ranked list
MIN_FINAL_SCORE = 0.45                 # below this a promo is listed as rejected, not recommended
# The brief's three tests are an AND, not an average: a great cliffhanger must not buy back a missing hook.
# Re-anchored for the v5 score scale at ~30th-35th percentile of top candidates (unsupervised; no labels).
FLOORS_HEUR = {"hook": 0.45, "cliff": 0.45, "context": 0.35}
FLOORS_LLM = {"hook_score": 3, "cliffhanger_score": 3, "standalone_score": 3}

# ---- Clip-quality score weights (sum to 1). Editing convenience is NOT part of quality: an editor can
# fix a jump cut, not a weak story. It is reported separately (edit_effort) and only breaks ties.
W_HEUR = {"hook": 0.30, "cliff": 0.30, "context": 0.20, "escalation": 0.10, "intensity": 0.10}
W_LLM = {"hook": 0.25, "cliff": 0.25, "standalone": 0.25, "stakes": 0.15, "heuristic": 0.10}
EDIT_TIEBREAK = 0.02

# ---- Launch (first posting) ranking: an editorial policy, not an industry law. All values are settable.
# LAUNCH_WINDOW: clips from episodes <= this are "Green" by position. Default 10 is an assumption anchored on
# free-episode counts on four rival apps (typically 9-10; GASP via Geek Vibes Nation, 15 Oct 2026). Kuku TV's paywall is
# subscription-based, so set this per show (--launch-window); 0 disables the position rule (content only).
LAUNCH_WINDOW = 10
LAUNCH_BATCH = 10                      # size of the first posting batch
LAUNCH_MAX_AMBER = 2                   # teases from later episodes allowed in the batch; Red never
LAUNCH_MIN_PCTL = 50                   # launch picks must also be in the top half by quality (no filler)
AMBER_DISTANCE_PENALTY = 0.10          # later Amber clips rank lower (scaled by distance past the window)
ANGLE_REPEAT_PENALTY = 0.03            # soft push toward different story angles; no quotas

# ---- Lexicons (genre-level, never character names) ---------------------------------
# Delivery notes are the writer's own emotion tags, e.g. "(Outraged, Accusing)".
DELIVERY = {
    1.0: "shouting screaming furious enraged outraged livid seething raging venomous threatening menacing "
         "vicious savage murderous hysterical roaring thunderous explosive",
    0.85: "accusatory accusing accusation spiteful cruel sinister chilling taunting mocking scornful contemptuous "
          "sneering gloating horrified shocked gasping sobbing panicked terrified frantic betrayed devastated "
          "malicious dangerous lethal ruthless deadly fierce harsh brutal snarling hissing wrathful",
    0.7: "smug desperate pleading begging weeping heartbroken defiant demanding commanding cold icy dismissive "
         "condescending arrogant incredulous bitter urgent sharp steely intense stunned anguished tearful "
         "triumphant victorious crying choking broken raw passionate disgusted humiliated cutting biting",
    0.6: "husky intimate longing yearning seductive sultry aching smoldering flustered charged",   # romance tension
    0.5: "breathless trembling stern unyielding firm authoritative grave final shaken confused suspicious "
         "disbelieving dark ominous resolute determined proud haughty imperious tense guarded whispering "
         "low strained wounded hurt fearful nervous anxious shaky",
    0.25: "solemn sincere earnest hesitant vulnerable quiet soft gentle calm tender reassuring warm kind "
          "polite steady thoughtful curious relieved safe soothing loving affectionate playful amused light",
}
DELIVERY_DEFAULT = 0.35

# Dialogue words that signal stakes in short-form drama (betrayal, identity, danger, status).
LINE_STRONG = set("""
betray betrayed betrayal liar lie lied lying kill killed killing murder murdered dead die dying death poison
poisoned blood marry married marrying wedding divorce pregnant baby heir secret secrets truth confess prison
arrest arrested hang hanged gallows witch witchcraft burn burned ash nothing worthless never disgrace shame
ruin ruined destroy revenge duke duchess crown throne king queen prince princess fortune inheritance ring
stole steal thief slap hate monster beast coward traitor trap threat punish kneel beg enemy guilty curse
cursed exile banish banished disown disowned orphan bastard scandal affair stranger impostor fraud
ceo billionaire contract divorce papers paternity dna mistress heiress fired company custody
""".split())
RELATION_TERMS = set("""
husband wife sister brother father mother son daughter fiance fiancé bride groom family duke duchess lord lady
king queen prince princess maid servant master boss mistress stepmother stepsister uncle aunt widow heir
""".split())
CONTEXT_DEPENDENT = ["remember", "last night", "yesterday", "earlier", "as i said", "you promised", "like before"]

ACTION_STRONG = {
    0.95: "slap slaps slapped strikes struck punch punches shove shoves shoved stab stabs stabbed blood bloody "
          "bleeding collapses collapsed faints fainted chokes strangles shatters shattered smashes explosion "
          "gunshot scream screams screaming",
    0.75: "grabs grabbed seizes seized yanks throws hurls slams crashes falls kneels drops kisses kiss "
          "drags dragged restrains restrained shackles chains dagger knife blade sword pistol flames fire burning "
          "recoils flinches gasps sobs tears rips bursts kicks wound wounded scar scarred wrenches lunges",
    0.5: "trembling clutches clutching storms snaps glares freezes stunned widen widens tightens shaking "
         "reveals unveils",
}
SFX_STRONG = {
    0.95: "slap crack shatter shattering gunshot explosion scream splintering bone crash crashes",
    0.75: "thud bang slam slams violent scuffle thunder",
    0.5: "gasp gasps ominous cliffhanger sting dramatic swell swells",   # mood, not an event a stranger notices
}
CAMERA_DRAMATIC = ["snap zoom", "crash zoom", "whip pan", "extreme close-up", "extreme macro", "slow motion",
                   "dutch angle", "smash cut"]
TAIL_CLIFF_WORDS = ["ominous", "cliffhanger", "freezes", "frozen", "widen", "gasps", "stunned", "shock", "horror",
                    "sting", "cut to black", "pale", "drains", "thunderous", "tense"]
RESOLVED_DELIVERY = ["tender", "safe", "warm", "gentle", "soft", "relieved", "reassuring", "calm", "loving",
                     "content", "peaceful", "soothing"]

# ---- Story angles (for labelling and soft diversity only). Evidence = matched words.
ANGLES = {
    "betrayal": "betray betrayed betrayal lied liar cheat cheated traitor sold swore promised trusted",
    "humiliation": "embarrass embarrassed humiliate humiliated disgrace shame worthless nothing beneath kneel beg "
                   "laugh mock mocked rejected nobody beggar penniless disowned plain",
    "revenge": "regret revenge avenge punish power destroy ruin pay consequences judge court justice",
    "romance": "love loved kiss kissed heart touch hold beautiful mine embrace darling tender husky intimate "
               "passionate flustered longing yearning",
    "secret_identity": "stranger identity secret heir alive recognize disguise hidden impostor",
    "danger": "blood wound guards sword knife dagger poison fire chains prison gallows hang strike burn",
}
# Words that, in a clip from AFTER the launch window, suggest it confirms a later reveal (spoiler evidence).
STATUS_TERMS = ["duke", "duchess", "earl", "countess", "baron", "heir", "heiress", "king", "queen", "prince",
                "princess", "emperor", "empress", "crown", "ceo", "chairman", "chairwoman", "owner", "billionaire",
                "alpha", "luna"]
PAYOFF_PATTERNS = [
    r"\b(i am|i'm|he is|he's|she is|she's|you are|you're)\s+(the|your|his|her|my|their)\s+(\w+\s+)?({status})\b",
    r"\byou (poisoned|killed|murdered|framed|sold|stole|forged|tortured)\b",
    r"\b(declared dead|lawful (wife|husband)|rightful (heir|owner)|i now pronounce|proclaim\w*)\b",
    r"\b(arrested|sentenced|executed|confessed|stripped of)\b",
]
# A first line starting like this usually answers a question the viewer did not hear.
ANSWER_MARKERS = ("because", "yes", "no", "not", "so ", "and ", "but ", "then ", "that's", "that is", "exactly",
                  "of course")

# ---- LLM providers (OpenAI-compatible unless noted). Keys come from env vars only. ----
PROVIDERS = {
    "gemini":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash",
                  "key_env": "GEMINI_API_KEY"},
    "groq":      {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile",
                  "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free",
                   "key_env": "OPENROUTER_API_KEY"},
    "ollama":    {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b-instruct", "key_env": None},
    "openai":    {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "model": "claude-haiku-5-5",
                  "key_env": "ANTHROPIC_API_KEY"},
}
LLM_TIMEOUT = 90          # seconds per request
LLM_RETRIES = 3           # attempts per episode before falling back to heuristic scores
LLM_TEMPERATURE = 0.2
LLM_BACKOFF = 8            # seconds x attempt after HTTP 429/5xx
LLM_MIN_INTERVAL = 6.5    # seconds between calls (Gemini 2.5 Flash free tier: ~10 requests/min)
