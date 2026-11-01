"""MovieFlix - a Netflix / Hotstar style movie discovery app.

Stack: Streamlit + TMDB API (+ optional OMDb for real IMDb ratings) + scikit-learn (TF-IDF recommender).
All navigation state lives in the URL (?view=movie&id=123), so every page is deep-linkable and the
browser Back button works.
"""
import html
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote_plus, urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    HAS_SKLEARN = True
except Exception:  # scikit-learn is optional; recommender tabs degrade gracefully
    HAS_SKLEARN = False

st.set_page_config(page_title="MovieFlix | Discover your next obsession", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------
TMDB = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p"
WATCHLIST_FILE = Path("watchlist.json")
REGIONS = {"IN": "India", "US": "United States", "GB": "United Kingdom", "CA": "Canada", "AU": "Australia", "AE": "UAE", "SG": "Singapore", "DE": "Germany", "FR": "France", "JP": "Japan"}
LANGS = {"Any language": "", "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te", "Malayalam": "ml", "Kannada": "kn", "Bengali": "bn", "Marathi": "mr", "Korean": "ko", "Japanese": "ja", "Spanish": "es", "French": "fr"}
SORTS = {"Most popular": "popularity.desc", "Highest rated": "vote_average.desc", "Newest first": "date.desc", "Box office": "revenue.desc"}
KINDS = ["Title", "Actor", "Director", "Genre"]
MOODS = [("🧠 Mind-bending", "878,9648"), ("🌈 Feel-good", "35,10751"), ("⚡ Edge of your seat", "53,28"), ("💜 Heartstrings", "10749,18"), ("🌑 Dark & mysterious", "27,9648"), ("🕵️ Crime files", "80"), ("🎨 Animated escapes", "16")]
GENRE_ALIASES = {"sci-fi": "science fiction", "scifi": "science fiction", "sci fi": "science fiction", "romcom": "romance", "rom-com": "romance", "scary": "horror", "cartoon": "animation", "anime": "animation"}


def _secret(name):
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return (value or os.getenv(name, "") or "").strip()


API_KEY = _secret("TMDB_API_KEY")
OMDB_KEY = _secret("OMDB_API_KEY")  # optional: enables real IMDb / Rotten Tomatoes ratings


def qp(name, default=""):
    value = st.query_params.get(name, default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return value or default


REGION = qp("region", "IN")
if REGION not in REGIONS:
    REGION = "IN"

# ----------------------------------------------------------------------------------------------
# HTTP + cache (thread-safe, so rows can be fetched in parallel)
# ----------------------------------------------------------------------------------------------
_CACHE, _LOCK, _HTTP = {}, threading.Lock(), requests.Session()


def _cached_json(cache_key, url_, params, ttl=3600):
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(cache_key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    try:
        response = _HTTP.get(url_, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return {}
    with _LOCK:
        if len(_CACHE) > 3000:
            _CACHE.clear()
        _CACHE[cache_key] = (now, data)
    return data


def T(path, params=None, ttl=3600):
    """TMDB GET. Returns {} on any failure (and does not cache failures)."""
    if not API_KEY:
        return {}
    q = dict(params or {})
    cache_key = ("tmdb", path, tuple(sorted((k, str(v)) for k, v in q.items())))
    q["api_key"] = API_KEY
    return _cached_json(cache_key, f"{TMDB}/{path.lstrip('/')}", q, ttl)


def fetch_many(jobs):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda j: T(j[0], j[1]), jobs))


def omdb_info(imdb_id):
    if not OMDB_KEY or not imdb_id:
        return {}
    return _cached_json(("omdb", imdb_id), "https://www.omdbapi.com/", {"i": imdb_id, "apikey": OMDB_KEY}, 86400)


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def esc(x):
    return html.escape(str(x if x is not None else ""), quote=True)


def clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def img(path, size="w342"):
    return f"{IMG}/{size}{path}" if path else ""


def year_of(it):
    d = it.get("release_date") or it.get("first_air_date") or ""
    return d[:4]


def title_of(it):
    return it.get("title") or it.get("name") or it.get("original_title") or it.get("original_name") or "Untitled"


def media_of(it, default="movie"):
    m = it.get("media_type")
    return m if m in ("movie", "tv") else default


def url(**kw):
    q = {"region": REGION}
    q.update({k: v for k, v in kw.items() if v not in (None, "")})
    return html.escape("?" + urlencode(q))


def merged_url(**kw):
    q = {k: (v if isinstance(v, str) else v[0]) for k, v in st.query_params.items()}
    q.update({k: str(v) for k, v in kw.items()})
    return html.escape("?" + urlencode(q))


def load_watchlist():
    try:
        return json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_watchlist(items):
    try:
        WATCHLIST_FILE.write_text(json.dumps(items), encoding="utf-8")
    except OSError:
        pass


def provider_link(name, title, fallback):
    n, q = name.lower(), quote_plus(title)
    if "netflix" in n:
        return f"https://www.netflix.com/search?q={q}"
    if "prime" in n or "amazon" in n:
        return f"https://www.primevideo.com/search/?phrase={q}"
    if "youtube" in n:
        return f"https://www.youtube.com/results?search_query={q}"
    return fallback


# ----------------------------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------------------------
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#07060d;--line:rgba(167,139,250,.22);--mute:#a9a3c6}
header[data-testid="stHeader"],[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important}
.stApp{background:radial-gradient(ellipse at 10% -10%,#2a1257 0,transparent 38%),radial-gradient(ellipse at 100% 8%,#0e2c57 0,transparent 30%),var(--bg);color:#f7f4ff;font-family:'DM Sans',sans-serif}
.block-container,[data-testid="stMainBlockContainer"]{max-width:none!important;padding:0 4vw 3rem!important}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important}
.nav{position:fixed;top:0;left:0;right:0;height:66px;z-index:1000;display:flex;align-items:center;gap:28px;padding:0 4vw;background:linear-gradient(180deg,rgba(7,6,13,.94),rgba(7,6,13,.55));backdrop-filter:blur(14px);border-bottom:1px solid rgba(167,139,250,.12)}
.logo{font-family:'Bebas Neue',sans-serif;font-size:2.1rem;letter-spacing:3px;background:linear-gradient(100deg,#fff 5%,#d9b4ff 45%,#8b5cf6 75%,#38bdf8);-webkit-background-clip:text;background-clip:text;color:transparent!important;text-decoration:none!important}
.logo i{-webkit-text-fill-color:#f43f5e;font-style:normal}
.links{display:flex;gap:6px;flex:1;overflow-x:auto;scrollbar-width:none}
.links a{color:#cfc9ea!important;text-decoration:none!important;font-weight:600;font-size:.92rem;padding:8px 14px;border-radius:999px;transition:.2s;white-space:nowrap}
.links a:hover{background:rgba(139,92,246,.18);color:#fff!important}
.links a.on{background:linear-gradient(100deg,#6d28d9,#4f46e5);color:#fff!important}
.region{position:relative}
.region summary{list-style:none;cursor:pointer;padding:8px 14px;border:1px solid var(--line);border-radius:999px;font-size:.85rem;font-weight:600;color:#ddd6fe;white-space:nowrap}
.region summary::-webkit-details-marker{display:none}
.rmenu{position:absolute;right:0;top:46px;background:#120e24;border:1px solid var(--line);border-radius:14px;padding:8px;min-width:200px;box-shadow:0 20px 50px rgba(0,0,0,.6)}
.rmenu a{display:block;padding:8px 12px;border-radius:9px;color:#d8d2f2!important;text-decoration:none!important;font-size:.88rem}
.rmenu a:hover{background:rgba(139,92,246,.2)}
.spacer{height:92px}
.hero{position:relative;height:84vh;min-height:540px;max-height:780px;margin:0 -4vw 8px;overflow:hidden}
.slide{position:absolute;inset:0;background-size:cover;background-position:center 18%;opacity:0;visibility:hidden}
.slide::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(7,6,13,.96) 0%,rgba(7,6,13,.7) 34%,rgba(7,6,13,.08) 72%),linear-gradient(0deg,#07060d 2%,transparent 40%),linear-gradient(180deg,rgba(7,6,13,.75),transparent 24%)}
.hc{position:absolute;left:4vw;bottom:15%;max-width:640px}
.hc h1{font-family:'Bebas Neue',sans-serif!important;font-size:clamp(2.8rem,6.4vw,5.6rem);letter-spacing:2px;line-height:.95;margin:.3rem 0 .6rem;text-shadow:0 6px 40px rgba(0,0,0,.6)}
.hc p{color:#d9d4ee;font-size:1.02rem;line-height:1.55;margin:.4rem 0 1.1rem}
.eyebrow{color:#c4b5fd;text-transform:uppercase;letter-spacing:2.4px;font-size:.74rem;font-weight:700}
.meta{display:flex;gap:10px;align-items:center;color:#cfc9ea;font-weight:600;font-size:.92rem;flex-wrap:wrap}
.star{color:#fde68a}
.btns{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 24px;border-radius:12px;font-weight:700;font-size:.95rem;text-decoration:none!important;transition:.2s;color:#fff!important;background:linear-gradient(100deg,#7c3aed,#4f46e5);border:1px solid rgba(196,181,253,.5)}
.btn:hover{transform:translateY(-2px);box-shadow:0 0 26px rgba(139,92,246,.5)}
.btn.ghost{background:rgba(255,255,255,.1);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.22)}
.btn.red{background:linear-gradient(100deg,#e11d48,#f43f5e);border-color:rgba(253,164,175,.6)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
.chip{display:inline-block;padding:7px 15px;border-radius:999px;background:rgba(139,92,246,.14);border:1px solid var(--line);color:#e6defc!important;font-size:.86rem;font-weight:600;text-decoration:none!important;transition:.2s}
a.chip:hover{background:rgba(139,92,246,.35);transform:translateY(-2px)}
.chip.k{background:rgba(255,255,255,.06)}
.rowwrap{margin:22px 0 4px}
.rowhead{display:flex;align-items:baseline;justify-content:space-between}
.rowhead h3{font-size:1.3rem;margin:0;color:#f4f1ff}
.rowhead a{color:#c4b5fd!important;font-size:.85rem;text-decoration:none!important;font-weight:600}
.row{display:flex;gap:14px;overflow-x:auto;padding:16px 6px 26px;scroll-snap-type:x proximity;scrollbar-width:thin;scrollbar-color:#3b2d63 transparent}
.row .card,.row .pcard{flex:0 0 168px;scroll-snap-align:start}
.row .pcard{flex-basis:132px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:22px 16px;margin:14px 0}
.card{display:block;text-decoration:none!important;color:inherit!important;transition:transform .25s cubic-bezier(.2,.8,.2,1)}
.card:hover{transform:translateY(-6px) scale(1.05);z-index:5}
.poster{position:relative;aspect-ratio:2/3;border-radius:14px;overflow:hidden;background:#15112a;border:1px solid rgba(167,139,250,.14);box-shadow:0 10px 26px rgba(0,0,0,.45);transition:.25s}
.card:hover .poster{box-shadow:0 18px 44px rgba(139,92,246,.5);border-color:rgba(196,181,253,.65)}
.poster img{width:100%;height:100%;object-fit:cover;display:block}
.noposter{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:10px;text-align:center;color:#9d97ba;font-size:.85rem}
.badge{position:absolute;top:8px;left:8px;background:rgba(7,6,13,.8);backdrop-filter:blur(6px);color:#fde68a;font-size:.74rem;font-weight:700;padding:3px 8px;border-radius:999px}
.match{position:absolute;top:8px;right:8px;background:linear-gradient(100deg,#10b981,#38bdf8);color:#04121a;font-size:.72rem;font-weight:800;padding:3px 8px;border-radius:999px}
.ov{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:flex-end;padding:12px;background:linear-gradient(0deg,rgba(7,6,13,.97) 14%,rgba(7,6,13,.62) 58%,transparent);opacity:0;transition:.25s}
.card:hover .ov{opacity:1}
.ot{font-weight:700;font-size:.95rem;line-height:1.2}
.om{color:#c4b5fd;font-size:.76rem;margin:3px 0 6px;font-weight:600}
.od{color:#cfc9ea;font-size:.74rem;line-height:1.4}
.oc{margin-top:8px;font-size:.78rem;font-weight:700;color:#fff}
.cap{margin-top:8px;font-size:.84rem;font-weight:600;color:#e9e5fb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pcard{display:block;text-decoration:none!important;color:#f1edff!important;text-align:center;transition:.25s}
.pcard:hover{transform:translateY(-5px)}
.pcard img,.noface{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:50%;border:2px solid rgba(167,139,250,.3);background:#15112a;display:flex;align-items:center;justify-content:center;font-size:2.4rem;transition:.25s}
.pcard:hover img{border-color:#c4b5fd;box-shadow:0 10px 30px rgba(139,92,246,.45)}
.pcard b{display:block;margin-top:10px;font-size:.88rem;line-height:1.2}
.pcard small{display:block;color:var(--mute);font-size:.76rem;margin-top:3px}
.dhero{position:relative;margin:0 -4vw 10px;padding:112px 4vw 44px;background-size:cover;background-position:center top}
.dhero::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(7,6,13,.97) 10%,rgba(7,6,13,.78) 55%,rgba(7,6,13,.45)),linear-gradient(0deg,#07060d 3%,transparent 45%)}
.dh-in{position:relative;display:flex;gap:40px;flex-wrap:wrap;align-items:flex-start}
.dposter{width:290px;max-width:100%;border-radius:18px;border:1px solid rgba(196,181,253,.35);box-shadow:0 30px 70px rgba(0,0,0,.7),0 0 60px rgba(124,58,237,.25)}
.dinfo{flex:1;min-width:300px;max-width:820px}
.dinfo h1{font-family:'Bebas Neue',sans-serif!important;font-size:clamp(2.4rem,5vw,4.2rem);letter-spacing:1.5px;line-height:1;margin:.25rem 0 .3rem}
.dinfo h1 span{color:#a78bfa}
.tag{color:#c4b5fd;font-style:italic;margin-bottom:6px}
.rates{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:12px 0}
.rt{font-size:.92rem;color:#cfc9ea}
.rt b{font-size:1.3rem;color:#fff}
.ring{--p:70;width:62px;height:62px;border-radius:50%;background:conic-gradient(#a78bfa calc(var(--p)*1%),rgba(255,255,255,.14) 0);display:grid;place-items:center;position:relative}
.ring::before{content:"";position:absolute;inset:6px;border-radius:50%;background:#0d0a1b}
.ring span{position:relative;font-weight:700;font-size:.9rem}
.score{padding:8px 14px;border-radius:12px;background:rgba(255,255,255,.07);border:1px solid var(--line);font-size:.82rem;color:#cfc9ea}
.score b{display:block;font-size:1.05rem;color:#fff}
.score.imdb b{color:#facc15}
.ov2{color:#ddd8f1;line-height:1.7;font-size:1.02rem;margin:10px 0 14px}
.crew{color:#cfc9ea;font-size:.95rem;margin:4px 0}
.crew b{color:#fff}
.sec{margin:26px 0 4px;font-size:1.45rem;font-weight:700}
.sec small{color:var(--mute);font-weight:500;font-size:.85rem;margin-left:8px}
.provgrp{margin:10px 0 4px;color:#c4b5fd;font-size:.74rem;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.prov{display:inline-flex;align-items:center;gap:10px;padding:8px 16px 8px 8px;margin:4px 8px 4px 0;border-radius:14px;background:rgba(23,18,42,.9);border:1px solid #30254c;color:#f1edff!important;text-decoration:none!important;font-weight:600;font-size:.9rem;transition:.2s}
.prov:hover{border-color:#c4b5fd;transform:translateY(-2px);box-shadow:0 8px 24px rgba(139,92,246,.35)}
.prov img{width:38px;height:38px;border-radius:10px}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin:12px 0}
.fact{padding:14px 16px;border-radius:14px;background:rgba(22,18,42,.7);border:1px solid var(--line)}
.fact small{display:block;color:var(--mute);font-size:.74rem;text-transform:uppercase;letter-spacing:1.4px;margin-bottom:4px}
.fact b{font-size:.95rem}
.panel{border:1px solid var(--line);background:linear-gradient(135deg,rgba(30,21,54,.84),rgba(12,15,32,.88));border-radius:20px;padding:22px 26px;margin:10px 0 18px}
.panel h2{margin:.3rem 0}
.note{color:#9189ab;font-size:.8rem}
.pager{display:flex;gap:14px;align-items:center;justify-content:center;margin:20px 0}
.pager a{padding:9px 20px;border-radius:12px;border:1px solid var(--line);color:#e6defc!important;text-decoration:none!important;font-weight:600}
.pager a:hover{background:rgba(139,92,246,.25)}
.pager span{color:var(--mute)}
.bio{color:#ddd8f1;line-height:1.7;max-width:820px}
[data-testid="stForm"]{border:1px solid var(--line)!important;border-radius:18px!important;background:rgba(20,17,38,.6)!important;padding:18px!important}
[data-testid="stTextInput"] input,[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div,[data-testid="stTextArea"] textarea{background:#141126!important;border-color:#30254c!important;border-radius:12px!important}
.stButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:12px;border:1px solid rgba(167,139,250,.5);background:linear-gradient(100deg,#6d28d9,#4f46e5);color:#fff;font-weight:700;transition:.18s}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{border-color:#c4b5fd;box-shadow:0 0 22px rgba(139,92,246,.4);color:#fff;transform:translateY(-1px)}
hr{border-color:rgba(167,139,250,.18)!important}
@media(max-width:760px){.nav{gap:10px}.logo{font-size:1.6rem}.hc{bottom:10%}.dposter{width:200px}.row .card{flex-basis:140px}}
</style>"""


# ----------------------------------------------------------------------------------------------
# HTML builders
# ----------------------------------------------------------------------------------------------
def nav_html(view):
    items = [("home", "Home"), ("browse", "Browse"), ("search", "Search"), ("foryou", "✨ For You"), ("mylist", "My List")]
    links = "".join(f'<a class="{"on" if view == v else ""}" href="{url(view=v)}" target="_self">{label}</a>' for v, label in items)
    regions = "".join(f'<a href="{merged_url(region=c)}" target="_self">{c} · {n}</a>' for c, n in REGIONS.items())
    return (f'<div class="nav"><a class="logo" href="{url(view="home")}" target="_self">MOVIEFLIX<i>.</i></a><div class="links">{links}</div>'
            f'<details class="region"><summary>🌍 {REGION}</summary><div class="rmenu">{regions}</div></details></div>')


def card_html(it, default_type="movie", badge=None):
    if not it.get("id"):
        return ""
    mt = media_of(it, default_type)
    t = esc(title_of(it))
    rating = it.get("vote_average") or 0
    poster = img(it.get("poster_path"))
    pic = f'<img src="{poster}" alt="{t}" loading="lazy">' if poster else f'<div class="noposter">🎬<span>{t}</span></div>'
    rb = f'<span class="badge">★ {rating:.1f}</span>' if rating else ""
    mb = f'<span class="match">{esc(badge)}</span>' if badge else ""
    ov = esc(clip(it.get("overview"), 115))
    kind = "Series" if mt == "tv" else "Movie"
    return (f'<a class="card" href="{url(view="movie", id=it["id"], type=mt)}" target="_self"><div class="poster">{pic}{rb}{mb}'
            f'<div class="ov"><div class="ot">{t}</div><div class="om">{year_of(it) or "—"} · {kind}</div><div class="od">{ov}</div><div class="oc">▶ View details</div></div></div>'
            f'<div class="cap">{t}</div></a>')


def grid_html(items, mt="movie"):
    return '<div class="grid">' + "".join(card_html(i, mt) for i in items if i.get("id")) + "</div>"


def row_html(title, items, mt="movie", more=None):
    items = [i for i in items if i.get("poster_path")]
    if not items:
        return ""
    more_html = f'<a href="{more}" target="_self">Explore all ›</a>' if more else ""
    cards = "".join(card_html(i, mt) for i in items[:20])
    return f'<div class="rowwrap"><div class="rowhead"><h3>{title}</h3>{more_html}</div><div class="row">{cards}</div></div>'


def pcard_html(p):
    pic = img(p.get("profile_path"), "w185")
    im = f'<img src="{pic}" alt="{esc(p.get("name"))}" loading="lazy">' if pic else '<div class="noface">👤</div>'
    sub = esc(clip(p.get("character") or p.get("known_for_department") or "", 34))
    return f'<a class="pcard" href="{url(view="person", id=p["id"])}" target="_self">{im}<b>{esc(p.get("name"))}</b><small>{sub}</small></a>'


def hero_html(items):
    slides = [i for i in items if i.get("backdrop_path")][:5]
    if not slides:
        return ""
    n = len(slides)
    dur = 7 * n
    vis = 100 / n
    kf = (f"@keyframes heroFade{{0%{{opacity:0;visibility:hidden}}2%{{opacity:1;visibility:visible}}{max(vis - 2, 3):.2f}%{{opacity:1;visibility:visible}}"
          f"{vis + 2:.2f}%{{opacity:0;visibility:hidden}}100%{{opacity:0;visibility:hidden}}}}")
    parts = []
    for i, it in enumerate(slides):
        mt = media_of(it, "movie")
        anim = "opacity:1;visibility:visible" if n == 1 else f"animation:heroFade {dur}s linear {i * 7}s infinite both"
        rating = it.get("vote_average") or 0
        parts.append(
            f'<div class="slide" style="background-image:url({img(it["backdrop_path"], "w1280")});{anim}"><div class="hc">'
            f'<div class="eyebrow">#{i + 1} trending this week</div><h1>{esc(title_of(it))}</h1>'
            f'<div class="meta"><span class="star">★ {rating:.1f}</span><span>{year_of(it)}</span><span>{"Series" if mt == "tv" else "Movie"}</span></div>'
            f'<p>{esc(clip(it.get("overview"), 230))}</p><div class="btns">'
            f'<a class="btn" href="{url(view="movie", id=it["id"], type=mt, t=1)}" target="_self">▶ Watch trailer</a>'
            f'<a class="btn ghost" href="{url(view="movie", id=it["id"], type=mt)}" target="_self">ⓘ More info</a></div></div></div>')
    return f"<style>{kf}</style><div class=\"hero\">{''.join(parts)}</div>"


def chips_html(pairs):
    return '<div class="chips">' + "".join(f'<a class="chip" href="{href}" target="_self">{esc(label)}</a>' for label, href in pairs) + "</div>"


def spacer():
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------
def home_rows():
    base = {"sort_by": "popularity.desc", "vote_count.gte": 200, "include_adult": "false", "language": "en-US"}

    def genre(gid, label, emoji):
        return (f"{emoji} {label}", "discover/movie", {**base, "with_genres": gid}, "movie", url(view="browse", genres=gid))

    return [
        ("🔥 Trending this week", "trending/movie/week", {}, "movie", None),
        ("🎟️ In theatres now", "movie/now_playing", {"region": REGION, "language": "en-US"}, "movie", None),
        ("🇮🇳 Bollywood blockbusters", "discover/movie", {**base, "with_original_language": "hi", "vote_count.gte": 120}, "movie", url(view="browse", lang="hi")),
        ("🌶️ South Indian smashers", "discover/movie", {**base, "with_original_language": "te|ta|ml|kn", "vote_count.gte": 120}, "movie", None),
        ("📺 Binge-worthy series", "trending/tv/week", {}, "tv", url(view="browse", type="tv")),
        ("⭐ All-time top rated", "movie/top_rated", {}, "movie", url(view="browse", sort="Highest rated", rating="8", votes="2000")),
        genre(28, "Action & adrenaline", "💥"),
        genre(878, "Sci-Fi worlds", "🚀"),
        genre(35, "Laugh out loud", "😂"),
        genre(53, "Edge-of-seat thrillers", "🔪"),
        genre(27, "Horror nights", "👻"),
        genre(10749, "Romance", "💞"),
        genre(16, "Animation", "🎨"),
    ]


def render_home():
    rows = home_rows()
    data = fetch_many([(r[1], r[2]) for r in rows])
    st.markdown(hero_html(data[0].get("results", [])) or '<div class="spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="rowhead" style="margin-top:6px"><h3>Pick a vibe</h3></div>', unsafe_allow_html=True)
    st.markdown(chips_html([(label, url(view="browse", genres=ids)) for label, ids in MOODS]), unsafe_allow_html=True)
    for (title, _p, _q, mt, more), result in zip(rows, data):
        block = row_html(title, result.get("results", []), mt, more)
        if block:
            st.markdown(block, unsafe_allow_html=True)


def render_search():
    spacer()
    q, kind, minr = qp("q"), qp("kind", "Title"), float(qp("minr", "0") or 0)
    kind = kind if kind in KINDS else "Title"
    st.markdown('<div class="eyebrow">Search</div><h2 style="margin:.2rem 0 .6rem">Find any movie, series, actor or director</h2>', unsafe_allow_html=True)
    rating_opts = [0, 5, 6, 7, 8, 9]
    with st.form("search_form"):
        c1, c2, c3, c4 = st.columns([4, 1.4, 1.4, 1.1])
        text = c1.text_input("Search", value=q, placeholder="e.g. Interstellar, Shah Rukh Khan, Nolan, thriller", label_visibility="collapsed")
        kind_in = c2.selectbox("Search by", KINDS, index=KINDS.index(kind), label_visibility="collapsed")
        minr_in = c3.selectbox("Min rating", rating_opts, index=rating_opts.index(int(minr)) if int(minr) in rating_opts else 0, format_func=lambda x: "Any rating" if x == 0 else f"★ {x}+", label_visibility="collapsed")
        go = c4.form_submit_button("Search")
    if go:
        st.query_params.update({"view": "search", "q": text.strip(), "kind": kind_in, "minr": str(minr_in)})
        st.rerun()
    if not q:
        st.markdown(chips_html([("Christopher Nolan", url(view="search", q="Christopher Nolan", kind="Director")), ("Shah Rukh Khan", url(view="search", q="Shah Rukh Khan", kind="Actor")),
                                ("Thriller", url(view="search", q="Thriller", kind="Genre")), ("Dune", url(view="search", q="Dune", kind="Title")),
                                ("Anime", url(view="search", q="Anime", kind="Genre")), ("Rajkumar Hirani", url(view="search", q="Rajkumar Hirani", kind="Director"))]), unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="eyebrow">Pro tip</div>Search an actor or director to explore their filmography. Open any title for cast photos, trailer and where to stream it. Use <b>Browse</b> for rating, year, language and genre filters.</div>', unsafe_allow_html=True)
        return
    items, people = [], []
    if kind == "Title":
        for pg in (1, 2):
            res = T("search/multi", {"query": q, "page": pg, "include_adult": "false", "language": "en-US"}).get("results", [])
            items += [x for x in res if x.get("media_type") in ("movie", "tv")]
    elif kind in ("Actor", "Director"):
        people = T("search/person", {"query": q, "include_adult": "false", "language": "en-US"}).get("results", [])
        dept = "Acting" if kind == "Actor" else "Directing"
        people.sort(key=lambda p: (p.get("known_for_department") != dept, -(p.get("popularity") or 0)))
        if people:
            top = people[0]
            credits = T(f"person/{top['id']}/combined_credits", {"language": "en-US"})
            raw = credits.get("cast", []) if kind == "Actor" else [x for x in credits.get("crew", []) if x.get("job") == "Director"]
            uniq = {}
            for x in raw:
                if x.get("poster_path") and x.get("id") and x.get("media_type") in ("movie", "tv"):
                    uniq[(x["media_type"], x["id"])] = x
            items = sorted(uniq.values(), key=lambda x: x.get("popularity", 0), reverse=True)[:60]
    else:
        movie_genres = T("genre/movie/list", {"language": "en-US"}).get("genres", [])
        needle = GENRE_ALIASES.get(q.lower(), q.lower())
        match = next((g for g in movie_genres if needle in g["name"].lower() or g["name"].lower() in needle), None)
        if not match:
            st.info("No genre matched. Try: " + ", ".join(g["name"] for g in movie_genres))
            return
        for pg in (1, 2):
            items += T("discover/movie", {"with_genres": match["id"], "sort_by": "popularity.desc", "vote_count.gte": 100, "page": pg, "language": "en-US"}).get("results", [])
    items = [x for x in items if (x.get("vote_average") or 0) >= minr]
    if people:
        st.markdown(f'<div class="sec">People<small>top matches for “{esc(q)}”</small></div><div class="row">{"".join(pcard_html(p) for p in people[:8])}</div>', unsafe_allow_html=True)
    if items:
        label = {"Actor": "Filmography", "Director": "Directed"}.get(kind, "Results")
        st.markdown(f'<div class="sec">{label}<small>{len(items)} titles · “{esc(q)}”</small></div>', unsafe_allow_html=True)
        st.markdown(grid_html(items), unsafe_allow_html=True)
    else:
        st.info("No results found. Try a shorter name, another search type, or lower the minimum rating.")


def render_browse():
    spacer()
    mt = qp("type", "movie")
    mt = mt if mt in ("movie", "tv") else "movie"
    genres_list = T(f"genre/{mt}/list", {"language": "en-US"}).get("genres", [])
    name_by_id = {str(g["id"]): g["name"] for g in genres_list}
    id_by_name = {v: k for k, v in name_by_id.items()}
    sel_ids = [x for x in qp("genres").split(",") if x in name_by_id]
    lang = qp("lang")
    sort_name = qp("sort", "Most popular")
    sort_name = sort_name if sort_name in SORTS else "Most popular"
    y1, y2 = int(qp("y1", "1990") or 1990), int(qp("y2", "2027") or 2027)
    rating = float(qp("rating", "6") or 0)
    vote_opts = [0, 25, 100, 500, 2000]
    votes = int(qp("votes", "100") or 100)
    page = max(1, int(qp("p", "1") or 1))
    st.markdown('<div class="eyebrow">Browse</div><h2 style="margin:.2rem 0 .6rem">Filter your way to the perfect watch</h2>', unsafe_allow_html=True)
    with st.form("browse_form"):
        r1 = st.columns([1.1, 2.6, 1.3, 1.3])
        mt_in = r1[0].radio("Type", ["Movies", "Series"], index=0 if mt == "movie" else 1, horizontal=True)
        g_in = r1[1].multiselect("Genres (any of)", list(name_by_id.values()), default=[name_by_id[i] for i in sel_ids])
        lang_codes = list(LANGS.values())
        lang_in = r1[2].selectbox("Language", list(LANGS), index=lang_codes.index(lang) if lang in lang_codes else 0)
        sort_in = r1[3].selectbox("Sort by", list(SORTS), index=list(SORTS).index(sort_name))
        r2 = st.columns([2, 2, 1])
        years = r2[0].slider("Release year", 1950, 2027, (max(1950, min(y1, 2027)), max(1950, min(y2, 2027))))
        rating_in = r2[1].slider("Minimum rating (TMDB audience score, IMDb-style /10)", 0.0, 10.0, rating, 0.5)
        votes_in = r2[2].selectbox("Min votes", vote_opts, index=vote_opts.index(votes) if votes in vote_opts else 2)
        go = st.form_submit_button("Apply filters")
    if go:
        new_mt = "movie" if mt_in == "Movies" else "tv"
        ids = ",".join(id_by_name[n] for n in g_in) if new_mt == mt else ""
        st.query_params.update({"view": "browse", "type": new_mt, "genres": ids, "lang": LANGS[lang_in], "sort": sort_in, "y1": str(years[0]), "y2": str(years[1]),
                                "rating": str(rating_in), "votes": str(votes_in), "p": "1"})
        st.rerun()
    sort_val = SORTS[sort_name]
    if sort_val == "date.desc":
        sort_val = "primary_release_date.desc" if mt == "movie" else "first_air_date.desc"
    if sort_val == "revenue.desc" and mt == "tv":
        sort_val = "popularity.desc"
    dkey = "primary_release_date" if mt == "movie" else "first_air_date"
    params = {"language": "en-US", "include_adult": "false", "sort_by": sort_val, "vote_average.gte": rating, "vote_count.gte": votes, "page": page,
              f"{dkey}.gte": f"{y1}-01-01", f"{dkey}.lte": f"{y2}-12-31"}
    if sel_ids:
        params["with_genres"] = "|".join(sel_ids)
    if lang:
        params["with_original_language"] = lang
    data = T(f"discover/{mt}", params)
    results = data.get("results", [])
    total_pages = min(int(data.get("total_pages", 1) or 1), 500)
    if not results:
        st.info("Nothing matches these filters. Try lowering the rating or widening the years.")
        return
    shown = ", ".join(name_by_id[i] for i in sel_ids) or "All genres"
    st.markdown(f'<div class="sec">{int(data.get("total_results", 0)):,} titles<small>{esc(shown)} · ★ {rating:g}+ · {y1}–{y2}</small></div>', unsafe_allow_html=True)
    st.markdown(grid_html(results, mt), unsafe_allow_html=True)
    prev_l = f'<a href="{merged_url(p=page - 1)}" target="_self">‹ Previous</a>' if page > 1 else ""
    next_l = f'<a href="{merged_url(p=page + 1)}" target="_self">Next ›</a>' if page < total_pages else ""
    st.markdown(f'<div class="pager">{prev_l}<span>Page {page} of {total_pages}</span>{next_l}</div>', unsafe_allow_html=True)


def pick_videos(d):
    vids = [v for v in (d.get("videos") or {}).get("results", []) if v.get("site") == "YouTube" and v.get("key")]
    rank = {"Trailer": 0, "Teaser": 1, "Clip": 2, "Featurette": 3}
    vids.sort(key=lambda v: (rank.get(v.get("type"), 9), not v.get("official")))
    return vids


def fmt_money(v):
    return f"${v:,.0f}" if v else ""


def render_movie():
    spacer()
    try:
        mid = int(qp("id", "0"))
    except ValueError:
        mid = 0
    mt = qp("type", "movie")
    mt = mt if mt in ("movie", "tv") else "movie"
    d = T(f"{mt}/{mid}", {"append_to_response": "videos,credits,recommendations,similar,external_ids,watch/providers", "include_video_language": "en,hi,null", "language": "en-US"}) if mid else {}
    if not d.get("id"):
        st.error("Couldn't load this title. Check your TMDB key and internet connection.")
        return
    title = title_of(d)
    year = year_of(d)
    imdb_id = (d.get("external_ids") or {}).get("imdb_id") or d.get("imdb_id")
    om = omdb_info(imdb_id)
    vids = pick_videos(d)
    trailer_url = f"https://www.youtube.com/watch?v={vids[0]['key']}" if vids else f"https://www.youtube.com/results?search_query={quote_plus(title + ' ' + year + ' official trailer')}"
    runtime = d.get("runtime") or (d.get("episode_run_time") or [None])[0]
    meta = [f'<span class="chip k">{x}</span>' for x in [year, f"{runtime} min" if runtime else "", f"{d['number_of_seasons']} season(s)" if d.get("number_of_seasons") else "", "Series" if mt == "tv" else "Movie"] if x]
    genres = "".join(f'<a class="chip" href="{url(view="browse", type=mt, genres=g["id"])}" target="_self">{esc(g["name"])}</a>' for g in d.get("genres", []))
    score = d.get("vote_average") or 0
    pct = int(round(score * 10))
    rates = f'<div class="ring" style="--p:{pct}"><span>{pct}%</span></div><div class="rt"><b>{score:.1f}</b>/10 · {int(d.get("vote_count") or 0):,} votes</div>'
    if om.get("imdbRating") and om.get("imdbRating") != "N/A":
        rates += f'<div class="score imdb"><small>IMDb</small><b>★ {esc(om["imdbRating"])}</b></div>'
    for r in om.get("Ratings", []) if isinstance(om.get("Ratings"), list) else []:
        if r.get("Source") == "Rotten Tomatoes":
            rates += f'<div class="score"><small>Rotten Tomatoes</small><b>🍅 {esc(r.get("Value"))}</b></div>'
    credits = d.get("credits") or {}
    directors = [x["name"] for x in credits.get("crew", []) if x.get("job") == "Director"][:3] or [x["name"] for x in d.get("created_by", [])][:3]
    writers = [x["name"] for x in credits.get("crew", []) if x.get("job") in ("Screenplay", "Writer", "Story")][:3]
    crew = ""
    if directors:
        crew += f'<div class="crew"><b>{"Created by" if mt == "tv" and d.get("created_by") else "Director"}:</b> {esc(", ".join(directors))}</div>'
    if writers:
        crew += f'<div class="crew"><b>Writers:</b> {esc(", ".join(writers))}</div>'
    cast = [c for c in credits.get("cast", [])][:18]
    if cast:
        crew += f'<div class="crew"><b>Starring:</b> {esc(", ".join(c["name"] for c in cast[:5]))}</div>'
    buttons = f'<a class="btn red" href="{esc(trailer_url)}" target="_blank" rel="noopener">▶ Trailer on YouTube</a>'
    buttons += f'<a class="btn" href="https://www.themoviedb.org/{mt}/{mid}/watch?locale={REGION}" target="_blank" rel="noopener">📺 Where to watch</a>'
    if imdb_id:
        buttons += f'<a class="btn ghost" href="https://www.imdb.com/title/{esc(imdb_id)}/" target="_blank" rel="noopener">IMDb ↗</a>'
    tag = f'<div class="tag">“{esc(d["tagline"])}”</div>' if d.get("tagline") else ""
    poster = f'<img class="dposter" src="{img(d.get("poster_path"), "w500")}" alt="{esc(title)}">' if d.get("poster_path") else ""
    bd = img(d.get("backdrop_path"), "w1280")
    hero = (f'<div class="dhero" style="background-image:url({bd})"><div class="dh-in">{poster}<div class="dinfo"><div class="eyebrow">{"Series" if mt == "tv" else "Movie"} · MovieFlix</div>'
            f'<h1>{esc(title)} <span>{("(" + year + ")") if year else ""}</span></h1>{tag}<div class="chips">{"".join(meta)}{genres}</div><div class="rates">{rates}</div>'
            f'<p class="ov2">{esc(d.get("overview") or "No overview is available.")}</p>{crew}<div class="btns">{buttons}</div></div></div></div>')
    st.markdown(hero, unsafe_allow_html=True)

    wl = load_watchlist()
    in_list = any(x.get("id") == mid and x.get("media_type") == mt for x in wl)
    if st.button("✓ In My List — click to remove" if in_list else "＋ Add to My List", key=f"wl_{mt}_{mid}"):
        if in_list:
            wl = [x for x in wl if not (x.get("id") == mid and x.get("media_type") == mt)]
        else:
            wl.append({"id": mid, "media_type": mt, "title": title, "poster_path": d.get("poster_path"), "vote_average": d.get("vote_average"),
                       "release_date": d.get("release_date") or d.get("first_air_date"), "overview": clip(d.get("overview"), 200)})
        save_watchlist(wl)
        st.rerun()

    st.markdown('<div class="sec">Trailer & videos</div>', unsafe_allow_html=True)
    if vids:
        labels = [f"{v.get('type', 'Video')} — {clip(v.get('name', ''), 60)}" for v in vids[:8]]
        pick = 0
        if len(labels) > 1:
            pick = labels.index(st.selectbox("Choose a video", labels, key=f"vid_{mt}_{mid}"))
        left, mid_col, right = st.columns([1, 3.2, 1])
        with mid_col:
            video_url = f"https://www.youtube.com/watch?v={vids[pick]['key']}"
            if qp("t") == "1" and pick == 0:
                try:
                    st.video(video_url, autoplay=True, muted=True)
                except TypeError:
                    st.video(video_url)
            else:
                st.video(video_url)
            st.markdown(f'<div class="note">Video not playing? <a class="chip" href="{esc(video_url)}" target="_blank" rel="noopener">Open on YouTube ↗</a></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="note">No trailer listed on TMDB. <a class="chip" href="{esc(trailer_url)}" target="_blank" rel="noopener">Search YouTube ↗</a></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sec">Where to watch<small>{esc(REGIONS[REGION])}</small></div>', unsafe_allow_html=True)
    all_prov = (d.get("watch/providers") or {}).get("results", {})
    prov = all_prov.get(REGION, {})
    watch_link = prov.get("link") or f"https://www.themoviedb.org/{mt}/{mid}/watch?locale={REGION}"
    groups = [("flatrate", "Stream"), ("free", "Free"), ("ads", "Free with ads"), ("rent", "Rent"), ("buy", "Buy")]
    seen, block = set(), ""
    for key, label in groups:
        entries = [p for p in prov.get(key, []) if (key, p.get("provider_name")) not in seen]
        if not entries:
            continue
        chips = ""
        for p in entries:
            seen.add((key, p.get("provider_name")))
            logo = img(p.get("logo_path"), "w92")
            logo_html = f'<img src="{logo}" alt="">' if logo else ""
            chips += f'<a class="prov" href="{esc(provider_link(p.get("provider_name", ""), title, watch_link))}" target="_blank" rel="noopener">{logo_html}{esc(p.get("provider_name"))}</a>'
        block += f'<div class="provgrp">{label}</div>{chips}'
    if block:
        st.markdown(block + f'<div class="note" style="margin-top:8px">Netflix, Prime Video and YouTube open a search for this title. <a class="chip" href="{esc(watch_link)}" target="_blank" rel="noopener">All options & direct links ↗</a></div>', unsafe_allow_html=True)
    else:
        other = ", ".join(c for c in all_prov if c != REGION and all_prov[c].get("flatrate"))[:80]
        extra = f" Streaming in: {esc(other)}." if other else ""
        st.markdown(f'<div class="note">TMDB lists no streaming provider for this title in {esc(REGIONS[REGION])} right now.{extra} Availability changes often.</div>', unsafe_allow_html=True)
    q = quote_plus(title)
    st.markdown(chips_html([("Search Netflix", f"https://www.netflix.com/search?q={q}"), ("Search Prime Video", f"https://www.primevideo.com/search/?phrase={q}"),
                            ("Search YouTube", f"https://www.youtube.com/results?search_query={q}"), ("Google it", f"https://www.google.com/search?q=watch+{q}+online")]).replace('target="_self"', 'target="_blank" rel="noopener"'), unsafe_allow_html=True)

    if cast:
        st.markdown('<div class="sec">Cast<small>click anyone for their filmography</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="row">{"".join(pcard_html(c) for c in cast)}</div>', unsafe_allow_html=True)

    langs = {s.get("iso_639_1"): s.get("english_name") for s in d.get("spoken_languages", [])}
    facts = [("Status", d.get("status")), ("Release date" if mt == "movie" else "First aired", d.get("release_date") or d.get("first_air_date")),
             ("Original language", langs.get(d.get("original_language")) or (d.get("original_language") or "").upper()),
             ("Budget", fmt_money(d.get("budget"))), ("Box office", fmt_money(d.get("revenue"))),
             ("Episodes", d.get("number_of_episodes")), ("Studio", ", ".join(c["name"] for c in d.get("production_companies", [])[:2])),
             ("Countries", ", ".join(c["name"] for c in d.get("production_countries", [])[:2])), ("Awards", om.get("Awards") if om.get("Awards") not in (None, "N/A") else "")]
    fact_html = "".join(f'<div class="fact"><small>{esc(k)}</small><b>{esc(v)}</b></div>' for k, v in facts if v)
    if fact_html:
        st.markdown(f'<div class="sec">Details</div><div class="facts">{fact_html}</div>', unsafe_allow_html=True)

    recs = (d.get("recommendations") or {}).get("results", []) or (d.get("similar") or {}).get("results", [])
    block = row_html("More like this", recs, mt)
    if block:
        st.markdown(block, unsafe_allow_html=True)


def render_person():
    spacer()
    try:
        pid = int(qp("id", "0"))
    except ValueError:
        pid = 0
    p = T(f"person/{pid}", {"append_to_response": "combined_credits", "language": "en-US"}) if pid else {}
    if not p.get("id"):
        st.error("Couldn't load this person.")
        return
    photo = f'<img class="dposter" style="width:250px;border-radius:50%" src="{img(p.get("profile_path"), "w500")}" alt="">' if p.get("profile_path") else ""
    facts = " · ".join(x for x in [p.get("known_for_department"), f"Born {p['birthday']}" if p.get("birthday") else "", p.get("place_of_birth") or ""] if x)
    st.markdown(f'<div class="dh-in">{photo}<div class="dinfo"><div class="eyebrow">Cast & crew</div><h1>{esc(p.get("name"))}</h1><div class="chips"><span class="chip k">{esc(facts)}</span></div>'
                f'<p class="bio">{esc(clip(p.get("biography") or "No biography available.", 950))}</p></div></div>', unsafe_allow_html=True)
    cc = p.get("combined_credits") or {}

    def uniq(items):
        out = {}
        for x in items:
            if x.get("poster_path") and x.get("id") and x.get("media_type") in ("movie", "tv"):
                out[(x["media_type"], x["id"])] = x
        return sorted(out.values(), key=lambda x: x.get("popularity", 0), reverse=True)

    cast_items = uniq(cc.get("cast", []))[:48]
    directed = uniq([x for x in cc.get("crew", []) if x.get("job") == "Director"])[:24]
    if cast_items:
        st.markdown('<div class="sec">Known for</div>' + grid_html(cast_items), unsafe_allow_html=True)
    if directed:
        st.markdown('<div class="sec">Directed</div>' + grid_html(directed), unsafe_allow_html=True)


def render_mylist():
    spacer()
    st.markdown('<div class="eyebrow">Your collection</div><h2 style="margin:.2rem 0 .6rem">My List</h2>', unsafe_allow_html=True)
    wl = load_watchlist()
    if not wl:
        st.markdown('<div class="panel">Nothing saved yet. Open any title and press <b>＋ Add to My List</b>.</div>', unsafe_allow_html=True)
        return
    st.markdown(grid_html(wl), unsafe_allow_html=True)
    st.markdown('<div class="note">Saved in watchlist.json next to app.py (shared by everyone using this deployment).</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------
# Machine learning: taste-match recommender (TF-IDF content similarity + collaborative signal + quality prior)
# ----------------------------------------------------------------------------------------------
def taste_match(titles):
    favs = []
    for t in titles:
        res = T("search/movie", {"query": t, "include_adult": "false", "language": "en-US"}).get("results", [])
        if res:
            favs.append(res[0])
    if not favs:
        return [], []
    gmap = {g["id"]: g["name"] for g in T("genre/movie/list", {"language": "en-US"}).get("genres", [])}
    fav_ids = {f["id"] for f in favs}
    jobs = [(f"movie/{f['id']}/{ep}", {"language": "en-US"}) for f in favs for ep in ("recommendations", "similar")]
    cand, freq = {}, Counter()
    for (path, _), out in zip(jobs, fetch_many(jobs)):
        weight = 2 if path.endswith("recommendations") else 1
        for m in out.get("results", [])[:20]:
            if m.get("id") in fav_ids or not m.get("poster_path"):
                continue
            cand[m["id"]] = m
            freq[m["id"]] += weight
    if not cand:
        return favs, []

    def doc(m):
        g = " ".join(gmap.get(i, "").replace(" ", "") for i in m.get("genre_ids", []))
        return f"{title_of(m)} {m.get('overview', '')} {g} {g} {g} lang{m.get('original_language', '')}"

    ids = list(cand)
    docs = [doc(cand[i]) for i in ids]
    if HAS_SKLEARN:
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        X = vec.fit_transform(docs + [doc(f) for f in favs])
        profile = np.asarray(X[len(docs):].mean(axis=0))
        sim = linear_kernel(profile, X[:len(docs)]).ravel()
        sim = sim / sim.max() if sim.max() > 0 else sim
    else:
        sim = np.zeros(len(ids))
    quality = np.array([(cand[i].get("vote_average") or 0) / 10 * min(1.0, (cand[i].get("vote_count") or 0) / 200) for i in ids])
    collab = np.array([freq[i] for i in ids], dtype=float)
    collab = collab / collab.max()
    score = 0.55 * sim + 0.25 * quality + 0.20 * collab
    order = np.argsort(-score)[:24]
    return favs, [(cand[ids[i]], float(score[i])) for i in order]


def normalize_local_df(df):
    aliases = {"title": ["title", "original_title", "movie_title", "name"], "overview": ["overview", "description", "plot"], "genres": ["genres", "genre"], "cast": ["cast", "actors", "starring"],
               "director": ["director", "directors"], "keywords": ["keywords", "tags"], "original_language": ["original_language", "language"], "release_date": ["release_date", "release_year", "year"]}
    cols = {str(c).strip().lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for target, options in aliases.items():
        found = next((cols[o] for o in options if o in cols), None)
        out[target] = (df[found] if found else "")
        out[target] = out[target].fillna("").astype(str)
    out["title"] = out["title"].str.strip()
    out = out[out["title"].ne("")].drop_duplicates(subset=["title"], keep="first")
    return out.reset_index(drop=True)


def find_local_csv():
    candidates = []
    env_path = os.getenv("MOVIE_CSV_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates += [Path("IMDB_Data.csv"), Path("data/IMDB_Data.csv")]
    return next((p for p in candidates if p.exists() and p.is_file()), None)


@st.cache_resource(show_spinner="Training TF-IDF model on your dataset…")
def build_local_model(path, mtime):
    df = normalize_local_df(pd.read_csv(path, low_memory=False))
    if df.empty:
        return df, None
    text = (df["title"] + " " + df["genres"] + " " + df["genres"] + " " + df["director"] + " " + df["cast"] + " " + df["keywords"] + " " + df["overview"]).str.lower()
    vec = TfidfVectorizer(stop_words="english", max_features=60000, ngram_range=(1, 2), sublinear_tf=True)
    return df, vec.fit_transform(text)


def render_foryou():
    spacer()
    st.markdown('<div class="eyebrow">Powered by machine learning</div><h2 style="margin:.2rem 0 .6rem">✨ For You</h2>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🎯 Taste match", "📂 My CSV dataset"])
    with tab1:
        favs_q = qp("favs")
        with st.form("fav_form"):
            c1, c2 = st.columns([5, 1.2])
            txt = c1.text_input("Movies you love (comma-separated, up to 5)", value=favs_q, placeholder="Interstellar, Dangal, Inception")
            go = c2.form_submit_button("Find my next watch")
        if go:
            st.query_params.update({"view": "foryou", "favs": txt})
            st.rerun()
        titles = [t.strip() for t in favs_q.split(",") if t.strip()][:5]
        if not titles:
            st.markdown('<div class="panel"><div class="eyebrow">How it works</div>Tell MovieFlix a few favourites. It collects similar titles, builds a TF-IDF profile from plot + genre text, and ranks candidates by '
                        '<b>content similarity (55%)</b>, <b>audience quality (25%)</b> and <b>how often TMDB links them to your favourites (20%)</b>.</div>', unsafe_allow_html=True)
        else:
            favs, ranked = taste_match(titles)
            if not favs:
                st.warning("Couldn't find those titles. Check the spelling and try again.")
            else:
                st.markdown(f'<div class="sec">Your taste profile</div>{grid_html(favs).replace("class=\"grid\"", "class=\"grid\" style=\"grid-template-columns:repeat(auto-fill,minmax(120px,140px))\"")}', unsafe_allow_html=True)
                if ranked:
                    st.markdown('<div class="sec">Picked for you<small>ranked by match score</small></div>', unsafe_allow_html=True)
                    st.markdown('<div class="grid">' + "".join(card_html(m, "movie", f"{int(s * 100)}% match") for m, s in ranked) + "</div>", unsafe_allow_html=True)
                else:
                    st.info("No recommendations were returned. Try different titles.")
    with tab2:
        path = find_local_csv()
        if not HAS_SKLEARN:
            st.warning("Install scikit-learn to use the recommender: pip install scikit-learn")
        elif not path:
            st.info("Place IMDB_Data.csv beside app.py (or set MOVIE_CSV_PATH) to get content-based recommendations from your own dataset.")
        else:
            df, X = build_local_model(str(path), path.stat().st_mtime)
            if X is None or df.empty:
                st.error("The CSV loaded, but no recognised title column has values. Check the headers.")
            else:
                pick = st.selectbox("Pick a title you liked", df["title"].tolist()[:20000])
                idx = int(df.index[df["title"] == pick][0])
                sims = linear_kernel(X[idx], X).ravel()
                order = [i for i in sims.argsort()[::-1] if i != idx][:12]
                out = df.iloc[order][["title", "release_date", "genres", "director", "cast"]].copy()
                out.insert(0, "match %", (sims[order] * 100).round(1))
                st.dataframe(out, hide_index=True)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def setup_panel():
    spacer()
    st.markdown('<div class="panel"><div class="eyebrow">One-time setup</div><h2>Connect your TMDB key</h2>Get a free API key (v3 auth) at <a class="chip" href="https://www.themoviedb.org/settings/api" target="_blank" rel="noopener">themoviedb.org/settings/api</a>, '
                'then create <b>.streamlit/secrets.toml</b> next to app.py. Never paste keys into chat or commit them to GitHub.</div>', unsafe_allow_html=True)
    st.code('TMDB_API_KEY = "your_tmdb_v3_key"\nOMDB_API_KEY = "optional_key_for_real_IMDb_ratings"', language="toml")


def main():
    view = qp("view", "home")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(nav_html(view), unsafe_allow_html=True)
    if not API_KEY:
        setup_panel()
    else:
        pages = {"home": render_home, "browse": render_browse, "search": render_search, "movie": render_movie, "person": render_person, "foryou": render_foryou, "mylist": render_mylist}
        pages.get(view, render_home)()
    st.divider()
    st.markdown('<div class="note">MovieFlix is an independent discovery project. This product uses the TMDB API but is not endorsed or certified by TMDB. Streaming availability is supplied by TMDB / JustWatch '
                'and may change by region and date. IMDb and Rotten Tomatoes scores (optional) come from OMDb.</div>', unsafe_allow_html=True)


main()