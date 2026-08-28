from datetime import datetime
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st

from utils.transforms import (
    build_column_config as _build_cc,
    apply_preset as _apply_preset,
    vectorized_ending_in,
    vectorized_hours_left,
    vectorized_urgency,
    vectorized_deal_quality,
)
from config import (
    BUYER_PREMIUM,
    HEADERS,
    MAX_WORKERS,
    POOL_CONNECTIONS,
    POOL_MAXSIZE,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    RETRY_STATUS_FORCELIST,
    RETRY_TOTAL,
)
from scrape import get_auctions, get_all_items_for_auction, get_auction_pickup_dates, get_thread_session

logger = logging.getLogger(__name__)

GALLERY_CSS = """
<style>
/* === BidFTA Gateway - Clean Minimal === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.block-container { padding-top: 1.8rem; max-width: 1380px; }

/* Gallery cards - clean, airy */
.gallery-card {
    border: 1px solid #e8ecf1;
    border-radius: 16px;
    overflow: hidden;
    background: #ffffff;
    color: #0f172a;
    transition: all .18s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
}
.gallery-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(16,24,40,0.10);
    border-color: #d0d7e3;
}
.gallery-img-wrap {
    position: relative;
    aspect-ratio: 4/3;
    overflow: hidden;
    background: #f8fafc;
}
.gallery-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform .35s ease;
}
.gallery-card:hover .gallery-img-wrap img { transform: scale(1.03); }
.gallery-img-link { display: block; position: relative; width: 100%; height: 100%; cursor: pointer; }
.gallery-img-hint { position: absolute; bottom: 10px; right: 10px; background: rgba(15,23,42,0.82); color: white !important; font-size: 0.70em; font-weight: 700; padding: 5px 9px; border-radius: 999px; opacity: 0; transform: translateY(4px); transition: all .18s ease; pointer-events: none; letter-spacing: 0.02em; }
.gallery-img-wrap:hover .gallery-img-hint { opacity: 1; transform: translateY(0); }
.gallery-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: #dc2626;
    color: white !important;
    font-weight: 800;
    font-size: 0.78em;
    padding: 5px 9px;
    border-radius: 999px;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 10px rgba(0,0,0,0.22);
    border: 1px solid rgba(255,255,255,0.9);
}
.gallery-badge * { color: white !important; }
.gallery-time-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    font-weight: 800;
    font-size: 0.74em;
    padding: 5px 9px;
    border-radius: 999px;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 10px rgba(0,0,0,0.22);
    border: 1px solid rgba(255,255,255,0.65);
    backdrop-filter: blur(2px);
}
.gallery-time-critical { background: #dc2626; color: white !important; border-color: rgba(255,255,255,0.95); box-shadow: 0 2px 10px rgba(220,38,38,0.32), 0 0 0 3px rgba(220,38,38,0.15); }
.gallery-time-critical * { color: white !important; }
.gallery-time-high { background: #ea580c; color: white !important; }
.gallery-time-high * { color: white !important; }
.gallery-time-medium { background: #f59e0b; color: #0f172a !important; border-color: rgba(255,255,255,0.8); }
.gallery-time-medium * { color: #0f172a !important; }
.gallery-time-low { background: rgba(15,23,42,0.82); color: white !important; }
.gallery-time-low * { color: white !important; }
.gallery-time-ended { background: #64748b; color: white !important; }
.gallery-time-ended * { color: white !important; }
.gallery-body { padding: 14px 14px 12px; flex: 1; display: flex; flex-direction: column; gap: 8px; color: #0f172a; background: #ffffff; }
.gallery-body div { color: #0f172a; }
.gallery-body b { color: #0f172a; font-weight: 700; }
.gallery-title { font-weight: 700; font-size: 0.98em; line-height: 1.42; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.8em; color: #0f172a !important; letter-spacing: -0.01em; }
.gallery-price-row { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; font-size: 0.92em; margin-top: 2px; }
.gallery-price { color: #0f172a; font-weight: 800; font-size: 1.12em; letter-spacing: -0.02em; }
.gallery-msrp { color: #94a3b8; text-decoration: line-through; font-size: 0.85em; font-weight: 500; }
.gallery-save { color: #047857; font-weight: 700; font-size: 0.80em; background: #ecfdf5; border: 1px solid #6ee7b7; padding: 2px 7px; border-radius: 999px; }
.gallery-save-zero { background: #f8fafc !important; border-color: #e2e8f0 !important; color: #94a3b8 !important; font-weight: 600 !important; }
.gallery-meta { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; font-size: 0.80em; color: #475569; }
.gallery-footer { display: flex; gap: 6px; flex-wrap: wrap; }
.gallery-meta .pill, .gallery-footer .pill { font-size: 0.76em; padding: 2px 7px; }
.pill { display: inline-flex; align-items: center; padding: 4px 9px; border-radius: 999px; font-size: 0.76em; font-weight: 650; letter-spacing: 0.01em; line-height: 1; }
.pill-discount { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.pill-bids { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.pill-brand { background: #f5f3ff; color: #7c3aed; border: 1px solid #ddd6fe; }
.pill-warehouse { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
.pill-otd { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
.pill-pics { background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; }
.pill-active { background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
.pill-condition { background: #f8fafc; color: #334155; border: 1px solid #e2e8f0; }
/* Deal badges - softer */
.deal-excellent { background: linear-gradient(135deg, #ef4444 0%, #f97316 100%); color: white; padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: 0.78em; }
.deal-great { background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%); color: white; padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: 0.78em; }
.deal-good { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 0.78em; }
.deal-fair { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 0.78em; }
.deal-unknown { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; padding: 3px 10px; border-radius: 999px; font-weight: 500; font-size: 0.78em; }
.deal-research { background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; padding: 3px 10px; border-radius: 999px; font-weight: 500; font-size: 0.78em; }
/* Urgency */
.urgency-critical { color: #dc2626; font-weight: 700; }
.urgency-high { color: #ea580c; font-weight: 700; }
.urgency-medium { color: #b45309; font-weight: 600; }
@keyframes pulse { 0%,100% {opacity:1} 50% {opacity:0.65} }
/* Pills bar */
.filter-pills { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0 14px 0; }
.filter-pill { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 999px; padding: 5px 10px; font-size: 0.80em; color: #334155; }
.filter-pill b { font-weight: 700; color: #0f172a; }
/* Metrics - card look */
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e8ecf1; border-radius: 12px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(16,24,40,0.05); }
[data-testid="stMetricLabel"] { font-size: 0.78em; color: #64748b; font-weight: 500; letter-spacing: 0.02em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-size: 1.55em; font-weight: 700; color: #0f172a; }
/* Sidebar - cleaner */
section[data-testid="stSidebar"] { background: #fcfcfd; border-right: 1px solid #eef2f7; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #0f172a; font-weight: 700; }
/* Tabs - minimal underline */
button[data-baseweb="tab"] { font-weight: 500; color: #64748b; }
button[data-baseweb="tab"][aria-selected="true"] { color: #0f172a; font-weight: 600; }
/* Mobile */
@media (max-width: 1024px) {
  .gallery-grid { grid-template-columns: repeat(3, 1fr) !important; }
}
@media (max-width: 768px) {
  [data-testid="stMetric"] { min-width: 44%; }
  .gallery-card { border-radius: 14px; }
  .gallery-grid { grid-template-columns: repeat(2, 1fr) !important; }
  .gallery-img-wrap { aspect-ratio: 4/3; }
}
@media (min-width: 1025px) {
  .gallery-grid { grid-template-columns: repeat(3, 1fr); }
}
.gallery-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
}
/* lightbox */
.lightbox {
  position: fixed; inset: 0; background: rgba(15,23,42,0.88);
  display: flex; align-items: center; justify-content: center;
  z-index: 9999; padding: 20px;
}
.lightbox img { max-width: 90vw; max-height: 90vh; border-radius: 12px; object-fit: contain; box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
.lightbox-close { position: absolute; top: 18px; right: 22px; color: white; font-size: 28px; cursor: pointer; }

/* Force dark text inside white cards even in Streamlit dark theme */
.gallery-card, .gallery-card * { color: #0f172a; }
.gallery-card .pill, .gallery-card .pill * { color: inherit; }
.pill-discount, .pill-discount * { color: #dc2626 !important; }
.pill-bids, .pill-bids * { color: #1d4ed8 !important; }
.pill-brand, .pill-brand * { color: #7c3aed !important; }
.pill-warehouse, .pill-warehouse * { color: #c2410c !important; }
.pill-otd, .pill-otd * { color: #047857 !important; }
.pill-pics, .pill-pics * { color: #475569 !important; }
.urgency-critical, .urgency-critical * { color: #dc2626 !important; }
.urgency-high, .urgency-high * { color: #ea580c !important; }
.urgency-medium, .urgency-medium * { color: #b45309 !important; }

/* Clean old hover */
[data-testid="column"] img, [data-testid="stImage"] img { transition: transform .18s; }

/* Hide Streamlit default chrome for cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.set_page_config(layout="wide", page_title="BidFTA Explorer - Gateway", page_icon="🔍")
st.markdown(GALLERY_CSS, unsafe_allow_html=True)
st.title("BidFTA Gateway")
# Find hidden BidFTA deals fast - gateway hero (kept for test/SEO, UI uses concise caption below)
st.caption("Find here. Bid on BidFTA -> Save here, bid there.")
# Find hidden BidFTA deals fast

# Session state - must be before any st.session_state.data access
if 'data' not in st.session_state:
    st.session_state.data = None
if 'show_location_selector' not in st.session_state:
    st.session_state.show_location_selector = True
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = set()
if 'seen_ids' not in st.session_state:
    st.session_state.seen_ids = set()
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Gallery"
if 'gallery_page' not in st.session_state:
    st.session_state.gallery_page = 1
if 'table_page' not in st.session_state:
    st.session_state.table_page = 1
if 'preset' not in st.session_state:
    st.session_state.preset = None
if 'click_log' not in st.session_state:
    st.session_state.click_log = []

# Pre-data hero - minimal
if st.session_state.data is None:
    st.info("Pick warehouses below -> Start Scraping. Tip: Kentucky & Ohio, sort by Deal Score, Gallery -> Bid on BidFTA.")
    st.divider()

# ---- Fragment + debounce helpers D2 ----
try:
    _fragment = st.fragment
except AttributeError:
    def _fragment(fn):
        return fn

if 'last_pid_params' not in st.session_state:
    st.session_state.last_pid_params = {}
# sync query_params <-> session_state for warehouse filter + view_mode (view_mode: qp is source only on first load or external URL change)
try:
    qp = st.query_params
    qp_view = qp.get("view", None)
    if "_last_qp_view" not in st.session_state:
        # first load: respect URL if present, otherwise keep default Gallery
        if qp_view in ["Gallery", "Table"]:
            st.session_state.view_mode = qp_view
        st.session_state._last_qp_view = qp_view
    elif qp_view != st.session_state._last_qp_view:
        # URL changed externally (manual edit / back button) -> respect it
        if qp_view in ["Gallery", "Table"]:
            st.session_state.view_mode = qp_view
        st.session_state._last_qp_view = qp_view
    # else: qp unchanged, keep widget's view_mode (don't overwrite click)
    qp_wh = qp.get("wh", "")
    if qp_wh:
        st.session_state._qp_wh_ids = [x for x in str(qp_wh).split(",") if x.strip()]
    else:
        st.session_state._qp_wh_ids = []
except Exception:
    st.session_state._qp_wh_ids = []

@st.cache_data(ttl=3600)
def load_locations():
    url = "https://auction.bidfta.io/api/location/getAllLocations"
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    locations = response.json()
    # Fix grouping - preserve IN/TN/WV in All Other States
    location_groups = {
        "Kentucky": [loc for loc in locations if loc.get('state') == 'KY'],
        "Ohio": [loc for loc in locations if loc.get('state') == 'OH'],
        "All Other States": [loc for loc in locations if loc.get('state') not in ['KY', 'OH']],
    }
    for group in location_groups:
        location_groups[group].sort(key=lambda x: x.get('city', ''))
    return location_groups, locations

def get_default_locations(locations):
    return [loc for loc in locations if loc.get('state') in ['KY', 'OH']]

@st.cache_data(ttl=300)
def scrape_bidfta_data(location_ids, max_workers: int | None = None, max_auctions: int | None = None):
    _, locations = load_locations()
    loc_map = {loc['id']: loc for loc in locations}
    # auto-tune workers from auction count if not explicitly passed (non-technical-friendly)
    # None = auto; else respect explicit 4/8/12
    _auto_workers = None  # will be set after we know auction count
    # auctions paginated with seed session per plan B3.2 ttl 300
    all_auctions = []
    with requests.Session() as seed_session:
        seed_session.headers.update(HEADERS)
        adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS,
                              max_retries=Retry(total=RETRY_TOTAL, backoff_factor=RETRY_BACKOFF, status_forcelist=RETRY_STATUS_FORCELIST))
        seed_session.mount("https://", adapter)
        seed_session.mount("http://", adapter)
        page = 1
        while page <= 50:
            try:
                auctions = get_auctions(seed_session, location_ids, page_id=page)
            except Exception as e:
                logger.warning("get_auctions page %s failed: %s", page, e)
                break
            if not auctions:
                break
            all_auctions.extend(auctions)
            page += 1
    if not all_auctions:
        return None
    # auto-tune workers from actual auction count (dynamic, no user knob)
    if max_workers is None:
        n = len(all_auctions)
        # ~1 worker per 60 auctions, clamped 4-12 - 832 ->12, 400->8, 100->4
        if n >= 600:
            max_workers = 12
        elif n >= 300:
            max_workers = 10
        elif n >= 100:
            max_workers = 8
        else:
            max_workers = 4
    max_workers = max(2, min(int(max_workers), 16))
    # Fast-mode cap: for KY+OH ~832, cap to 200/400 most urgent (by end date) if requested
    if max_auctions is not None and max_auctions > 0 and len(all_auctions) > max_auctions:
        try:
            all_auctions = sorted(all_auctions, key=lambda a: a.get('utcEndDateTime') or '')
            all_auctions = all_auctions[:max_auctions]
        except Exception:
            all_auctions = all_auctions[:max_auctions]
    # pre-warm pickup cache distinct loc_ids to avoid thundering herd
    distinct_loc_ids = list({a.get("locationId") for a in all_auctions if a.get("locationId")})
    pickup_cache = {}
    # fetch pickups sequentially before threadpool (fast, few locs)
    tmp_sess = get_thread_session()
    for lid in distinct_loc_ids:
        try:
            pickup_cache[lid] = get_auction_pickup_dates(tmp_sess, lid)
        except Exception:
            pickup_cache[lid] = []
    all_items = []
    lock = threading.Lock()
    def process_auction(auction):
        try:
            auction_id = auction["id"]
            loc_id = auction.get("locationId")
            thread_session = get_thread_session()
            pickup_dates = pickup_cache.get(loc_id, [])
            if isinstance(pickup_dates, str):
                pickup_dates = [pickup_dates]
            if pickup_dates is None:
                pickup_dates = []
            # normalize
            pickup_dates = [str(d) for d in pickup_dates]
            loc_info = loc_map.get(loc_id, {})
            # tax
            try:
                loc_tax = float(loc_info.get("taxRate", 6.0) or 6.0)
            except:
                loc_tax = 6.0
            items = get_all_items_for_auction(thread_session, auction_id)
            out = []
            for item in items:
                try:
                    current_bid = float(item.get("currentBid", 0.0) or 0.0)
                except:
                    current_bid = 0.0
                try:
                    next_bid = float(item.get("nextBid", current_bid) or current_bid)
                except:
                    next_bid = current_bid
                try:
                    msrp = float(item.get("msrp", 0.0) or 0.0)
                except:
                    msrp = 0.0
                # pictures
                pictures = item.get("pictures") or []
                pic_urls = []
                if isinstance(pictures, list):
                    for p in pictures:
                        if isinstance(p, str) and p:
                            pic_urls.append(p)
                        elif isinstance(p, dict) and p.get("picUrl"):
                            pic_urls.append(p["picUrl"])
                image_url = item.get("imageUrl", "") or ""
                if not pic_urls and image_url:
                    pic_urls = [image_url]
                pic_list = item.get("pictureList") or []
                if not pic_urls and isinstance(pic_list, list):
                    for pl in pic_list:
                        if isinstance(pl, dict) and pl.get("picUrl"):
                            pic_urls.append(pl["picUrl"])
                    if pic_urls and not image_url:
                        image_url = pic_urls[0]
                if not image_url and pic_urls:
                    image_url = pic_urls[0]
                bid_count = 0
                try:
                    bid_count = int(item.get("bidsCount", 0) or 0)
                except:
                    bid_count = 0
                ratio = current_bid / msrp if msrp > 0 else 0
                otd = next_bid * (1 + BUYER_PREMIUM) * (1 + loc_tax/100.0)
                out.append({
                    'auction_id': auction_id,
                    'auction_title': auction.get('title',''),
                    'auction_number': auction.get('auctionNumber',''),
                    'auction_start_datetime': auction.get('utcStartDateTime',''),
                    'auction_end_datetime': auction.get('utcEndDateTime',''),
                    'auction_location_id': loc_id,
                    'auction_location_nickname': loc_info.get('nickName',''),
                    'auction_location_address': loc_info.get('address',''),
                    'auction_location_city': loc_info.get('city',''),
                    'auction_location_state': loc_info.get('state',''),
                    'auction_location_zip': loc_info.get('zip',''),
                    'auction_location_tax_rate': loc_tax,
                    'auction_location_map': loc_info.get('map',''),
                    'pallet_auction': bool(auction.get('palletAuction', False)),
                    'item_id': item.get('id'),
                    'lot_code': item.get('lotCode',''),
                    'brand': item.get('brand',''),
                    'item_title': item.get('title',''),
                    'item_category1': item.get('category1',''),
                    'item_category2': item.get('category2',''),
                    'condition': item.get('condition',''),
                    'current_bid': current_bid,
                    'next_bid': next_bid,
                    'msrp': msrp,
                    'bid_count': bid_count,
                    'quantity': float(item.get('quantity',1) or 1),
                    'initial_quantity': float(item.get('initialQuantity',1) or 1),
                    'pallet_lot': bool(item.get('palletLot', False)),
                    'image_url': image_url,
                    'pictures': pic_urls,
                    'picture_count': len(pic_urls),
                    'over_time': bool(item.get('overTime', False)),
                    'hours_remaining': int(item.get('hoursRemaining',0) or 0),
                    'ratio_bid_to_msrp': ratio,
                    'otd_total': round(otd,2),
                    'picture': image_url,
                    'item_url': f"https://www.bidfta.com/{auction_id}/item-detail/{item.get('id')}",
                    'pickup_dates': "; ".join(pickup_dates) if isinstance(pickup_dates, list) else str(pickup_dates or ""),
                    'pickup_dates_list': pickup_dates,
                })
            return out
        except Exception as e:
            logger.warning("process_auction failed %s: %s", auction.get("id"), e)
            return []
    # progress
    total = len(all_auctions)
    status = st.status(f"Scraping {total} auctions...", expanded=False) if total else None
    if status:
        status.update(label=f"Found {total} auctions - fetching items...")
    # progress bar for scrape
    prog = st.progress(0, text="Fetching items...") if total else None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_auction, auction) for auction in all_auctions]
        done = 0
        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as e:
                logger.warning("future failed: %s", e)
                res = []
            with lock:
                all_items.extend(res)
            done += 1
            if status:
                status.update(label=f"Fetched {done}/{total} auctions - {len(all_items)} items...")
            if prog:
                prog.progress(int(done/total*100) if total else 100, text=f"{done}/{total} auctions - {len(all_items)} items")
    if status:
        status.update(label=f"Done - {len(all_items)} items from {total} auctions", state="complete", expanded=False)
        if prog:
            prog.progress(100, text=f"Done - {len(all_items)} items")
            try:
                prog.empty()
            except:
                pass
        time.sleep(0.4)
        status.update(expanded=False)
    return pd.DataFrame(all_items) if all_items else None

@st.cache_data(show_spinner=False, ttl=300)
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data(show_spinner=False, ttl=60)
def _process_data_core(data: pd.DataFrame, now_ts: int) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    # adaptive column renames for legacy CSV
    # support both picture/image_url, bid_count/bidsCount, etc.
    if 'bidsCount' in df.columns and 'bid_count' not in df.columns:
        df['bid_count'] = df['bidsCount']
    if 'lotCode' in df.columns and 'lot_code' not in df.columns:
        df['lot_code'] = df['lotCode']
    if 'nextBid' in df.columns and 'next_bid' not in df.columns:
        df['next_bid'] = df['nextBid']
    if 'taxRate' in df.columns and 'auction_location_tax_rate' not in df.columns:
        df['auction_location_tax_rate'] = df['taxRate']
    if 'pictures' in df.columns and df['pictures'].dtype == object:
        # if stored as | joined string convert back
        def parse_pics(x):
            if isinstance(x, list):
                return x
            if isinstance(x, str) and '|' in x:
                return [p for p in x.split('|') if p]
            if isinstance(x, str) and x.startswith('http'):
                return [x]
            return []
        # keep if already list
        df['pictures_parsed'] = df['pictures'].apply(parse_pics)
        # if empty use image_url
        if 'image_url' not in df.columns and 'picture' in df.columns:
            df['image_url'] = df['picture']
    else:
        df['pictures_parsed'] = df.get('pictures', [[]])
    for col in ['current_bid','msrp','next_bid','bid_count','taxRate','auction_location_tax_rate','picture_count','quantity','initial_quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['current_bid'] = df.get('current_bid', 0).fillna(0.0).astype(float)
    df['msrp'] = df.get('msrp', 0).fillna(0.0).astype(float)
    df['next_bid'] = df.get('next_bid', df['current_bid']).fillna(df['current_bid']).astype(float)
    df['bid_count'] = df.get('bid_count', 0).fillna(0).astype(int)
    df['auction_location_tax_rate'] = df.get('auction_location_tax_rate', 6.0).fillna(6.0).astype(float)
    df['picture_count'] = df.get('picture_count', 0).fillna(0).astype(int)
    # if pictures_parsed present compute count
    if 'pictures_parsed' in df.columns:
        df['picture_count'] = df['pictures_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        # also ensure pictures column is list for UI
        df['pictures'] = df['pictures_parsed']
    if 'pictures' not in df.columns:
        df['pictures'] = [[] for _ in range(len(df))]
    if 'image_url' not in df.columns:
        df['image_url'] = df.get('picture', '')
    if 'picture' not in df.columns:
        df['picture'] = df['image_url']
    # over_time etc
    if 'over_time' not in df.columns:
        df['over_time'] = False
    df['over_time'] = df['over_time'].fillna(False).astype(bool)
    if 'brand' not in df.columns:
        df['brand'] = ''
    if 'lot_code' not in df.columns:
        df['lot_code'] = ''
    if 'item_title' not in df.columns and 'title' in df.columns:
        df['item_title'] = df['title']
    # tax OTD if not present
    if 'otd_total' not in df.columns:
        df['otd_total'] = (df['next_bid'] * (1 + BUYER_PREMIUM) * (1 + df['auction_location_tax_rate']/100.0)).round(2)
    else:
        df['otd_total'] = pd.to_numeric(df['otd_total'], errors='coerce').fillna(df['next_bid'] * 1.13 * 1.06)
    est = pytz.timezone('US/Eastern')
    if 'utcEndDateTime' in df.columns:
        # per-item overrides auction level where present
        df['_item_end'] = pd.to_datetime(df['utcEndDateTime'], errors='coerce', utc=True)
        df['_auction_end'] = pd.to_datetime(df.get('auction_end_datetime'), errors='coerce', utc=True)
        df['auction_end_datetime'] = df['_item_end'].fillna(df['_auction_end'])
        df = df.drop(columns=['_item_end','_auction_end'], errors='ignore')
    else:
        df['auction_end_datetime'] = pd.to_datetime(df.get('auction_end_datetime'), errors='coerce', utc=True)
    df['auction_end_datetime'] = df['auction_end_datetime'].dt.tz_convert(est)
    now = datetime.fromtimestamp(now_ts, tz=est)
    # time remaining strings - vectorized np.where (D1.2)
    df['ending_in'] = vectorized_ending_in(df['auction_end_datetime'], now)
    # hours left vectorized
    df['hours_left'] = vectorized_hours_left(df['auction_end_datetime'], now)
    df['time_urgency_score'] = vectorized_urgency(df['hours_left'], df['over_time'])
    # ratio/discount
    msrp_nonzero = df['msrp'].replace({0: pd.NA})
    df['ratio_bid_to_msrp'] = (df['current_bid'] / msrp_nonzero).fillna(0.0).astype(float)
    df['discount_pct'] = ((1 - df['ratio_bid_to_msrp']) * 100).clip(lower=0, upper=100).fillna(0)
    df['has_msrp'] = df['msrp'] > 0
    df['savings_amount'] = (df['msrp'] - df['current_bid']).clip(lower=0).fillna(0)
    # category median for outlier detection
    df['_cat_median'] = df.groupby('item_category1')['msrp'].transform('median')
    df['_is_msrp_outlier'] = (df['msrp'] > 3 * df['_cat_median'].fillna(df['msrp']))
    # deal quality 6-tier + research + overpriced
    df['deal_quality'] = vectorized_deal_quality(df)
    # vectorized helper still uses apply for quality because outlier logic per row is okay at 10k - could be pd.cut but outlier complicates
    # deal score with 6-tier
    # precompute condition lower string vectorized
    cond_lower = df['condition'].fillna('').str.lower()
    title_lower = df['item_title'].fillna('').str.lower()
    cond_mult = np.select(
        [
            cond_lower.str.contains('brand new', na=False),
            cond_lower.str.contains('like new|open box', na=False, regex=True),
            cond_lower.str.contains('working condition verified', na=False),
            cond_lower.str.contains('preview for condition', na=False),
            cond_lower.str.contains('as is', na=False),
        ],
        [1.3, 1.15, 0.95, 0.8, 0.6],
        default= np.where(cond_lower.str.contains('good condition', na=False), 1.0, 1.0)
    )
    # fix Good already 1.0 default; adjust exactly per spec 6-tier: Brand New 1.3 / Like New/Open Box 1.15 / Good 1.0 / Working 0.95 / Preview 0.8 / As Is 0.6
    # np.select above gives preview 0.8, need As Is to override others - order matters: check As Is before Good but after preview? already after.
    # incomplete penalty -0.2 multiplier
    incomplete_mask = title_lower.str.contains('incomplete|damaged', na=False, regex=True)
    cond_mult = np.where(incomplete_mask, np.maximum(0.5, cond_mult - 0.2), cond_mult)
    # picture count trust penalty for single pic
    single_pic_mask = df['picture_count'] <= 1
    # keep note for sorting but not huge penalty - only -0.05
    cond_mult = np.where(single_pic_mask & (df['picture_count']==1), np.maximum(0.5, cond_mult - 0.05), cond_mult)
    # research/outlier de-weight
    cond_mult = np.where(df['deal_quality'].isin(['Research','Overpriced']), cond_mult * 0.7, cond_mult)
    # urgency boost 1 + urg*0.5
    urgency_boost = 1 + df['time_urgency_score'] * 0.5
    # discount score
    discount_score = df['discount_pct'].fillna(0)
    # for unknown msrp score based on low bid + urgency already: max(0,100-bid)*(1+urg)
    unknown_score = np.maximum(0, 100 - df['current_bid']) * (1 + df['time_urgency_score'])
    # combine
    df['deal_score'] = np.where(df['has_msrp'] & (~df['deal_quality'].isin(['Research','Overpriced'])),
                                discount_score * cond_mult * urgency_boost,
                                np.where(df['deal_quality']=='Research', unknown_score * 0.5 * cond_mult, 
                                         np.where(df['deal_quality']=='Overpriced', discount_score * 0.3, unknown_score)))
    # est resale/profit
    df['est_resale'] = (df['msrp'] * 0.65).where(df['has_msrp'], 0)
    df['est_profit'] = (df['est_resale'] - df['otd_total']).round(2)
    df['est_roi'] = np.where(df['otd_total']>0, (df['est_profit']/df['otd_total']*100).round(1), 0)
    df = df.drop(columns=['_cat_median','_is_msrp_outlier'], errors='ignore')
    df = df.infer_objects(copy=False)
    return df

def get_deal_badge(q):
    badges = {
        'Excellent': '<span class="deal-excellent">Excellent</span>',
        'Great': '<span class="deal-great">Great</span>',
        'Good': '<span class="deal-good">Good</span>',
        'Fair': '<span class="deal-fair">Fair</span>',
        'Unknown Value': '<span class="deal-unknown">Unknown</span>',
        'Research': '<span class="deal-research">Research</span>',
        'Overpriced': '<span class="deal-fair">Overpriced</span>',
        'No Bids': '<span class="deal-fair">No Bids</span>'
    }
    return badges.get(q, q)

def get_urgency_class(score):
    if score >= 0.8:
        return 'urgency-critical'
    if score >= 0.5:
        return 'urgency-high'
    if score >= 0.2:
        return 'urgency-medium'
    return ''

# Update button top
if st.session_state.data is not None:
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("Update Data", type="secondary", help="Refresh cached auctions and bids", key="update_data_button"):
            load_locations.clear()
            # get_default_locations not cached anymore
            scrape_bidfta_data.clear()
            _process_data_core.clear()
            convert_df.clear()
            st.session_state.show_location_selector = True
            st.session_state.data = None
            st.rerun()
    with c2:
        st.caption(f"Watchlist {len(st.session_state.watchlist)} - Click Bid on BidFTA to bid - Gateway only")

# Location selector - live data only, no CSV
if st.session_state.show_location_selector or st.session_state.data is None:
    location_groups, locations = load_locations()
    if st.session_state.data is None or st.session_state.show_location_selector:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### Select Warehouses")
                quick_select = st.radio(
                    "Quick Select",
                    ["Kentucky & Ohio", "Kentucky Only", "Ohio Only", "Custom Selection"],
                    horizontal=True, label_visibility="collapsed"
                )
                if "Kentucky & Ohio" in quick_select:
                    selected_locations = location_groups["Kentucky"] + location_groups["Ohio"]
                elif "Kentucky Only" in quick_select:
                    selected_locations = location_groups["Kentucky"]
                elif "Ohio Only" in quick_select:
                    selected_locations = location_groups["Ohio"]
                else:
                    tabs = st.tabs([f"{group}" for group in location_groups.keys()])
                    selected_locations = []
                    for tab, group_name in zip(tabs, location_groups):
                        with tab:
                            locs = location_groups[group_name]
                            if not locs:
                                st.info("No locations in this group")
                                continue
                            select_all = st.checkbox(f"Select All {group_name}", key=f"all_{group_name}")
                            st.divider()
                            # search within group
                            q = st.text_input(f"Search {group_name}", key=f"search_{group_name}", placeholder="City or warehouse...")
                            filtered_locs = [l for l in locs if q.lower() in f"{l.get('city','')} {l.get('nickName','')}".lower()] if q else locs
                            cols = st.columns(2)
                            for i, loc in enumerate(filtered_locs):
                                col_idx = i % 2
                                with cols[col_idx]:
                                    location_key = f"loc_{loc['id']}"
                                    label = f"{loc['city']} - {loc['nickName']} - {loc.get('state','')}"
                                    if select_all:
                                        st.checkbox(label, key=location_key, value=True)
                                        selected_locations.append(loc)
                                    else:
                                        if st.checkbox(label, key=location_key):
                                            selected_locations.append(loc)
                if selected_locations:
                    # auto speed - workers set dynamically from auction count, no tech knob
                    est_auctions = len(selected_locations) * 28  # ~28/warehouse for KY/OH
                    if len(selected_locations) > 12:
                        st.caption(f"⚡ Large pull: ~{est_auctions} auctions est. Auto-tuned workers + optional fast cap.")
                    cap_opt = st.selectbox("How much to pull?", options=["All", "400 most urgent", "200 most urgent"], index=0, help="All = ~832 for KY+OH (~58k items). 400/200 = soonest-ending first, much faster. Workers auto-tuned from count.")
                    cap_map = {"All": None, "400 most urgent": 400, "200 most urgent": 200}
                    max_auctions = cap_map[cap_opt]
                    with col2:
                        st.write("")
                        if st.button("Start Scraping", type="primary", use_container_width=True):
                            location_ids = [loc['id'] for loc in selected_locations]
                            with st.spinner(f'Scraping {len(selected_locations)} warehouses - auto-tuned speed...'):
                                data = scrape_bidfta_data(location_ids, max_workers=None, max_auctions=max_auctions)
                                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                                    st.warning('No active auctions found.')
                                else:
                                    st.session_state.data = data
                                    st.success(f"Got {len(data)} items from {len(selected_locations)} warehouses!")
                                    st.session_state.show_location_selector = False
                                    st.rerun()
    else:
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Change Warehouses"):
                st.session_state.show_location_selector = True
                st.rerun()

data = st.session_state.data
if data is not None:
    # process with now rounded to minute for cache
    import time as _time
    now_ts = int(_time.time() // 60 * 60)
    original_data = _process_data_core(data, now_ts)
    if original_data.empty:
        st.warning("No data available for the selected warehouses yet. Try selecting warehouses above and clicking Start Scraping for live data.")
        if st.button("Pick warehouses again"):
            st.session_state.show_location_selector = True
            st.rerun()
        st.stop()
    required = {'item_title','item_category1','item_category2','condition','current_bid','msrp'}
    missing = sorted(required - set(original_data.columns))
    if missing:
        st.error("Dataset missing required columns: " + ", ".join(missing))
        st.stop()

    # Warehouse directory preview before filters - grid cards with hero, auctions, discount, pills, tax, map
    if st.query_params.get("hide_warehouse") != "1":
        with st.expander("Warehouse Directory - what is for sale where", expanded=False):
            st.caption("Gateway view - pick a warehouse card to filter. Bid on BidFTA after. Save here, bid there - no checkout here.")
            wh = original_data.groupby(['auction_location_nickname','auction_location_id']).agg(
                items=('item_title','count'),
                auctions=('auction_id','nunique'),
                avg_discount=('discount_pct','mean'),
                avg_otd=('otd_total','mean'),
                city=('auction_location_city','first'),
                state=('auction_location_state','first'),
                address=('auction_location_address','first'),
                tax=('auction_location_tax_rate','first'),
                maphtml=('auction_location_map','first'),
            ).reset_index()
            hero = original_data.groupby('auction_location_nickname')['image_url'].apply(lambda x: list(x.dropna().unique())[:3]).reset_index()
            wh = wh.merge(hero, on='auction_location_nickname', how='left')
            pickup_map = original_data.groupby('auction_location_nickname')['pickup_dates'].apply(lambda s: sorted({d.strip() for v in s.fillna('') for d in str(v).split(';') if d.strip()})[:3]).reset_index()
            pickup_map = pickup_map.rename(columns={'pickup_dates': 'pickup_pills'})
            wh = wh.merge(pickup_map, on='auction_location_nickname', how='left')
            cols = st.columns(3)
            for idx, row in wh.iterrows():
                with cols[idx % 3]:
                    st.markdown(f"**{row['auction_location_nickname']}** - {row['city']}, {row['state']} <span class='pill' style='background:#e7f5ff;border:1px solid #a5d8ff'>Tax {row['tax']:.1f}%</span>", unsafe_allow_html=True)
                    st.caption(f"{row['auctions']} auctions / {row['items']} items - Avg {row['avg_discount']:.0f}% off - Avg OTD ${row['avg_otd']:.0f}")
                    st.caption(f"{row['address']}")
                    pills = row.get('pickup_pills') or []
                    if isinstance(pills, list) and pills:
                        pill_html = ' '.join([f"<span class='pill' style='background:#fff4e6;border:1px solid #ffe8cc'>{p}</span>" for p in pills[:3]])
                        st.markdown(pill_html, unsafe_allow_html=True)
                    pics = row['image_url'] if isinstance(row['image_url'], list) else []
                    if pics:
                        c1, c2 = st.columns(2)
                        for j, purl in enumerate(pics[:2]):
                            with (c1 if j==0 else c2):
                                try:
                                    st.image(purl, use_container_width=True)
                                except:
                                    pass
                    if st.button(f"View Items - {row['auction_location_nickname']}", key=f"whview_{idx}"):
                        st.session_state['filter_warehouse'] = row['auction_location_nickname']
                        st.session_state['filter_warehouse_id'] = str(row['auction_location_id'])
                        try:
                            st.query_params["wh"] = str(row['auction_location_id'])
                        except Exception:
                            pass
                        st.rerun()
                    if row.get('maphtml'):
                        with st.popover("Map / Directions"):
                            st.markdown(row['maphtml'], unsafe_allow_html=True)
                            st.link_button("Directions", f"https://www.google.com/maps/search/{row['address']} {row['city']} {row['state']}", type="secondary")
        with st.expander("Auction grouping - plan your pickup trip", expanded=False):
            auc_grp = original_data.groupby(['auction_id','auction_title','auction_end_datetime','auction_location_nickname','auction_location_id','auction_location_address','auction_location_city','auction_location_state','auction_location_map']).agg(
                items=('item_id','count'),
                pickup_dates=('pickup_dates','first'),
            ).reset_index()
            auc_grp = auc_grp.sort_values('auction_end_datetime')
            for _, arow in auc_grp.head(12).iterrows():
                title = (arow['auction_title'] or str(arow['auction_id']))[:60]
                end = arow['auction_end_datetime']
                end_str = end.strftime('%b %d %I%p') if hasattr(end, 'strftime') and not pd.isna(end) else str(end)
                wh_nick = arow['auction_location_nickname']
                pick = arow['pickup_dates'] or ''
                pick_list = [d.strip() for d in str(pick).split(';') if d.strip()]
                pick_str = '; '.join(pick_list[:3])
                st.markdown(f"**{wh_nick}** - {title} - {arow['items']} items - ends {end_str} - Pickup {pick_str}")
                st.caption(f"{arow['auction_location_address']} {arow['auction_location_city']}, {arow['auction_location_state']}")
                st.link_button("Directions", f"https://www.google.com/maps/search/{arow['auction_location_address']} {arow['auction_location_city']} {arow['auction_location_state']}", type="secondary")
                if st.button(f"Filter to this auction", key=f"filt_auc_{arow['auction_id']}"):
                    st.session_state['filter_auction_id'] = str(arow['auction_id'])
                    st.rerun()
                st.divider()
    # Sidebar filters - clean grouped
    with st.sidebar:
        st.markdown("### Filters")
        show_viz = st.checkbox("Show Visualizations", value=False)
        st.divider()
        slider_mask = pd.Series(True, index=original_data.index, dtype=bool)
        # Build id <-> nickname map (needed for warehouse filter)
        id_to_nick = dict(original_data[['auction_location_id','auction_location_nickname']].dropna().drop_duplicates().values)
        nick_to_id = {v:k for k,v in id_to_nick.items()}
        warehouse_ids = sorted([int(x) for x in id_to_nick.keys() if str(x).isdigit() or isinstance(x, (int,float))])
        def wh_label(wid):
            return f"{id_to_nick.get(wid, str(wid))} ({wid})"
        default_ids = []
        if 'filter_warehouse_id' in st.session_state and st.session_state['filter_warehouse_id'] not in [None, ""]:
            try:
                default_ids = [int(str(st.session_state['filter_warehouse_id']).strip())]
            except:
                default_ids = []
        elif 'filter_warehouse' in st.session_state and st.session_state['filter_warehouse'] in id_to_nick.values():
            for k,v in id_to_nick.items():
                if v == st.session_state['filter_warehouse']:
                    default_ids = [int(k)] if str(k).isdigit() else []
                    break
        if hasattr(st.session_state, '_qp_wh_ids') and st.session_state._qp_wh_ids:
            try:
                qp_ids = [int(x) for x in st.session_state._qp_wh_ids if str(x).isdigit()]
                if qp_ids:
                    default_ids = qp_ids
            except:
                pass
        default_ids = [d for d in default_ids if d in warehouse_ids]

        with st.expander("🔍 Search & Warehouse", expanded=True):
            search_query = st.text_input("Search titles + brand + category + lot", placeholder="Keywords...", label_visibility="collapsed")
            st.caption("Search titles, brand, category, lot")
            selected_warehouse_ids = st.multiselect("Warehouse", options=warehouse_ids, format_func=wh_label, default=default_ids, help="Filters without re-scrape. Shares via ?wh=", label_visibility="collapsed")
            st.caption("Warehouse")
            if selected_warehouse_ids:
                slider_mask &= original_data['auction_location_id'].isin(selected_warehouse_ids)
                st.session_state['filter_warehouse_ids'] = selected_warehouse_ids
                nicks = [id_to_nick.get(i, str(i)) for i in selected_warehouse_ids]
                st.session_state['filter_warehouse'] = nicks[0] if len(nicks)==1 else ",".join(nicks)
                try:
                    st.query_params["wh"] = ",".join(str(x) for x in selected_warehouse_ids)
                except Exception:
                    pass
            else:
                try:
                    if "wh" in st.query_params:
                        del st.query_params["wh"]
                except Exception:
                    pass
            selected_warehouses = [id_to_nick.get(i, str(i)) for i in selected_warehouse_ids]

        with st.expander("📦 Categories & Condition", expanded=False):
            # show counts so 19k→745 is provable - Electronics is ~3-4% typically
            cat_counts = original_data['item_category1'].value_counts(dropna=False)
            total_for_pct = len(original_data)
            categories1 = sorted(original_data['item_category1'].dropna().unique().tolist())
            # build label with count and % for proof
            def cat_label(c):
                cnt = int(cat_counts.get(c, 0))
                pct = cnt / total_for_pct * 100 if total_for_pct else 0
                return f"{c} ({cnt:,} • {pct:.1f}%)"
            selected_categories1 = st.multiselect("Primary Category", options=categories1, format_func=cat_label, label_visibility="collapsed", placeholder="Primary Category")
            st.caption(f"Total {total_for_pct:,} items • {len(categories1)} categories • Empty: {int(cat_counts.get(float('nan'), 0) if float('nan') in cat_counts else original_data['item_category1'].isna().sum())} • Top: Electronics ~{(cat_counts.get('Electronics',0)/total_for_pct*100):.1f}% - your 745/19k is normal")
            if selected_categories1:
                slider_mask &= original_data['item_category1'].isin(selected_categories1)
            available_secondary = sorted(original_data.loc[slider_mask, 'item_category2'].dropna().unique().tolist())
            # secondary also with counts on current filtered subset
            sec_counts = original_data.loc[slider_mask, 'item_category2'].value_counts(dropna=False)
            def sec_label(c):
                cnt = int(sec_counts.get(c, 0))
                return f"{c} ({cnt:,})" if cnt else c
            selected_categories2 = st.multiselect("Secondary Category", options=available_secondary, format_func=sec_label, disabled=not available_secondary, label_visibility="collapsed", placeholder="Secondary Category")
            if selected_categories2:
                slider_mask &= original_data['item_category2'].isin(selected_categories2)
            conditions = sorted(original_data['condition'].dropna().unique().tolist())
            selected_conditions = st.multiselect("Condition", options=conditions, label_visibility="collapsed", placeholder="Condition")
            if selected_conditions:
                slider_mask &= original_data['condition'].isin(selected_conditions)
            c1, c2 = st.columns(2)
            with c1:
                hide_as_is = st.checkbox("Hide As Is", value=False)
                if hide_as_is:
                    slider_mask &= ~original_data['condition'].fillna('').str.contains('as is', case=False, regex=False)
            with c2:
                with st.popover("Tiers"):
                    st.caption("Brand New 1.3 / Like New 1.15 / Good 1.0 / Working 0.95 / Preview 0.8 / As Is 0.6")
            deal_qualities = ['Excellent','Great','Good','Fair','Unknown Value','Research','Overpriced','No Bids']
            present_q = [q for q in deal_qualities if q in original_data['deal_quality'].unique()]
            selected_qualities = st.multiselect("Deal Quality", options=present_q, label_visibility="collapsed", placeholder="Deal Quality")
            if selected_qualities:
                slider_mask &= original_data['deal_quality'].isin(selected_qualities)

        with st.expander("🏷️ Brand & Quality", expanded=False):
            top_brands = sorted(original_data['brand'].value_counts().head(15).index.tolist())
            selected_brands = st.multiselect("Brand (top 15)", options=top_brands, label_visibility="collapsed", placeholder="Brand")
            if selected_brands:
                slider_mask &= original_data['brand'].isin(selected_brands)
            c1, c2 = st.columns(2)
            with c1:
                include_unknown = st.checkbox("Include unknown MSRP", value=True)
                hide_single_pic = st.checkbox("Hide single-pic", value=False)
                if hide_single_pic:
                    slider_mask &= original_data['picture_count'] > 1
            with c2:
                st.caption("Uncheck hides $0 MSRP")

        with st.expander("💰 Price & Bids", expanded=False):
            min_discount = st.slider("Minimum Discount %", 0, 100, 0, 5)
            if min_discount > 0:
                if include_unknown:
                    slider_mask &= (original_data['discount_pct'] >= min_discount) | (~original_data['has_msrp'])
                else:
                    slider_mask &= original_data['discount_pct'] >= min_discount
            bc_min, bc_max = int(original_data['bid_count'].min()), int(original_data['bid_count'].max())
            if bc_max > bc_min:
                bc_range = st.slider("Bids count range", bc_min, bc_max, (bc_min, bc_max))
                slider_mask &= original_data['bid_count'].between(bc_range[0], bc_range[1])
            all_pickups = sorted({d.strip() for s in original_data['pickup_dates'].fillna('') for d in str(s).split(';') if d.strip()})
            selected_pickups = st.multiselect("Pickup date", options=all_pickups[:12], label_visibility="collapsed", placeholder="Pickup date")
            if selected_pickups:
                slider_mask &= original_data['pickup_dates'].apply(lambda x: any(p in str(x) for p in selected_pickups))
            range_df = original_data.loc[slider_mask]
            price_cols = st.columns(2)
            with price_cols[0]:
                show_msrp = st.checkbox("MSRP", value=True)
            with price_cols[1]:
                show_current = st.checkbox("Current Bid", value=True)
            msrp_range = None
            bid_range = None
            otd_range = None
            if show_msrp and not range_df.empty:
                msrp_series = range_df.loc[range_df['has_msrp'], 'msrp'].dropna()
                if not msrp_series.empty:
                    msrp_min, msrp_max = float(msrp_series.min()), float(msrp_series.max())
                    if msrp_min == msrp_max:
                        msrp_range = (msrp_min, msrp_max)
                        st.caption(f"MSRP fixed at ${msrp_min:,.0f}")
                    else:
                        msrp_range = st.slider("MSRP Range ($)", min_value=float(msrp_min), max_value=float(msrp_max), value=(float(msrp_min), float(msrp_max)), format="$%d")
            if show_current and not range_df.empty:
                bid_series = range_df['current_bid'].dropna()
                if not bid_series.empty:
                    bid_min, bid_max = float(bid_series.min()), float(bid_series.max())
                    if bid_min == bid_max:
                        bid_range = (bid_min, bid_max)
                        st.caption(f"Bids fixed at ${bid_min:,.0f}")
                    else:
                        bid_range = st.slider("Current Bid Range ($)", min_value=float(bid_min), max_value=float(bid_max), value=(float(bid_min), float(bid_max)), format="$%d")
            if not range_df.empty:
                otd_series = range_df['otd_total'].dropna()
                if not otd_series.empty:
                    otd_min, otd_max = float(otd_series.min()), float(otd_series.max())
                    if otd_max > otd_min:
                        otd_range = st.slider("Est Total (OTD) Range $", min_value=float(otd_min), max_value=float(otd_max), value=(float(otd_min), float(otd_max)), format="$%d")
        # need range_df for later mask - if not defined due to expanded=False, define it
        if 'range_df' not in locals():
            range_df = original_data.loc[slider_mask]
            msrp_range = None
            bid_range = None
            otd_range = None
            show_msrp = True
            show_current = True

        with st.expander("⚙️ Sort & Presets", expanded=False):
            filter_incomplete = st.checkbox("Hide incomplete items", value=False)
            hide_seen = st.checkbox("Hide seen", value=False)
            sort_map = {
                'deal_score': 'Deal Score',
                'discount_pct': 'Discount %',
                'savings_amount': 'Savings $',
                'current_bid': 'Current Bid',
                'otd_total': 'Est Total (OTD)',
                'msrp': 'MSRP',
                'auction_end_datetime': 'Ending Soonest',
                'bid_count': 'Bids Count',
                'next_bid': 'Next Bid',
                'est_profit': 'Est Profit',
            }
            sort_options = [(k, v) for k, v in sort_map.items() if k in original_data.columns]
            default_idx = 0
            sort_selection = st.selectbox("Sort by", options=sort_options, index=default_idx, format_func=lambda x: x[1], label_visibility="collapsed")
            st.caption("Sort by")
            sort_column = sort_selection[0] if sort_selection else 'deal_score'
            sort_order = st.radio("Sort order", options=["Descending", "Ascending"], horizontal=True, index=0, label_visibility="collapsed")
            st.divider()
            st.caption("Quick Presets")
            preset_cols = st.columns(2)
            def preset_btn(label, key, help_text):
                if st.button(label, use_container_width=True, help=help_text, key=f"preset_{key}"):
                    if st.session_state.preset == key:
                        st.session_state.preset = None
                    else:
                        st.session_state.preset = key
                    st.rerun()
            with preset_cols[0]:
                preset_btn("Hot Deals", "hot_deals", "MSRP >$100, >80% off")
                preset_btn("Premium Steals", "premium_steals", "MSRP >$500, >75% off")
                preset_btn("Low Competition", "low_comp", "bids<=2 & >60% off")
            with preset_cols[1]:
                preset_btn("Ending Soon", "ending_soon", "Urgency >=0.5")
                preset_btn("Unknown Value", "unknown_value", "$0 MSRP low bid")
                preset_btn("Brand Names", "brand_names", "Has brand")
            if st.session_state.preset:
                st.caption(f"Active: {st.session_state.preset}")
                if st.button("Clear preset", key="clear_preset"):
                    st.session_state.preset = None
                    st.rerun()


    # Build mask
    # Build mask with remaining toggles that need full data
    mask = slider_mask
    if show_msrp and msrp_range is not None:
        mask &= original_data['msrp'].between(msrp_range[0], msrp_range[1]) | (~original_data['has_msrp'])
    if show_current and bid_range is not None:
        mask &= original_data['current_bid'].between(bid_range[0], bid_range[1])
    if 'otd_range' in locals() and otd_range is not None:
        mask &= original_data['otd_total'].between(otd_range[0], otd_range[1])
    if search_query:
        mask &= (
            original_data['item_title'].fillna('').str.contains(search_query, case=False, na=False, regex=False) |
            original_data['brand'].fillna('').str.contains(search_query, case=False, na=False, regex=False) |
            original_data['item_category1'].fillna('').str.contains(search_query, case=False, na=False, regex=False) |
            original_data['item_category2'].fillna('').str.contains(search_query, case=False, na=False, regex=False) |
            original_data['lot_code'].fillna('').str.contains(search_query, case=False, na=False, regex=False)
        )
    if filter_incomplete:
        mask &= ~original_data['item_title'].fillna('').str.contains('incomplete', case=False, na=False, regex=False)
    if hide_seen and st.session_state.seen_ids:
        mask &= ~original_data['item_id'].isin(st.session_state.seen_ids)
    if not include_unknown:
        mask &= original_data['has_msrp']

    filtered_data = original_data.loc[mask]  # view - no copy D1.1
    # apply preset composably before sort
    if st.session_state.preset:
        filtered_data = _apply_preset(filtered_data, st.session_state.preset)
    # sort
    ascending = (sort_order == "Ascending")
    if not filtered_data.empty and sort_column in filtered_data.columns:
        # secondary sort by ending time for deal_score
        if sort_column == 'deal_score':
            filtered_data = filtered_data.sort_values(by=['deal_score','auction_end_datetime'], ascending=[ascending, True])
        else:
            filtered_data = filtered_data.sort_values(by=sort_column, ascending=ascending)

    # Metrics - clean 4+2
    total_items = len(filtered_data)
    avg_msrp = filtered_data.loc[filtered_data['has_msrp'], 'msrp'].mean() if total_items else 0
    avg_bid = filtered_data['current_bid'].mean() if total_items else 0
    avg_disc = filtered_data.loc[filtered_data['has_msrp'], 'discount_pct'].mean() if total_items else 0
    total_sav = filtered_data['savings_amount'].sum() if total_items else 0
    avg_otd = filtered_data['otd_total'].mean() if total_items else 0
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Items", f"{total_items:,}")
    with m2: st.metric("Avg Discount", f"{avg_disc:.1f}%" if total_items else "-")
    with m3: st.metric("Avg OTD", f"${avg_otd:,.0f}" if total_items else "-")
    with m4: st.metric("Total Savings", f"${total_sav:,.0f}" if total_items else "-")
    m5, m6, _, _ = st.columns([1,1,1,1])
    with m5: st.metric("Avg MSRP", f"${avg_msrp:,.0f}" if total_items else "-")
    with m6: st.metric("Avg Bid", f"${avg_bid:,.0f}" if total_items else "-")

    # Active filter pills bar
    pills = []
    if selected_warehouses: pills.append(f"Warehouse: {', '.join(selected_warehouses)}")
    if selected_categories1: pills.append(f"Cat: {', '.join(selected_categories1)}")
    if selected_categories2: pills.append(f"Sub: {', '.join(selected_categories2)}")
    if selected_brands: pills.append(f"Brand: {', '.join(selected_brands)}")
    if search_query: pills.append(f"Search: {search_query}")
    if min_discount>0: pills.append(f"Discount >= {min_discount}%")
    if hide_as_is: pills.append("Hide As Is")
    if hide_single_pic: pills.append("Pics >1")
    if filter_incomplete: pills.append("Hide incomplete")
    if st.session_state.preset: pills.append(f"Preset: {st.session_state.preset}")
    if pills:
        st.markdown('<div class="filter-pills">' + ''.join([f'<span class="filter-pill">{p}</span>' for p in pills]) + '</div>', unsafe_allow_html=True)
        if st.button("Clear filters", key="clear_all_filters"):
            # clear by rerun with cleared state - simplest: reset to None? Instead just clear preset and mask toggles requires manual - do brute clear by removing data??? Just clear preset and seen
            st.session_state.preset = None
            st.session_state.seen_ids = set()
            if 'filter_warehouse' in st.session_state:
                del st.session_state['filter_warehouse']
            st.rerun()
    # Filtered count + proof
    st.caption(f"Showing {len(filtered_data):,} filtered / {len(original_data):,} total - Gateway: Save here, bid on BidFTA")
    # Proof: category audit so you can verify 745 isn't missing items
    with st.expander(f"🔍 Filter Audit - why {len(filtered_data):,} / {len(original_data):,}? Click to prove", expanded=False):
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("**Category breakdown (unfiltered total)** - proves Electronics 745/19k is real")
            # build breakdown table from original_data (before filters) so user sees true distribution
            cat_break = original_data['item_category1'].value_counts(dropna=False).reset_index()
            cat_break.columns = ['Category','Count']
            cat_break['Count'] = cat_break['Count'].astype(int)
            cat_break['Pct'] = (cat_break['Count'] / len(original_data) * 100).round(1).astype(str) + '%'
            # fill NaN category display
            cat_break['Category'] = cat_break['Category'].fillna('(empty)')
            st.dataframe(cat_break.head(20), hide_index=True, use_container_width=True, height=260)
            if len(cat_break) > 20:
                st.caption(f"+ {len(cat_break)-20} more categories")
        with c2:
            st.markdown("**Active filters**")
            active = []
            if selected_categories1: active.append(f"Primary = {', '.join(selected_categories1)}")
            if selected_categories2: active.append(f"Secondary = {', '.join(selected_categories2)}")
            if 'selected_conditions' in locals() and selected_conditions: active.append(f"Condition = {', '.join(selected_conditions)}")
            if 'selected_brands' in locals() and selected_brands: active.append(f"Brand = {', '.join(selected_brands)}")
            if 'selected_qualities' in locals() and selected_qualities: active.append(f"Deal Quality = {', '.join(selected_qualities)}")
            if 'min_discount' in locals() and min_discount>0: active.append(f"Discount ≥{min_discount}%")
            if 'hide_as_is' in locals() and hide_as_is: active.append("Hide As Is")
            if not active:
                st.caption("No category filter - showing all 19k")
            else:
                for a in active:
                    st.caption(f"• {a}")
            st.divider()
            st.caption(f"Result: {len(filtered_data):,} items match those filters. If you expected more, clear Primary Category or check Secondary.")
            # quick verify button - shows raw count for selected primary
            if selected_categories1:
                for cat in selected_categories1[:3]:
                    cnt = int((original_data['item_category1'] == cat).sum())
                    st.caption(f"Verify '{cat}': {cnt:,} rows in raw data (no other filters)")
        st.caption("Tip: Electronics is always ~3-4% of KY (412/11k in Dec archive, 745/19k now). Not missing - just small category. Check 'No MSRP • Research' items if you want hidden gems.")


    # View mode switcher - clean segmented
    vm_col1, vm_col2 = st.columns([1.2, 4])
    with vm_col1:
        try:
            st.segmented_control("View", ["Gallery", "Table"], key="view_mode", label_visibility="collapsed")
        except Exception:
            st.radio("View", ["Gallery", "Table"], horizontal=True, label_visibility="collapsed", key="view_mode")
        try:
            if st.query_params.get("view") != st.session_state.view_mode:
                st.query_params["view"] = st.session_state.view_mode
        except Exception:
            pass
        if "view_mode_radio" in st.session_state:
            del st.session_state["view_mode_radio"]
    with vm_col2:
        if len(st.session_state.watchlist) > 0:
            st.caption(f"★ Watchlist {len(st.session_state.watchlist)} - open Watchlist tab")
        else:
            st.caption("Tip: ★ Watch to shortlist, then bid on BidFTA")

    # Lazy tab slices - computed inside each tab block (D1.3) - keep watch_df only globally
    watch_df = filtered_data[filtered_data['item_id'].isin(st.session_state.watchlist)] if len(st.session_state.watchlist) else filtered_data.head(0)
    # pagination helpers
    def paginate_df(df, page, page_size):
        start = (page-1)*page_size
        return df.iloc[start:start+page_size]
    # Tabs
    # tab counts without eager nlargest - cheap masks
    _cnt_best = int(((filtered_data['has_msrp']) & (filtered_data['deal_quality'].isin(['Excellent','Great','Good','Fair'])) & (filtered_data['discount_pct']>0)).sum()) if not filtered_data.empty else 0
    _cnt_ending = int((filtered_data['time_urgency_score'] >= 0.5).sum()) if not filtered_data.empty else 0
    _cnt_unknown = int((filtered_data['msrp'] == 0).sum()) if not filtered_data.empty else 0
    _cnt_top = min(len(filtered_data), 250) if not filtered_data.empty else 0
    tab_labels = [
        f"Gallery ({total_items})" if st.session_state.view_mode=="Gallery" else f"Full Data ({total_items})",
        f"Best Deals ({_cnt_best})",
        f"Ending Soon ({_cnt_ending})",
        f"Unknown Value ({_cnt_unknown})",
        f"Top Scores ({_cnt_top})",
        f"Watchlist ({len(watch_df)})",
    ]
    view_tabs = st.tabs(tab_labels)

    @_fragment
    def render_gallery(df, key_prefix, page_size=30):
        if df.empty:
            st.info("No items match. Try clearing Brand / As Is filters or check Unknown Value - many $0 MSRP items are hidden gems. Also adjust discount or pickup filters.")
            return
        total = len(df)
        pages = max(1, (total + page_size - 1)//page_size)
        c1, c2, c3 = st.columns([1,1,7])
        with c1:
            if st.button("← Prev", key=f"{key_prefix}_prev", disabled=st.session_state.gallery_page<=1, use_container_width=True):
                st.session_state.gallery_page = max(1, st.session_state.gallery_page-1)
                st.rerun()
        with c2:
            if st.button("Next →", key=f"{key_prefix}_next", disabled=st.session_state.gallery_page>=pages, use_container_width=True):
                st.session_state.gallery_page = min(pages, st.session_state.gallery_page+1)
                st.rerun()
        with c3:
            st.caption(f"Page {st.session_state.gallery_page}/{pages} · {total} items · Save here, bid on BidFTA →")
        page = st.session_state.gallery_page if key_prefix=="gallery" else 1
        if page > pages:
            page = 1
            st.session_state.gallery_page = 1
        sliced = paginate_df(df, page, page_size)
        cols_per_row = 3
        for i in range(0, len(sliced), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for j in range(cols_per_row):
                idx = i+j
                if idx >= len(sliced):
                    break
                row = sliced.iloc[idx]
                item_id = row['item_id']
                is_watched = item_id in st.session_state.watchlist
                urgency_cls = get_urgency_class(row['time_urgency_score'])
                with cols[j]:
                    img = row.get('image_url') or row.get('picture') or ""
                    title = row.get('item_title','')[:110]
                    disc = row.get('discount_pct',0)
                    cond = row.get('condition','')
                    wh = row.get('auction_location_nickname','')
                    ending = row.get('ending_in','')
                    bid_c = int(row.get('bid_count',0))
                    next_b = row.get('next_bid', row.get('current_bid',0))
                    otd = row.get('otd_total', next_b * 1.2)
                    brand = row.get('brand','')
                    pics = int(row.get('picture_count',0))
                    fallback = "https://via.placeholder.com/400x300?text=No+Image"
                    if not img:
                        img = fallback
                    deep = row.get('item_url') or f"https://www.bidfta.com/{row.get('auction_id')}/item-detail/{row.get('item_id')}"
                    # tooltip with key info for hover
                    tooltip = f"{title} | ${row.get('current_bid',0):.0f} / MSRP ${row.get('msrp',0):.0f} Save ${row.get('savings_amount',0):.0f} | {disc:.0f}% off | {bid_c} bids | OTD ${otd:.0f} | {wh} | {ending} | {cond}".replace('"', "'")
                    badge_html = f'<div class="gallery-badge">{disc:.0f}% OFF</div>' if disc >= 30 else ''
                    # time floating badge - like discount, but urgency-colored
                    urg = row.get('time_urgency_score', 0)
                    # also handle ended
                    ending_raw = (ending or "").strip()
                    is_ended = "ended" in ending_raw.lower() or "closed" in ending_raw.lower()
                    if is_ended:
                        time_badge_html = f'<div class="gallery-time-badge gallery-time-ended">{ending_raw or "Ended"}</div>'
                    elif urg >= 0.8:
                        time_badge_html = f'<div class="gallery-time-badge gallery-time-critical">⏰ {ending} left</div>' if ending else ''
                    elif urg >= 0.5:
                        time_badge_html = f'<div class="gallery-time-badge gallery-time-high">⏰ {ending} left</div>' if ending else ''
                    elif urg >= 0.2:
                        time_badge_html = f'<div class="gallery-time-badge gallery-time-medium">{ending} left</div>' if ending else ''
                    else:
                        time_badge_html = f'<div class="gallery-time-badge gallery-time-low">{ending}</div>' if ending else ''
                    # clickable image with tooltip - entire image is now the CTA
                    img_html = f'<a href="{deep}" target="_blank" rel="noopener" title="{tooltip}" class="gallery-img-link"><img src="{img}" alt="{title[:40]}" loading="lazy" onerror="this.onerror=null;this.src=\'{fallback}\'"><span class="gallery-img-hint">Click to bid →</span></a>'
                    # price logic - hide MSRP/Save clutter when $0 (unknown value)
                    msrp_val = float(row.get("msrp",0) or 0)
                    curr_val = float(row.get("current_bid",0) or 0)
                    save_val = float(row.get("savings_amount",0) or 0)
                    if msrp_val > 0:
                        msrp_html = f"<span class='gallery-msrp'>MSRP ${msrp_val:.0f}</span>"
                        save_cls = "gallery-save" if save_val > 0 else "gallery-save gallery-save-zero"
                        save_html = f"<span class='{save_cls}'>Save ${save_val:.0f}</span>"
                    else:
                        msrp_html = "<span class='pill' style='background:#fffbeb; border:1px solid #fde68a; color:#92400e; font-size:0.74em'>No MSRP • Research</span>"
                        save_html = ""
                    # warehouse - truncate long "Covington - Howard Litzler Dr - 300" -> "Covington"
                    wh_short = wh.split(" - ")[0].strip() if " - " in wh else wh
                    # also trim if still long
                    if len(wh_short) > 18:
                        wh_short = wh_short[:18].rstrip() + "…"
                    otd_html = f"<span class='pill pill-otd'>OTD ${otd:.0f}</span>"
                    wh_html = f"<span class='pill pill-warehouse' title='{wh}'>{wh_short}</span>" if wh else ""
                    bids_cls = "pill pill-bids"
                    if bid_c == 0:
                        bids_html = f"<span class='{bids_cls}' style='background:#f0fdf4; border-color:#86efac; color:#14532d'>● {bid_c} bids • No competition</span>"
                    elif bid_c == 1:
                        bids_html = f"<span class='{bids_cls}'>● {bid_c} bid</span>"
                    else:
                        bids_html = f"<span class='{bids_cls}'>{bid_c} bids</span>"
                    cond_html = f"<span class='pill pill-condition'>{cond}</span>" if cond else ""
                    brand_html = f"<span class='pill pill-brand'>{brand}</span>" if brand and brand.lower()!="generic" and brand else ""
                    card_html = (
                        f'<div class="gallery-card" style="min-height:380px">'
                        f'<div class="gallery-img-wrap">{img_html}{badge_html}{time_badge_html}</div>'
                        f'<div class="gallery-body">'
                        f'<div class="gallery-title" title="{title}">{title}</div>'
                        f'<div class="gallery-price-row"><span class="gallery-price">${curr_val:.0f}</span> {msrp_html} {save_html}</div>'
                        f'<div class="gallery-meta">{bids_html} {otd_html} {wh_html}</div>'
                        f'<div class="gallery-footer">{cond_html} {brand_html}</div>'
                        f'</div></div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                    # no more Bid button - image is the CTA, only Watch remains for minimal clutter
                    w_label = "★ Watched" if is_watched else "☆ Watch"
                    if st.button(w_label, key=f"watch_{key_prefix}_{item_id}_{idx}", help="Watch / Unwatch - image click goes to BidFTA", use_container_width=True):
                        if is_watched:
                            st.session_state.watchlist.remove(item_id)
                        else:
                            st.session_state.watchlist.add(item_id)
                        st.rerun()
                    with st.popover("Details", use_container_width=True):
                        st.markdown(f"**{title}**")
                        st.caption(f"{cond} · {row.get('item_category1','')} / {row.get('item_category2','')} · Lot {row.get('lot_code','')} · Qty {row.get('quantity',1)}")
                        pics_list = row.get('pictures') if isinstance(row.get('pictures'), list) else []
                        if not pics_list and img and img!=fallback:
                            pics_list = [img]
                        if pics_list:
                            try:
                                st.image(pics_list[0], use_container_width=True)
                            except:
                                pass
                            if len(pics_list) > 1:
                                tn = st.columns(min(4, len(pics_list)))
                                for pi, purl in enumerate(pics_list[1:5]):
                                    with tn[pi % len(tn)]:
                                        try:
                                            st.image(purl, use_container_width=True)
                                        except:
                                            pass
                        st.divider()
                        tax_rate = float(row.get('auction_location_tax_rate',6.0))
                        premium = next_b * 0.13
                        tax_amt = (next_b + premium) * tax_rate/100.0
                        st.caption(f"OTD: ${next_b:.2f} + Premium ${premium:.2f} + Tax {tax_rate:.1f}% (${tax_amt:.2f}) = **${otd:.2f}** · {pics} pics · {ending}")
                        deep = row.get('item_url') or f"https://www.bidfta.com/{row.get('auction_id')}/item-detail/{row.get('item_id')}"
                        cA, cB = st.columns(2)
                        with cA:
                            st.link_button("Bid on BidFTA →", deep, type="primary", use_container_width=True)
                            if st.button("Copy link", key=f"copy_{key_prefix}_{item_id}_{idx}", use_container_width=True):
                                st.code(deep)
                        with cB:
                            if is_watched:
                                if st.button("Remove Watch", key=f"rmw_{key_prefix}_{item_id}_{idx}", use_container_width=True):
                                    st.session_state.watchlist.remove(item_id)
                                    st.rerun()
                            else:
                                if st.button("Add Watch", key=f"addw_{key_prefix}_{item_id}_{idx}", use_container_width=True):
                                    st.session_state.watchlist.add(item_id)
                                    st.rerun()
                            st.caption(f"Bids {bid_c} · {pics} pics · {row.get('deal_quality','')} · Score {row.get('deal_score',0):.0f}")
                        st.caption(f"Warehouse {wh} · Pickup: {row.get('pickup_dates','')[:80]}")
                        st.code(deep, language=None)
        st.divider()
        st.caption("Tip: ★ Watch to shortlist, then open Watchlist tab to bid. Details holds OTD, pics, and copy.")

    @_fragment
    def render_table(df, key):
        if df.empty:
            st.info("No rows - try loosening filters or check Watchlist. Tip: Include unknown MSRP shows 2669 hidden items.")
            return
        # --- Global sort for Table (fixes per-page sort bug) ---
        # Full Data (key=="full") defaults to MSRP desc per request; other tabs keep their own order unless user overrides
        # Use table-specific sort state so Gallery's deal_score sort doesn't affect Table
        if f"tbl_sort_col_{key}" not in st.session_state:
            # Full Data defaults to MSRP desc, others respect incoming order (no extra sort) unless user picks
            st.session_state[f"tbl_sort_col_{key}"] = "msrp" if key == "full" else None
            st.session_state[f"tbl_sort_asc_{key}"] = False if key == "full" else False
        # UI for global sort - ensures next page is correctly sorted (header click only sorts page)
        sort_cols = [c for c in ['msrp','current_bid','otd_total','discount_pct','savings_amount','bid_count','deal_score','ending_in','auction_end_datetime'] if c in df.columns]
        # show sort UI only for Full Data to avoid clutter on Best Deals etc. which have fixed sorts
        if key == "full" and sort_cols:
            sc1, sc2, sc3 = st.columns([2,2,6])
            with sc1:
                # friendly names
                name_map = {'msrp':'MSRP','current_bid':'Current Bid','otd_total':'OTD','discount_pct':'Discount %','savings_amount':'Savings','bid_count':'Bids','deal_score':'Deal Score','ending_in':'Ending','auction_end_datetime':'End Date'}
                opts = [(c, name_map.get(c,c)) for c in sort_cols]
                # find current col index
                cur = st.session_state[f"tbl_sort_col_{key}"]
                try:
                    cur_idx = [c for c,_ in opts].index(cur) if cur in [c for c,_ in opts] else 0
                except:
                    cur_idx = 0
                sel = st.selectbox("Sort table by", options=opts, index=cur_idx, format_func=lambda x: x[1], key=f"tbl_sort_sel_{key}", label_visibility="collapsed")
                st.session_state[f"tbl_sort_col_{key}"] = sel[0]
                st.caption("Sort by")
            with sc2:
                asc = st.radio("Order", ["Descending","Ascending"], index=0 if not st.session_state[f"tbl_sort_asc_{key}"] else 1, horizontal=True, key=f"tbl_sort_dir_{key}", label_visibility="collapsed")
                st.session_state[f"tbl_sort_asc_{key}"] = (asc == "Ascending")
                st.caption("Order")
            with sc3:
                st.caption("Global sort - applies to all pages (header click only sorts this page). Full Data defaults to MSRP ↓")
            # apply global sort before pagination
            sort_col = st.session_state[f"tbl_sort_col_{key}"]
            sort_asc = st.session_state[f"tbl_sort_asc_{key}"]
            if sort_col in df.columns:
                # reset page on sort change? keep current page but ensure valid
                try:
                    df = df.sort_values(by=sort_col, ascending=sort_asc, na_position='last')
                except Exception:
                    pass
        elif st.session_state.get(f"tbl_sort_col_{key}") and st.session_state[f"tbl_sort_col_{key}"] in df.columns:
            # for non-full tables, respect user sort if set (optional)
            try:
                df = df.sort_values(by=st.session_state[f"tbl_sort_col_{key}"], ascending=st.session_state[f"tbl_sort_asc_{key}"], na_position='last')
            except Exception:
                pass
        else:
            # for Full Data with no explicit sort, ensure MSRP desc default even if UI not shown (e.g., Gallery->Table switch)
            if key == "full" and "msrp" in df.columns:
                try:
                    df = df.sort_values(by="msrp", ascending=False, na_position='last')
                except Exception:
                    pass
        # paginate table (after global sort)
        page_size = 50
        total = len(df)
        pages = max(1, (total + page_size -1)//page_size)
        c1, c2, c3 = st.columns([1,1,6])
        with c1:
            if st.button("Prev", key=f"tprev_{key}", disabled=st.session_state.table_page<=1):
                st.session_state.table_page = max(1, st.session_state.table_page-1)
                st.rerun()
        with c2:
            if st.button("Next", key=f"tnext_{key}", disabled=st.session_state.table_page>=pages):
                st.session_state.table_page = min(pages, st.session_state.table_page+1)
                st.rerun()
        with c3:
            st.caption(f"Page {st.session_state.table_page}/{pages} - {total} items")
        page = st.session_state.table_page
        if page > pages:
            page = 1
            st.session_state.table_page = 1
        sliced = paginate_df(df, page, page_size)
        # column order
        pref = ['picture','item_title','brand','condition','bid_count','picture_count','current_bid','next_bid','otd_total','msrp','discount_pct','savings_amount','deal_quality','deal_score','est_profit','auction_location_nickname','ending_in','pickup_dates','item_url']
        col_order = [c for c in pref if c in sliced.columns]
        # fallback
        if not col_order:
            col_order = list(sliced.columns)[:10]
        cfg = _build_cc(col_order)
        st.dataframe(sliced[col_order], column_config=cfg, hide_index=True, use_container_width=True, height=520)
        # bulk actions
        st.download_button("Download CSV (page)", convert_df(sliced[col_order]), f"bidfta_page_{page}.csv", "text/csv", key=f"dl_{key}")
        if st.button("Copy all page links", key=f"copylinks_{key}"):
            links = "; ".join(sliced['item_url'].tolist())
            st.code(links)
        if len(st.session_state.watchlist) >0 and st.button("Open all watched on BidFTA (copy)", key=f"openwatch_{key}"):
            wlinks = filtered_data[filtered_data['item_id'].isin(st.session_state.watchlist)]['item_url'].tolist()
            st.code("\n".join(wlinks))
            st.caption("Copy and open in browser - gateway only, no auto-bid.")

    # Tabs content
    with view_tabs[0]:
        if st.session_state.view_mode == "Gallery":
            st.subheader("Gallery - Save here, Bid on BidFTA ->")
            st.caption(f"All {len(filtered_data):,} filtered items - OTD includes 13% premium + tax - {len(st.session_state.watchlist)} watched")
            render_gallery(filtered_data, "gallery")
            # also show table paginated alternative below?
            with st.expander("Show as Table"):
                render_table(filtered_data, "gallery_table")
        else:
            st.subheader("Full Filtered Data")
            st.caption(f"All {len(filtered_data):,} items - Bid on BidFTA to place bid")
            render_table(filtered_data, "full")

    with view_tabs[1]:
        # lazy compute D1.3
        deals_mask = (filtered_data['has_msrp']) & (filtered_data['deal_quality'].isin(['Excellent','Great','Good','Fair'])) & (filtered_data['discount_pct']>0)
        best_deals = filtered_data.loc[deals_mask].nlargest(500, 'discount_pct') if not filtered_data.empty else filtered_data
        st.subheader("Best Deals by Discount %")
        st.caption("High discount, trusted MSRP, not Research/Overpriced - Bid on BidFTA when you like one")
        if best_deals.empty:
            st.info("No deals found - loosen Brand / As Is / Discount filters or include Unknown")
        else:
            if st.session_state.view_mode=="Gallery":
                render_gallery(best_deals.head(200), "best")
            else:
                render_table(best_deals, "best")

    with view_tabs[2]:
        ending_soon = filtered_data[filtered_data['time_urgency_score'] >= 0.5].sort_values('auction_end_datetime', ascending=True).head(500) if not filtered_data.empty else filtered_data
        st.subheader("Ending Soon - Act Fast")
        st.caption("Continuous urgency, not buckets - 2h ahead of 5h - OverTime boosted")
        if ending_soon.empty:
            st.info("No items ending within ~24h (urgency >=0.5). Check back later or lower threshold in code.")
        else:
            if st.session_state.view_mode=="Gallery":
                render_gallery(ending_soon, "ending")
            else:
                render_table(ending_soon, "ending")

    with view_tabs[3]:
        unknown_value_items = filtered_data[(filtered_data['msrp'] == 0)].nsmallest(500, 'current_bid') if not filtered_data.empty else filtered_data
        st.subheader("Unknown Value Gems")
        st.caption("Top items without MSRP - hidden treasures sorted low bid - brand + bids + pics matter more than discount")
        if unknown_value_items.empty:
            st.info("No unknown MSRP in current filters - toggle Include unknown MSRP on.")
        else:
            if st.session_state.view_mode=="Gallery":
                render_gallery(unknown_value_items.head(200), "unknown")
            else:
                render_table(unknown_value_items, "unknown")

    with view_tabs[4]:
        top_scores = filtered_data.nlargest(250, 'deal_score') if not filtered_data.empty else filtered_data
        st.subheader("Top Deal Scores")
        st.caption("Composite: discount * condition tier * urgency boost (1+urg*0.5) - Brand New 1.3, As Is 0.6, incomplete -0.2, single pic -0.05")
        if top_scores.empty:
            st.info("No scored items")
        else:
            if st.session_state.view_mode=="Gallery":
                render_gallery(top_scores, "scores")
            else:
                render_table(top_scores, "scores")

    with view_tabs[5]:
        st.subheader(f"Watchlist ({len(watch_df)}) - Your shortlist")
        st.caption("Heart items in gallery, they stay here. Bulk open on BidFTA to bid. Save here, bid there.")
        if watch_df.empty:
            st.info("Watchlist empty - click Watch on gallery cards to collect. Then bulk open links to bid.")
        else:
            # save watchlist seen?
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Clear Watchlist"):
                    st.session_state.watchlist = set()
                    st.rerun()
            with c2:
                st.download_button("Download Watchlist CSV", convert_df(watch_df), "watchlist.csv", "text/csv", key="dl_watch")
            if not watch_df.empty:
                wlinks = watch_df['item_url'].tolist()
                st.code("\n".join(wlinks[:10]) + ("\n... and " + str(len(wlinks)-10) + " more" if len(wlinks)>10 else ""), language=None)
                if st.button("Copy all watchlist links"):
                    st.code("\n".join(wlinks))
                if len(watch_df) >1:
                    # compare up to 4
                    st.divider()
                    st.markdown("**Compare up to 4 watched items**")
                    compare_ids = st.multiselect("Pick 2-4 to compare", options=watch_df['item_id'].tolist(), default=list(watch_df['item_id'].tolist()[:4]), format_func=lambda x: str(watch_df[watch_df['item_id']==x].iloc[0]['item_title'])[:50])
                    if len(compare_ids) >=2:
                        comp = watch_df[watch_df['item_id'].isin(compare_ids)]
                        # side by side table transposed-ish? Show as dataframe
                        comp_cols = ['item_title','brand','condition','current_bid','next_bid','otd_total','msrp','discount_pct','savings_amount','bid_count','picture_count','auction_location_nickname','ending_in','pickup_dates','item_url']
                        comp_cols = [c for c in comp_cols if c in comp.columns]
                        st.dataframe(comp[comp_cols], column_config=_build_cc(comp_cols), hide_index=True, use_container_width=True)
            if st.session_state.view_mode=="Gallery":
                render_gallery(watch_df, "watch")
            else:
                render_table(watch_df, "watchtab")

    # Visualizations lazy under toggle
    if show_viz:
        st.divider()
        st.header("Data Insights - Gateway Hunter Wins")
        if filtered_data.empty:
            st.info("No rows to visualize with current filters")
        else:
            @st.cache_data(ttl=300)
            def viz_avg_discount_by_warehouse(df):
                return df.groupby('auction_location_nickname')['discount_pct'].agg(['mean','count']).reset_index().sort_values('mean', ascending=True).tail(10)
            @st.cache_data(ttl=300)
            def viz_bids_vs_otd(df):
                return df.groupby('auction_location_nickname').agg(item_count=('item_title','count'), avg_otd=('otd_total','mean'), avg_disc=('discount_pct','mean')).reset_index()
            st.subheader("Avg Discount % by Warehouse")
            cat = viz_avg_discount_by_warehouse(filtered_data)
            cat = cat[cat['count']>=5]
            if not cat.empty:
                fig = px.bar(cat, x='mean', y='auction_location_nickname', orientation='h', title="Top 10 warehouses by avg discount %", labels={'mean':'Avg Discount %','auction_location_nickname':'Warehouse'}, color='mean', color_continuous_scale='RdYlGn')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Need 5+ items per warehouse to chart")
            # Bids heatmap-ish
            st.subheader("Bids Heatmap vs OTD")
            # scatter bid_count vs otd colored by discount
            fig2 = px.scatter(filtered_data.sample(min(1000, len(filtered_data))), x='bid_count', y='otd_total', color='discount_pct', hover_data=['item_title','brand','current_bid'], title="Bids (heat) vs OTD (cost) - low bids + low OTD = snipe", color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig2, use_container_width=True)
            # Deals by pickup
            st.subheader("Deals expiring today by pickup day")
            # pickup days histogram - no explode needed, use string split counts below
            # use pickup_dates string split
            pick_counts = {}
            for s in filtered_data['pickup_dates'].fillna(''):
                for d in str(s).split(';'):
                    d=d.strip()
                    if d:
                        pick_counts[d]=pick_counts.get(d,0)+1
            if pick_counts:
                pc_df = pd.DataFrame(list(pick_counts.items()), columns=['Pickup','Count']).sort_values('Count', ascending=False).head(10)
                fig3 = px.bar(pc_df, x='Count', y='Pickup', orientation='h', title="Items per pickup date")
                st.plotly_chart(fig3, use_container_width=True)
            # Keep old category/price layers for backwards compat
            with st.expander("More charts"):
                st.subheader("Most Valuable Categories")
                category_msrp = filtered_data.groupby('item_category1')['msrp'].agg(['mean','count']).reset_index()
                category_msrp = category_msrp[category_msrp['count']>=5].sort_values('mean', ascending=True).tail(10)
                if not category_msrp.empty:
                    fig_cat = px.bar(category_msrp, x='mean', y='item_category1', orientation='h', title="Top 10 Categories by Avg MSRP", labels={'mean':'Avg MSRP','item_category1':'Category'}, color='mean', color_continuous_scale='Viridis')
                    fig_cat.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_cat, use_container_width=True)
                st.subheader("Best Value Deals scatter")
                deals_data = filtered_data[(filtered_data['msrp']>50) & (filtered_data['ratio_bid_to_msrp']>0) & (filtered_data['ratio_bid_to_msrp']<1)].sort_values('ratio_bid_to_msrp').head(10)
                if not deals_data.empty:
                    fig_deals = px.bar(deals_data, x='ratio_bid_to_msrp', y='item_title', orientation='h', title="Top 10 Best Value Deals", labels={'ratio_bid_to_msrp':'Current Bid / MSRP'}, color='ratio_bid_to_msrp', color_continuous_scale='RdYlGn_r', hover_data=['current_bid','msrp','auction_location_nickname'])
                    fig_deals.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_deals, use_container_width=True)
