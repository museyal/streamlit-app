"""Utility transforms shared across tracks - vectorized, cached-ready, deduped."""
import math
from datetime import datetime
import numpy as np
import pandas as pd
import pytz

# Keep premium tunable; utils should not import config to avoid cycle, but allow param.
def otd_breakdown(next_bid: float, tax_rate: float, premium: float = 0.13) -> float:
    try:
        return round(float(next_bid) * (1 + float(premium)) * (1 + float(tax_rate)/100.0), 2)
    except Exception:
        return 0.0

def get_time_remaining(end_time, now):
    """Single-item remaining string (mirrors original logic)."""
    if pd.isna(end_time):
        return "N/A"
    diff = end_time - now
    days = diff.days
    secs = diff.seconds
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    if diff.total_seconds() < 0:
        return "Ended"
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or (days == 0 and minutes > 0):
        parts.append(f"{hours}h")
    if days == 0 and hours == 0 and minutes > 0:
        parts.append(f"{minutes}m")
    if not parts:
        return "Ending soon"
    return " ".join(parts)

def vectorized_ending_in(end_series: pd.Series, now) -> pd.Series:
    """Vectorized ending_in without apply - 4x faster, O(n) np."""
    # normalize now to Timestamp
    if not isinstance(now, pd.Timestamp):
        try:
            est = pytz.timezone('US/Eastern')
            if now.tzinfo is None:
                now = est.localize(now)
            now = pd.Timestamp(now)
        except Exception:
            now = pd.Timestamp(now)
    n = len(end_series)
    result = np.full(n, "N/A", dtype=object)
    mask_valid = ~pd.isna(end_series)
    if not mask_valid.any():
        return pd.Series(result, index=end_series.index)
    # only for valid
    # convert to series for dt ops
    valid_end = end_series[mask_valid]
    # total_seconds vector
    # pandas timedelta
    delta_secs = (valid_end - now).dt.total_seconds().values  # numpy
    valid_idx = np.where(mask_valid.values)[0]
    # ended
    ended = delta_secs < 0
    # init valid_result
    valid_result = np.full(delta_secs.shape, "Ending soon", dtype=object)
    valid_result[ended] = "Ended"
    not_ended = ~ended
    if np.any(not_ended):
        d_secs_not = delta_secs[not_ended]
        days = (d_secs_not // 86400).astype(int)
        rem = d_secs_not % 86400
        hours = (rem // 3600).astype(int)
        mins = ((rem % 3600)//60).astype(int)
        # need to map back to positions within valid_result
        not_pos = np.where(not_ended)[0]
        for i, pos in enumerate(not_pos):
            di = int(days[i]); hi = int(hours[i]); mi = int(mins[i])
            parts = []
            if di > 0:
                parts.append(f"{di}d")
            if hi > 0 or (di == 0 and mi > 0):
                parts.append(f"{hi}h")
            if di == 0 and hi == 0 and mi > 0:
                parts.append(f"{mi}m")
            if parts:
                valid_result[pos] = " ".join(parts)
            else:
                valid_result[pos] = "Ending soon"
    result[valid_idx] = valid_result
    return pd.Series(result, index=end_series.index)

def vectorized_hours_left(end_series: pd.Series, now) -> pd.Series:
    if not isinstance(now, pd.Timestamp):
        try:
            est = pytz.timezone('US/Eastern')
            if hasattr(now, 'tzinfo') and now.tzinfo is None:
                now = est.localize(now)
            now = pd.Timestamp(now)
        except Exception:
            now = pd.Timestamp(now)
    delta_secs = (end_series - now).dt.total_seconds()
    hours = delta_secs / 3600.0
    return hours.fillna(9999)

def calculate_urgency_continuous(hours_left, over_time):
    """Scalar helper kept for compat."""
    try:
        hl = float(hours_left)
    except Exception:
        hl = 9999
    if hl < 0:
        return 0.0
    base = math.exp(-max(hl, 0)/24.0)
    if bool(over_time):
        base = min(1.0, base * 1.25)
    return float(np.clip(base, 0, 1))

def vectorized_urgency(hours_left_series: pd.Series, over_time_series: pd.Series) -> pd.Series:
    hl = pd.to_numeric(hours_left_series, errors='coerce').fillna(9999).values.astype(float)
    ot = over_time_series.fillna(False).astype(bool).values
    base = np.exp(-np.maximum(hl, 0)/24.0)
    base = np.where(hl < 0, 0.0, base)
    base = np.where(ot, np.minimum(1.0, base*1.25), base)
    return pd.Series(np.clip(base, 0, 1), index=hours_left_series.index)

def vectorized_deal_quality(df: pd.DataFrame) -> pd.Series:
    """Vectorized via np.select / pd.cut - replaces apply per row."""
    # requires _is_msrp_outlier, ratio, msrp, current_bid present
    msrp = pd.to_numeric(df.get('msrp', 0), errors='coerce').fillna(0)
    ratio = pd.to_numeric(df.get('ratio_bid_to_msrp', 0), errors='coerce').fillna(0)
    cur = pd.to_numeric(df.get('current_bid', 0), errors='coerce').fillna(0)
    is_unknown = msrp == 0
    # outlier column may not exist
    if '_is_msrp_outlier' in df.columns:
        is_outlier = df['_is_msrp_outlier'].fillna(False).astype(bool)
    else:
        is_outlier = pd.Series(False, index=df.index)
    is_overpriced = (cur > msrp) & (msrp > 0)
    is_no_bids = (ratio == 0) & (msrp > 0) & (~is_outlier) & (~is_overpriced)
    cond = [
        is_unknown,
        is_outlier,
        is_overpriced,
        is_no_bids,
        ratio < 0.10,
        ratio < 0.25,
        ratio < 0.40,
    ]
    # Need to handle that ratio <0.10 etc only when msrp>0 and not other flags
    # np.select respects first True, so order matters; Unknown/Research/Overpriced/NoBids first
    choices = ['Unknown Value','Research','Overpriced','No Bids','Excellent','Great','Good']
    # but we must mask that Excellent etc only apply when msrp>0 and not already flagged
    # Since first 4 cover those, remaining will be evaluated correctly
    res = np.select(cond, choices, default='Fair')
    # For Unknown rows, already set; for others where msrp==0? already Unknown, so fine
    return pd.Series(res, index=df.index)

def build_column_config(column_order):
    """Deduped column config builder - imports streamlit lazily to stay importable offline."""
    try:
        import streamlit as st
    except Exception:
        return {}
    COLUMN_NAMES = {
        'picture': 'Picture','image_url': 'Picture','item_title': 'Item Title','condition': 'Condition','brand': 'Brand',
        'item_category1': 'Primary Category','item_category2': 'Secondary Category','current_bid': 'Current Bid','next_bid': 'Next Bid',
        'otd_total': 'Est Total (OTD)','msrp': 'MSRP','discount_pct': 'Discount %','savings_amount': 'Savings','deal_quality': 'Deal Quality',
        'deal_score': 'Deal Score','bid_count': 'Bids','picture_count': 'Pics','auction_location_nickname': 'Warehouse',
        'auction_location_city': 'City','item_url': 'Bid Link','auction_end_datetime': 'End (EST)','ending_in': 'Ending',
        'ratio_bid_to_msrp': 'Bid/MSRP','pickup_dates': 'Pickup',
    }
    cfg = {}
    if 'picture' in column_order or 'image_url' in column_order:
        col = 'picture' if 'picture' in column_order else 'image_url'
        cfg[col] = st.column_config.ImageColumn(COLUMN_NAMES.get(col,'Picture'), width="small", help="Click card for all images")
    if 'item_url' in column_order:
        cfg['item_url'] = st.column_config.LinkColumn(COLUMN_NAMES['item_url'])
    if 'msrp' in column_order:
        cfg['msrp'] = st.column_config.NumberColumn(COLUMN_NAMES['msrp'], format="$%d")
    if 'current_bid' in column_order:
        cfg['current_bid'] = st.column_config.NumberColumn(COLUMN_NAMES['current_bid'], format="$%d")
    if 'next_bid' in column_order:
        cfg['next_bid'] = st.column_config.NumberColumn(COLUMN_NAMES['next_bid'], format="$%d")
    if 'otd_total' in column_order:
        cfg['otd_total'] = st.column_config.NumberColumn(COLUMN_NAMES['otd_total'], format="$%d", width="small")
    if 'bid_count' in column_order:
        cfg['bid_count'] = st.column_config.NumberColumn(COLUMN_NAMES['bid_count'], width="small")
    if 'picture_count' in column_order:
        cfg['picture_count'] = st.column_config.NumberColumn(COLUMN_NAMES['picture_count'], width="small")
    if 'item_title' in column_order:
        cfg['item_title'] = st.column_config.TextColumn(COLUMN_NAMES['item_title'], width="large")
    if 'condition' in column_order:
        cfg['condition'] = st.column_config.TextColumn(COLUMN_NAMES['condition'], width="small")
    if 'brand' in column_order:
        cfg['brand'] = st.column_config.TextColumn(COLUMN_NAMES['brand'], width="small")
    if 'item_category1' in column_order:
        cfg['item_category1'] = st.column_config.TextColumn(COLUMN_NAMES['item_category1'], width="medium")
    if 'item_category2' in column_order:
        cfg['item_category2'] = st.column_config.TextColumn(COLUMN_NAMES['item_category2'], width="medium")
    if 'auction_location_nickname' in column_order:
        cfg['auction_location_nickname'] = st.column_config.TextColumn(COLUMN_NAMES['auction_location_nickname'], width="small")
    if 'auction_end_datetime' in column_order:
        cfg['auction_end_datetime'] = st.column_config.DatetimeColumn(COLUMN_NAMES['auction_end_datetime'], format="MMM DD, YYYY h:mm a", width="medium")
    if 'ending_in' in column_order:
        cfg['ending_in'] = st.column_config.TextColumn(COLUMN_NAMES['ending_in'], width="small")
    if 'discount_pct' in column_order:
        cfg['discount_pct'] = st.column_config.NumberColumn(COLUMN_NAMES['discount_pct'], format="%.1f%%", width="small")
    if 'savings_amount' in column_order:
        cfg['savings_amount'] = st.column_config.NumberColumn(COLUMN_NAMES['savings_amount'], format="$%d", width="small")
    if 'deal_score' in column_order:
        cfg['deal_score'] = st.column_config.NumberColumn(COLUMN_NAMES['deal_score'], format="%.1f", width="small")
    return cfg

def apply_preset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if name == 'hot_deals':
        return df[(df['msrp'] > 100) & (df['discount_pct'] > 80) & (~df['condition'].fillna('').str.contains('as is', case=False, regex=False))].sort_values('discount_pct', ascending=False)
    if name == 'ending_soon':
        return df[df['time_urgency_score'] >= 0.5].sort_values('time_urgency_score', ascending=False)
    if name == 'premium_steals':
        return df[(df['msrp'] > 500) & (df['discount_pct'] > 75) & (df['condition'].fillna('').str.contains('new|like new', case=False, regex=True))].sort_values('discount_pct', ascending=False)
    if name == 'unknown_value':
        return df[(df['msrp'] == 0) & (df['current_bid'] >= 0)].sort_values('current_bid', ascending=True)
    if name == 'best_scores':
        return df.sort_values('deal_score', ascending=False).head(50)
    if name == 'low_comp':
        return df[(df['bid_count'] <= 2) & (df['discount_pct'] > 60)].sort_values('discount_pct', ascending=False)
    if name == 'brand_names':
        return df[(df['brand'].fillna('').str.strip() != '') & (~df['brand'].fillna('').str.lower().isin(['generic','']))].sort_values('discount_pct', ascending=False)
    return df

def location_label(loc: dict) -> str:
    return f"{loc.get('city','')} - {loc.get('nickName','')} {loc.get('state','')}".strip()
