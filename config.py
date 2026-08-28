"""
Shared configuration for BidFTA gateway.
Single source for endpoints, headers, premiums, limits, and schema.
"""

# BidFTA buyer premium (typical). Keep tunable - real fee varies by auction type.
BUYER_PREMIUM = 0.13
# Tuned for 800+ auctions (KY+OH): 8 workers ~2x faster than 4, still under 429 threshold with backoff+jitter.
# Pool sized for burst: 40 conns handles 8 workers x 5 keep-alive each. UI exposes 4/8/12 selector.
MAX_WORKERS = 8
POOL_MAXSIZE = 40
POOL_CONNECTIONS = 20
REQUEST_TIMEOUT = (5, 15)
RETRY_TOTAL = 3
RETRY_BACKOFF = 0.5
RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]

# Fast-mode guard for huge pulls (KY+OH ~832 auctions). None = all, else cap after sort by end date.
DEFAULT_MAX_AUCTIONS = None  # keep all by default; UI offers 200/400/All
FAST_AUCTION_CAPS = [200, 400, None]

BASE_URL = "https://auction.bidfta.io"
ORIGIN = "https://www.bidfta.com"
REFERER = "https://www.bidfta.com/"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": ORIGIN,
    "referer": REFERER,
    "user-agent": "Mozilla/5.0 (compatible; BidFTA-Gateway/1.0)",
}

# Toggle gallery flag
ENABLE_GALLERY = True
ENABLE_LIGHTBOX = True

# Unified item schema - 25 cols + derived. Used by both scrape.py and lit_app.py.
# Typed for docs - runtime uses list.
from typing import TypedDict, List, Optional

class ItemRow(TypedDict, total=False):
    auction_id: int
    auction_number: str
    auction_title: str
    auction_category: str
    auction_start_datetime: str
    auction_end_datetime: str
    auction_location_id: int
    auction_location_nickname: str
    auction_location_address: str
    auction_location_city: str
    auction_location_state: str
    auction_location_zip: str
    auction_location_tax_rate: float
    pickup_dates: str
    pickup_dates_list: List[str]
    item_id: int
    lot_code: str
    current_bid: float
    next_bid: float
    msrp: float
    condition: str
    brand: str
    item_title: str
    item_category1: str
    item_category2: str
    quantity: float
    initial_quantity: float
    bid_count: int
    pallet_lot: bool
    pallet_auction: bool
    image_url: str
    pictures: List[str]
    picture_count: int
    over_time: bool
    hours_remaining: int
    ratio_bid_to_msrp: float
    item_url: str
    # derived
    discount_pct: float
    savings_amount: float
    deal_quality: str
    deal_score: float
    time_urgency_score: float
    ending_in: str
    otd_total: float

ITEM_SCHEMA = [
    "auction_id",
    "auction_number",
    "auction_title",
    "auction_category",
    "auction_start_datetime",
    "auction_end_datetime",
    "auction_location_id",
    "auction_location_nickname",
    "auction_location_address",
    "auction_location_city",
    "auction_location_state",
    "auction_location_zip",
    "auction_location_tax_rate",
    "pickup_dates",
    "pickup_dates_list",
    "item_id",
    "lot_code",
    "current_bid",
    "next_bid",
    "msrp",
    "condition",
    "brand",
    "item_title",
    "item_category1",
    "item_category2",
    "quantity",
    "initial_quantity",
    "bid_count",
    "pallet_lot",
    "pallet_auction",
    "image_url",
    "pictures",
    "picture_count",
    "over_time",
    "hours_remaining",
    "ratio_bid_to_msrp",
    "item_url",
]

# Location grouping fix - preserve IN/TN/WV
LOCATION_GROUPS = ["Kentucky", "Ohio", "All Other States"]
