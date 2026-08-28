import csv
import json
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    BASE_URL,
    HEADERS,
    ITEM_SCHEMA,
    MAX_WORKERS,
    POOL_CONNECTIONS,
    POOL_MAXSIZE,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    RETRY_STATUS_FORCELIST,
    RETRY_TOTAL,
)

logger = logging.getLogger(__name__)

thread_local = threading.local()

def get_thread_session() -> requests.Session:
    session = getattr(thread_local, "session", None)
    if session is None:
        session = requests.Session()
        retry = Retry(
            total=RETRY_TOTAL,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=RETRY_STATUS_FORCELIST,
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(
            pool_connections=POOL_CONNECTIONS,
            pool_maxsize=POOL_MAXSIZE,
            max_retries=retry,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(HEADERS)
        thread_local.session = session
    return session

def get_all_locations(session):
    url = f"{BASE_URL}/api/location/getAllLocations"
    response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    try:
        locations = response.json()
    except json.JSONDecodeError as e:
        logger.error("get_all_locations JSON decode failed: %s", e)
        raise
    location_dict = {loc["id"]: loc for loc in locations}
    return location_dict

def get_auctions(session, location_ids, page_id=1):
    location_str = ",".join(str(loc) for loc in location_ids)
    url = f"{BASE_URL}/api/auction/getAuctions?pageId={page_id}&categories=Categories+-+All&pastAuction=false&selectedLocationIds={location_str}"
    response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as e:
        logger.error("get_auctions JSON decode failed page %s: %s", page_id, e)
        raise

def get_items_by_page(session, auction_id, page_id=1):
    url = f"{BASE_URL}/api/item/getItemsByAuctionId/{auction_id}?pageId={page_id}&auctionId={auction_id}"
    response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as e:
        logger.error("get_items_by_page JSON decode failed auction %s page %s: %s", auction_id, page_id, e)
        raise

def get_all_items_for_auction(session, auction_id, max_pages=50):
    """
    Fetch all items for a given auction by iterating page by page until no more items are returned.
    Bounded to avoid infinite loops on API error (duplicate page).
    """
    all_items = []
    page = 1
    seen_first = None
    while page <= max_pages:
        items = get_items_by_page(session, auction_id, page_id=page)
        if not items:
            break
        # guard duplicate page
        if seen_first is None and items:
            seen_first = items[0].get("id")
        elif items and seen_first is not None and len(items) == len(all_items) and items[0].get("id") == seen_first:
            logger.warning("Duplicate page detected for auction %s page %s - breaking", auction_id, page)
            break
        all_items.extend(items)
        page += 1
        # optional early exit if we have itemCount hint would go here, but API does not return it per auction in this path
    if page > max_pages:
        logger.warning("Hit max_pages %s for auction %s with %s items", max_pages, auction_id, len(all_items))
    return all_items

def get_auction_pickup_dates(session, location_id):
    url = f"{BASE_URL}/api/auction/getAuctionPickupDate?categories=Categories%20-%20All&locationIds={location_id}"
    response = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error("get_auction_pickup_dates JSON decode failed loc %s: %s", location_id, e)
        return []
    # normalize to list
    if data is None:
        return []
    if isinstance(data, list):
        return [str(d) for d in data]
    if isinstance(data, str):
        return [data]
    return []

def normalize_pickup(dates):
    if dates is None:
        return []
    if isinstance(dates, list):
        return [str(d) for d in dates]
    if isinstance(dates, str):
        return [dates] if dates else []
    return []

def fetch_auction_data(session, auction, location_data, pickup_dates_cache, pickup_lock):
    """
    Fetch all items for a single auction. Thread-safe pickup cache via double-checked locking.
    Return a list of item rows for that auction per ITEM_SCHEMA-ish.
    """
    try:
        auction_id = auction["id"]
        loc_id = auction.get("locationId")

        # double-checked locking for pickup dates
        pickup_dates = pickup_dates_cache.get(loc_id)
        if pickup_dates is None:
            with pickup_lock:
                pickup_dates = pickup_dates_cache.get(loc_id)
                if pickup_dates is None:
                    try:
                        pickup_dates = get_auction_pickup_dates(session, loc_id)
                    except Exception as e:
                        logger.warning("pickup fetch failed loc %s: %s", loc_id, e)
                        pickup_dates = []
                    pickup_dates_cache[loc_id] = pickup_dates

        pickup_dates = normalize_pickup(pickup_dates)
        pickup_str = "; ".join(pickup_dates)

        # Location details
        loc_info = location_data.get(loc_id, {})
        loc_nickname = loc_info.get("nickName", "")
        loc_address = loc_info.get("address", "")
        loc_city = loc_info.get("city", "")
        loc_state = loc_info.get("state", "")
        loc_zip = loc_info.get("zip", "")
        loc_tax = float(loc_info.get("taxRate", 6.0) or 6.0)

        # Auction fields
        auction_number = auction.get("auctionNumber", "")
        auction_title = auction.get("title", "")
        auction_category = auction.get("category", "")
        auction_start = auction.get("utcStartDateTime", "")
        auction_end = auction.get("utcEndDateTime", "")
        pallet_auction = bool(auction.get("palletAuction", False))

        # Fetch all items for this auction
        all_items = get_all_items_for_auction(session, auction_id)

        rows = []
        for item in all_items:
            current_bid = float(item.get("currentBid", 0.0) or 0.0)
            next_bid = float(item.get("nextBid", current_bid) or current_bid)
            msrp = float(item.get("msrp", 0.0) or 0.0)
            ratio = current_bid / msrp if msrp > 0 else 0
            item_id = item.get("id")
            item_url = f"https://www.bidfta.com/{auction_id}/item-detail/{item_id}"
            pictures = item.get("pictures") or []
            # pictures may be list of urls or dicts
            pic_urls = []
            if isinstance(pictures, list):
                for p in pictures:
                    if isinstance(p, str):
                        pic_urls.append(p)
                    elif isinstance(p, dict) and p.get("picUrl"):
                        pic_urls.append(p["picUrl"])
            # fallback to imageUrl / pictureList
            image_url = item.get("imageUrl", "")
            if not pic_urls and image_url:
                pic_urls = [image_url]
            if not image_url and pic_urls:
                image_url = pic_urls[0]
            pic_list = item.get("pictureList") or []
            if pic_list and not pic_urls:
                for pl in pic_list:
                    if isinstance(pl, dict) and pl.get("picUrl"):
                        pic_urls.append(pl["picUrl"])
                if pic_urls and not image_url:
                    image_url = pic_urls[0]

            row = {
                "auction_id": auction_id,
                "auction_number": auction_number,
                "auction_title": auction_title,
                "auction_category": auction_category,
                "auction_start_datetime": auction_start,
                "auction_end_datetime": auction_end,
                "auction_location_id": loc_id,
                "auction_location_nickname": loc_nickname,
                "auction_location_address": loc_address,
                "auction_location_city": loc_city,
                "auction_location_state": loc_state,
                "auction_location_zip": loc_zip,
                "auction_location_tax_rate": loc_tax,
                "pickup_dates": pickup_str,
                "pickup_dates_list": pickup_dates,
                "item_id": item_id,
                "lot_code": item.get("lotCode", ""),
                "current_bid": current_bid,
                "next_bid": next_bid,
                "msrp": msrp,
                "condition": item.get("condition", ""),
                "brand": item.get("brand", ""),
                "item_title": item.get("title", ""),
                "item_category1": item.get("category1", ""),
                "item_category2": item.get("category2", ""),
                "quantity": float(item.get("quantity", 1) or 1),
                "initial_quantity": float(item.get("initialQuantity", 1) or 1),
                "bid_count": int(item.get("bidsCount", 0) or 0),
                "pallet_lot": bool(item.get("palletLot", False)),
                "pallet_auction": pallet_auction,
                "image_url": image_url,
                "pictures": pic_urls,
                "picture_count": len(pic_urls),
                "over_time": bool(item.get("overTime", False)),
                "hours_remaining": int(item.get("hoursRemaining", 0) or 0),
                "ratio_bid_to_msrp": ratio,
                "item_url": item_url,
            }
            rows.append(row)

        return rows
    except Exception as e:
        logger.exception("fetch_auction_data failed for auction %s: %s", auction.get("id"), e)
        return []

def main():
    location_ids = [637,4,345,515,2,520,24,581,25,21,374]
    now = datetime.now()
    csv_filename = f"./data/auction_data_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    session = get_thread_session()

    logger.info("Fetching all locations information...")
    try:
        location_data = get_all_locations(session)
    except Exception as e:
        logger.error("Failed to fetch locations: %s", e)
        return
    logger.info("Loaded %s locations.", len(location_data))

    fieldnames = [f for f in ITEM_SCHEMA if f != "pickup_dates_list"]

    write_header = False
    try:
        with open(csv_filename, 'r', newline='', encoding='utf-8') as f:
            pass
    except FileNotFoundError:
        write_header = True

    # Collect all auctions first
    logger.info("Collecting all auctions...")
    all_auctions = []
    page = 1
    while True:
        try:
            auctions = get_auctions(session, location_ids, page_id=page)
        except Exception as e:
            logger.error("get_auctions page %s failed: %s", page, e)
            break
        if not auctions:
            break
        all_auctions.extend(auctions)
        page += 1
        if page > 50:
            logger.warning("Hit max auction pages 50")
            break
    logger.info("Found %s total auctions.", len(all_auctions))

    # Process all auctions in parallel with per-thread session
    pickup_dates_cache = {}
    pickup_lock = threading.Lock()
    write_lock = threading.Lock()
    total_items = 0

    logger.info("Fetching items from all auctions in parallel...")
    # need to ensure data dir exists
    import os
    os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        def process_and_write(auction):
            thread_session = get_thread_session()
            rows = fetch_auction_data(thread_session, auction, location_data, pickup_dates_cache, pickup_lock)
            # Write rows to CSV - need to handle pictures list -> json
            with write_lock:
                for r in rows:
                    # flatten pictures to string for CSV
                    r_out = dict(r)
                    if isinstance(r_out.get("pictures"), list):
                        r_out["pictures"] = "|".join(r_out["pictures"])
                    # drop pickup_dates_list
                    r_out.pop("pickup_dates_list", None)
                    writer.writerow({k: r_out.get(k, "") for k in fieldnames})
                return len(rows)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_and_write, a) for a in all_auctions]
            for future in as_completed(futures):
                try:
                    item_count = future.result()
                except Exception as e:
                    logger.error("process_and_write failed: %s", e)
                    item_count = 0
                total_items += item_count

    logger.info("Data collection completed. %s items processed total.", total_items)
    logger.info("Run 'python bidfta_analyze.py' to perform analytics on the collected data.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
