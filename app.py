from fastapi import FastAPI, HTTPException, Query, Request, Response, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, asyncio, logging, time
import httpx
from typing import Dict, List, Optional

# ----------------------------- Config -----------------------------
DEFAULT_STOCKS = [
    "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","AMD",
    "NFLX","SPY","QQQ","VTI","IWM","DIA","BRK-B","JPM","KO","XOM","UNH","AVGO"
]

REFRESH_INTERVAL_SEC = int(os.getenv("REFRESH_INTERVAL_SEC", "30"))
ALLOWED_STOCKS_ENV = os.getenv("ALLOWED_STOCKS", "")
ALLOWED_STOCKS = [s.strip().upper() for s in ALLOWED_STOCKS_ENV.split(",") if s.strip()] or DEFAULT_STOCKS

API_KEY_REQUIRED = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
SERVER_API_KEY = os.getenv("API_KEY")  # optional, your monetization gate
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

# ----------------------------- Logging -----------------------------
logger = logging.getLogger("realtime-price-api")
logger.setLevel(logging.INFO)

# Attach Logtail if configured
LOGTAIL_SOURCE_TOKEN = os.getenv("LOGTAIL_SOURCE_TOKEN")
if LOGTAIL_SOURCE_TOKEN:
    try:
        from logtail import LogtailHandler
        handler = LogtailHandler(source_token=LOGTAIL_SOURCE_TOKEN)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info("Logtail handler attached")
    except Exception as e:
        logger.error(f"Failed to attach Logtail handler: {e}")

# Also log to stdout
_stream = logging.StreamHandler()
_stream.setLevel(logging.INFO)
logger.addHandler(_stream)

# ----------------------------- App -----------------------------
app = FastAPI(title="Real-Time Crypto & Stock Price API (Lite)",
              version="1.0.0",
              description="Fast, cached price lookups for BTC, ETH, and popular stocks/ETFs (Yahoo Finance + CoinGecko).")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Models -----------------------------
class PricesResponse(BaseModel):
    updated_at: float
    crypto: Dict[str, float]
    stocks: Dict[str, float]

# ----------------------------- Cache -----------------------------
cache = {
    "crypto": {"data": {}, "updated_at": 0.0},
    "stocks": {"data": {}, "updated_at": 0.0},
}
cache_lock = asyncio.Lock()

# ----------------------------- Helpers -----------------------------
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
COINGECKO_SIMPLE_URL = "https://api.coingecko.com/api/v3/simple/price"

async def fetch_yahoo_quotes(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    # Deduplicate and chunk to avoid overly long URLs
    uniq = list(dict.fromkeys([s.upper() for s in symbols]))
    results: Dict[str, float] = {}

    async with httpx.AsyncClient(timeout=10) as client:
        # chunk into ~50 symbols per request
        for i in range(0, len(uniq), 50):
            chunk = uniq[i:i+50]
            params = {"symbols": ",".join(chunk)}
            r = await client.get(YAHOO_QUOTE_URL, params=params, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            data = r.json()
            for item in data.get("quoteResponse", {}).get("result", []):
                sym = item.get("symbol")
                price = item.get("regularMarketPrice")
                if sym and isinstance(price, (int, float)):
                    results[sym.upper()] = float(price)
    return results

async def fetch_coingecko_prices() -> Dict[str, float]:
    # Fetch BTC, ETH in USD
    ids = "bitcoin,ethereum"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(COINGECKO_SIMPLE_URL, params={"ids": ids, "vs_currencies": "usd"})
        r.raise_for_status()
        data = r.json()
        out = {}
        if "bitcoin" in data and "usd" in data["bitcoin"]:
            out["BTC"] = float(data["bitcoin"]["usd"])
        if "ethereum" in data and "usd" in data["ethereum"]:
            out["ETH"] = float(data["ethereum"]["usd"])
        return out

async def refresh_all(symbols: List[str] = None):
    symbols = symbols or ALLOWED_STOCKS
    try:
        crypto_prices, stock_prices = await asyncio.gather(
            fetch_coingecko_prices(),
            fetch_yahoo_quotes(symbols)
        )
        async with cache_lock:
            cache["crypto"]["data"] = crypto_prices
            cache["crypto"]["updated_at"] = time.time()
            cache["stocks"]["data"] = stock_prices
            cache["stocks"]["updated_at"] = time.time()
        logger.info({"event":"refresh_success","crypto_count":len(crypto_prices),"stock_count":len(stock_prices)})
    except Exception as e:
        logger.error({"event":"refresh_error","error":str(e)})

# Background refresh loop
_bg_task: Optional[asyncio.Task] = None
async def _refresher():
    await asyncio.sleep(1)
    while True:
        await refresh_all()
        await asyncio.sleep(REFRESH_INTERVAL_SEC)

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY_REQUIRED:
        if not SERVER_API_KEY or not x_api_key or x_api_key != SERVER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

@app.on_event("startup")
async def on_startup():
    global _bg_task
    _bg_task = asyncio.create_task(_refresher())
    logger.info({"event":"startup","refresh_interval_sec":REFRESH_INTERVAL_SEC,"allowed_stocks":ALLOWED_STOCKS})

@app.on_event("shutdown")
async def on_shutdown():
    global _bg_task
    if _bg_task:
        _bg_task.cancel()
        try:
            await _bg_task
        except asyncio.CancelledError:
            pass
    logger.info({"event":"shutdown"})

# ----------------------------- Routes -----------------------------
@app.get("/healthz")
async def health():
    return {"ok": True, "time": time.time()}

@app.get("/v1/prices", response_model=PricesResponse)
async def get_all_prices(_=Depends(require_api_key)):
    async with cache_lock:
        return {
            "updated_at": min(cache["crypto"]["updated_at"], cache["stocks"]["updated_at"]) or time.time(),
            "crypto": cache["crypto"]["data"],
            "stocks": {k: v for k, v in cache["stocks"]["data"].items() if k in ALLOWED_STOCKS},
        }

@app.get("/v1/crypto")
async def get_crypto(_=Depends(require_api_key)):
    async with cache_lock:
        return {"updated_at": cache["crypto"]["updated_at"], "crypto": cache["crypto"]["data"]}

@app.get("/v1/stocks")
async def get_stocks(symbols: Optional[str] = Query(None, description="Comma-separated e.g. AAPL,MSFT"), _=Depends(require_api_key)):
    async with cache_lock:
        data = cache["stocks"]["data"].copy()
    if symbols:
        req = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        # filter to allowed set
        req = [s for s in req if s in ALLOWED_STOCKS]
        data = {k: v for k, v in data.items() if k in req}
    else:
        # default to allowed list
        data = {k: v for k, v in data.items() if k in ALLOWED_STOCKS}
    return {"updated_at": cache["stocks"]["updated_at"], "stocks": data}

# Root
@app.get("/")
async def root():
    return {
        "name": "Real-Time Crypto & Stock Price API (Lite)",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/v1/prices", "/v1/crypto", "/v1/stocks?symbols=AAPL,MSFT"],
        "allowed_stocks": ALLOWED_STOCKS,
        "crypto": ["BTC","ETH"]
    }
