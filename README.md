# Real-Time Crypto & Stock Price API (Lite)

Fast, cached price lookups for **Bitcoin (BTC)**, **Ethereum (ETH)**, and **popular stocks/ETFs**.
Powered by **CoinGecko (crypto)** and **Yahoo Finance unofficial API (stocks)**.

- **Tech**: Python • FastAPI • httpx • in‑memory cache
- **Refresh**: every {REFRESH_INTERVAL_SEC} seconds (defaults to 30)
- **Logging**: Logtail (optional) via env var (no hardcoding)
- **Deploy**: Perfect for Render.com
- **Monetization**: Toggle header **API key** check for higher rate limits ($10–$20/mo idea)

## Quick Start (Local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export REFRESH_INTERVAL_SEC=30
# Optional: require API key
# export REQUIRE_API_KEY=true
# export API_KEY=your-secret-key
# Optional: Logtail
# export LOGTAIL_SOURCE_TOKEN=src-xxxxxxxxxxxxxxxx
uvicorn app:app --reload
```

Visit: `http://127.0.0.1:8000/docs`

## Endpoints

- `GET /` – service info
- `GET /healthz` – health check
- `GET /v1/prices` – **both** crypto + stocks (allowed set)
- `GET /v1/crypto` – BTC, ETH via CoinGecko
- `GET /v1/stocks?symbols=AAPL,MSFT` – Yahoo Finance quote API (unofficial)

### Response Example

```json
{
  "updated_at": 1725798400.123,
  "crypto": {"BTC": 63210.12, "ETH": 2588.34},
  "stocks": {"AAPL": 221.91, "MSFT": 419.31, "SPY": 554.02}
}
```

## Configuration (Env Vars)

| Variable | Default | Description |
|---|---|---|
| `REFRESH_INTERVAL_SEC` | `30` | Background refresh cadence |
| `ALLOWED_STOCKS` | preset list | Comma list of symbols users can query |
| `CORS_ORIGINS` | `*` | Comma list |
| `REQUIRE_API_KEY` | `false` | If `true`, all endpoints require `X-API-Key` |
| `API_KEY` | _none_ | The expected key when `REQUIRE_API_KEY=true` |
| `LOGTAIL_SOURCE_TOKEN` | _none_ | If set, logs stream to Logtail |

> **Note**: No logging keys are hardcoded. Render will inject env vars – set them in **Dashboard → Service → Environment**.

## Deploy to Render

1. Push this repo to GitHub.
2. Create a **Render → Web Service** → **Build Command**: `pip install -r requirements.txt`  
   **Start Command**: _leave blank_ (Render uses `Procfile`).
3. Set env vars (recommended):
   - `REFRESH_INTERVAL_SEC=30`
   - `REQUIRE_API_KEY=true`
   - `API_KEY=<your-paid-plan-key>`
   - `LOGTAIL_SOURCE_TOKEN=<your-logtail-source-token>`
4. Click **Deploy**.

## Monetization Hint

- Free tier: no API key, lower rate limits from your reverse proxy (e.g., Nginx or Cloudflare).  
- Paid tier: turn on `REQUIRE_API_KEY=true`, issue customer keys, and allow higher request rates.

## Notes

- Crypto: [CoinGecko Simple Price API](https://www.coingecko.com/en/api/documentation).  
- Stocks/ETFs: Yahoo Finance **unofficial** quote endpoint (`/v7/finance/quote`). Terms may change; use responsibly.
- This is a *lite* service—no historical bars or OHLCV. Extend as needed.
