"""
台股量化操盤決策系統 v2（單檔版）
執行：pip install streamlit yfinance pandas numpy plotly requests lxml html5lib beautifulsoup4 tzdata
      streamlit run tw_quant_app.py
      （背景推播模式）python tw_quant_app.py --daemon
"""
from __future__ import annotations

import datetime as dt
import html
import io
import json
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import quote
import warnings
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ============================================================================
# A. 資料來源層
# ============================================================================
TZ = ZoneInfo("Asia/Taipei")
UA = {"User-Agent": "Mozilla/5.0 (compatible; tw-quant/1.0)"}
try:
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
except Exception:
    pass


def now_tw() -> dt.datetime:
    return dt.datetime.now(TZ)


def _get(url: str, params: dict | None = None, timeout: int = 25) -> requests.Response:
    last = None
    for verify in (True, False):   # 部分政府網站憑證鏈在新版 Python 會驗證失敗
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout, verify=verify)
            r.raise_for_status()
            return r
        except requests.exceptions.SSLError as e:
            last = e
            continue
        except Exception as e:
            last = e
            break
    raise last  # type: ignore[misc]


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan


# ----------------------------------------------------------------------------
def load_universe():
    rows, msgs = [], []
    for mode, sfx in ((2, "TW"), (4, "TWO")):
        try:
            r = _get("https://isin.twse.com.tw/isin/C_public.jsp", params={"strMode": mode})
            tbl = pd.read_html(io.StringIO(r.content.decode("big5", errors="ignore")))[0]
            if not any("有價證券" in str(c) for c in tbl.columns):
                tbl.columns = tbl.iloc[0]
                tbl = tbl.iloc[1:]
            cols = [str(c) for c in tbl.columns]
            tbl.columns = cols
            c0 = next((c for c in cols if "有價證券代號" in c), cols[0])
            c_ind = next((c for c in cols if "產業別" in c), None)
            c_cfi = next((c for c in cols if "CFI" in c.upper()), None)
            for _, row in tbl.iterrows():
                cell = str(row[c0])
                if "\u3000" not in cell:
                    continue
                code, name = cell.split("\u3000", 1)
                code = code.strip()
                if not re.fullmatch(r"\d{4}", code):
                    continue
                if c_cfi:
                    cfi = str(row[c_cfi])
                    if cfi not in ("nan", "") and not cfi.startswith("ES"):
                        continue
                rows.append(dict(code=code, name=name.strip(), market=sfx,
                                 industry=str(row[c_ind]) if c_ind else ""))
        except Exception as e:
            msgs.append(f"股票清單（{'上市' if mode == 2 else '上櫃'}）抓取失敗：{e}")
    df = pd.DataFrame(rows).drop_duplicates("code") if rows else pd.DataFrame(columns=["code", "name", "market", "industry"])
    if not df.empty:
        msgs.append(f"股票清單：上市 {int((df.market == 'TW').sum())} 檔、上櫃 {int((df.market == 'TWO').sum())} 檔")
    return df, msgs


def load_daily_turnover():
    out, msgs = {}, []
    try:
        js = _get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL").json()
        for r in js:
            out[str(r.get("Code", "")).strip()] = _num(r.get("TradeValue"))
        msgs.append(f"上市當日成交值：{len(js)} 檔")
    except Exception as e:
        msgs.append(f"上市當日成交值抓取失敗：{e}")
    try:
        js = _get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes").json()
        n = 0
        for r in js:
            ck = next((k for k in r if "Code" in k), None)
            ak = next((k for k in r if "Amount" in k), None)
            if ck and ak:
                out[str(r[ck]).strip()] = _num(r[ak])
                n += 1
        msgs.append(f"上櫃當日成交值：{n} 檔")
    except Exception as e:
        msgs.append(f"上櫃當日成交值抓取失敗：{e}")
    return pd.Series(out, dtype=float).dropna(), msgs


def load_institutional(n_days: int = 5):
    """上市三大法人近 n 個交易日買賣超股數（加總）。上櫃暫無來源，UI 會顯示『無資料』。"""
    msgs, frames, used = [], [], []
    day = now_tw().date()
    attempts = 0
    while len(used) < n_days and attempts < 16:
        attempts += 1
        if day.weekday() < 5:
            try:
                js = _get("https://www.twse.com.tw/rwd/zh/fund/T86",
                          params={"date": day.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"}).json()
                if js.get("stat") == "OK" and js.get("data"):
                    f = js["fields"]
                    i_code = next(i for i, x in enumerate(f) if "證券代號" in x)
                    i_for = next(i for i, x in enumerate(f) if x.startswith("外陸資買賣超") or ("外資" in x and "買賣超" in x))
                    i_trust = next(i for i, x in enumerate(f) if x.startswith("投信買賣超"))
                    i_tot = next(i for i, x in enumerate(f) if x.startswith("三大法人買賣超"))
                    df = pd.DataFrame([[r[i_code].strip(), _num(r[i_for]), _num(r[i_trust]), _num(r[i_tot])]
                                       for r in js["data"]], columns=["code", "foreign", "trust", "total"])
                    frames.append(df)
                    used.append(day)
                time.sleep(0.5)
            except Exception as e:
                msgs.append(f"T86 {day} 失敗：{e}")
        day -= dt.timedelta(days=1)
    if not frames:
        msgs.append("三大法人資料抓取失敗，籌碼欄位將顯示『無資料』。")
        return pd.DataFrame(columns=["foreign", "trust", "total"]), [], msgs
    out = pd.concat(frames).groupby("code")[["foreign", "trust", "total"]].sum()
    msgs.append(f"三大法人（上市）：{len(used)} 個交易日 {min(used)}~{max(used)}")
    return out, used, msgs


def load_revenue_yoy():
    out, msgs = {}, []
    urls = ["https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
            "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"]
    for u in urls:
        try:
            js = _get(u).json()
            n = 0
            for r in js:
                ck = next((k for k in r if "公司代號" in k or k.lower().endswith("code")), None)
                yk = next((k for k in r if "去年同月增減" in k), None)
                if ck and yk:
                    out[str(r[ck]).strip()] = _num(r[yk])
                    n += 1
            msgs.append(f"月營收 YoY：{n} 檔（{'上市' if 'twse' in u else '上櫃'}）")
        except Exception as e:
            msgs.append(f"月營收抓取失敗（{'上市' if 'twse' in u else '上櫃'}）：{e}")
    return pd.Series(out, dtype=float), msgs


def load_flags():
    punish, notice, msgs = set(), set(), []
    for name, url, bucket in (("處置股", "https://openapi.twse.com.tw/v1/announcement/punish", punish),
                              ("注意股", "https://openapi.twse.com.tw/v1/announcement/notice", notice)):
        try:
            for r in _get(url).json():
                ck = next((k for k in r if k.lower().endswith("code") or "代號" in k), None)
                if ck:
                    bucket.add(str(r[ck]).strip())
            msgs.append(f"{name}（上市）：{len(bucket)} 檔")
        except Exception as e:
            msgs.append(f"{name}抓取失敗：{e}")
    return punish, notice, msgs


# ----------------------------------------------------------------------------
def download_prices(tickers: list[str], period: str = "4y", chunk: int = 100, min_len: int = 120, keep_rows: int | None = None) -> dict:
    import yfinance as yf

    out = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, period=period, group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = set(raw.columns.get_level_values(0))
            if "Close" in lvl0 and len(part) == 1:          # 單檔時層級可能反過來
                frames = {part[0]: raw.copy()}
                frames[part[0]].columns = frames[part[0]].columns.get_level_values(0)
            else:
                frames = {t: raw[t] for t in part if t in lvl0}
        else:
            frames = {part[0]: raw}
        for t, df in frames.items():
            try:
                df = df.dropna(how="all")
                df = df[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]]
                if df.shape[1] < 5 or len(df) < min_len:
                    continue
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df = df[~df.index.duplicated(keep="last")].sort_index().astype("float32")
                out[t] = df.tail(keep_rows) if keep_rows else df
            except Exception:
                continue
    return out


def links(code: str, market: str = "TW") -> dict:
    tv = "TPEX" if market == "TWO" else "TWSE"
    return {
        "Yahoo 股市": f"https://tw.stock.yahoo.com/quote/{code}",
        "Goodinfo": f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={code}",
        "玩股網": f"https://www.wantgoo.com/stock/{code}",
        "TradingView": f"https://tw.tradingview.com/chart/?symbol={tv}%3A{code}",
        "公開資訊觀測站": f"https://mops.twse.com.tw/mops/web/t05st01?co_id={code}",
    }


STATE_DIR = Path(os.environ.get("STATE_DIR", "."))   # 排程模式把狀態檔放在 state 分支的資料夾


def rss_mb() -> float:
    """目前程序記憶體（MB），只讀 /proc，不需額外套件。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def trim_memory():
    """強制回收並把空閒記憶體還給作業系統（降低長時間執行的碎片化）。"""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 機密設定：環境變數 > Streamlit secrets
# ----------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as _st
        v = _st.secrets.get(name, None)
        return str(v) if v else default
    except Exception:
        return default


# ----------------------------------------------------------------------------
# 名稱/市場別：多來源合併，避免只顯示代碼或名稱錯誤（公司簡稱優先）
# ----------------------------------------------------------------------------
def load_names():
    names, markets, msgs = {}, {}, []
    src = [("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "TW", ("公司簡稱",)),
           ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "TWO", ("公司簡稱", "CompanyAbbreviation")),
           ("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", "TW", ("Name",)),
           ("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes", "TWO", ("CompanyName",))]
    for url, mk, keys in src:
        try:
            js, n = _get(url).json(), 0
            for r in js:
                ck = next((k for k in r if "公司代號" in k or k.lower().endswith("code")), None)
                nk = next((k for k in keys if k in r), None) or next((k for k in r if "簡稱" in k), None)
                if ck and nk:
                    code, nm = str(r[ck]).strip(), str(r[nk]).strip()
                    if re.fullmatch(r"\d{4}", code) and nm:
                        names.setdefault(code, nm)          # 先到先贏 = 優先序
                        markets.setdefault(code, mk)
                        n += 1
            msgs.append(f"名稱來源 {url.split('/')[-1]}：{n} 檔")
        except Exception as e:
            msgs.append(f"名稱來源 {url.split('/')[-1]} 失敗：{e}")
    return names, markets, msgs


# ----------------------------------------------------------------------------
# 價格增量合併 / 盤中未收盤 K 線處理
# ----------------------------------------------------------------------------
def merge_recent(hist: dict, recent: dict) -> dict:
    out = dict(hist)
    for t, r in recent.items():
        if t in out:
            m = pd.concat([out[t], r])
            out[t] = m[~m.index.duplicated(keep="last")].sort_index()
        else:
            out[t] = r
    return out


def is_partial(df: pd.DataFrame, now: dt.datetime | None = None) -> bool:
    now = now or now_tw()
    return bool(len(df)) and df.index[-1].date() == now.date() and now.time() < dt.time(14, 0)


def closed_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:-1] if is_partial(df) else df


# ----------------------------------------------------------------------------
# 推播：Telegram / LINE Messaging API（LINE Notify 已於 2025/3/31 終止服務）
# ----------------------------------------------------------------------------
def send_telegram(token: str, chat_id: str, text: str):
    try:
        for i in range(0, len(text), 3800):
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=20,
                              json={"chat_id": chat_id, "text": text[i:i + 3800], "disable_web_page_preview": True})
            if not r.ok:
                return False, f"Telegram 失敗 {r.status_code}: {r.text[:150]}"
        return True, "Telegram ✔"
    except Exception as e:
        return False, f"Telegram 例外：{e}"


def send_line(token: str, user_id: str, text: str):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        for i in range(0, len(text), 4500):
            msg = [{"type": "text", "text": text[i:i + 4500]}]
            if user_id:
                url, body = "https://api.line.me/v2/bot/message/push", {"to": user_id, "messages": msg}
            else:
                url, body = "https://api.line.me/v2/bot/message/broadcast", {"messages": msg}
            r = requests.post(url, headers=headers, json=body, timeout=20)
            if not r.ok:
                return False, f"LINE 失敗 {r.status_code}: {r.text[:150]}"
        return True, "LINE ✔"
    except Exception as e:
        return False, f"LINE 例外：{e}"


def notify_cfg() -> dict:
    return dict(tg_token=get_secret("TELEGRAM_BOT_TOKEN"), tg_chat=get_secret("TELEGRAM_CHAT_ID"),
                line_token=get_secret("LINE_CHANNEL_TOKEN"), line_user=get_secret("LINE_USER_ID"),
                line_broadcast=get_secret("LINE_BROADCAST").strip().lower() in ("1", "true", "yes", "y"))


PRIVATE_CATS = {"hold"}      # 含成本/損益的持股警示：永遠不廣播給好友


def notify_routed(public_text: str, private_text: str, cfg: dict) -> list:
    """LINE 路由：
    只填 User ID、沒開廣播 → 全部只傳給你；
    開廣播 → 公開訊號廣播給所有好友，持股警示仍只傳給你（需 User ID）；
    沒填 User ID → 公開訊號廣播，持股警示不經 LINE 發送。Telegram 則全部傳給設定的 Chat。"""
    res, both = [], "\n\n".join(t for t in (public_text, private_text) if t)
    if cfg.get("tg_token") and cfg.get("tg_chat") and both:
        res.append(send_telegram(cfg["tg_token"], cfg["tg_chat"], both))
    if cfg.get("line_token"):
        user, bc = cfg.get("line_user", ""), cfg.get("line_broadcast", False)
        if user and not bc:
            if both:
                res.append(send_line(cfg["line_token"], user, both))
        else:
            if public_text:
                res.append(send_line(cfg["line_token"], "", public_text))
            if private_text and user:
                res.append(send_line(cfg["line_token"], user, private_text))
    return res


def notify_all(text: str, cfg: dict) -> list:
    return notify_routed(text, "", cfg)


# ----------------------------------------------------------------------------
# 訊號去重（檔案持久化，跨工作階段/重啟不重複推播）
# ----------------------------------------------------------------------------
ALERT_FILE = STATE_DIR / "alert_state.json"
_ALERT_LOCK = threading.Lock()


def _alert_read() -> dict:
    try:
        return json.loads(ALERT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _alert_write(state: dict):
    try:
        ALERT_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def alert_claim(keys: list) -> list:
    with _ALERT_LOCK:
        state = _alert_read()
        seen, now = state.setdefault("seen", {}), time.time()
        fresh = [k for k in dict.fromkeys(keys) if k not in seen]
        for k in fresh:
            seen[k] = now
        for k in [k for k, v in seen.items() if now - v > 4 * 86400]:
            del seen[k]
        _alert_write(state)
        return fresh


def alert_release(keys: list):
    with _ALERT_LOCK:
        state = _alert_read()
        for k in keys:
            state.get("seen", {}).pop(k, None)
        _alert_write(state)


def alert_state_get(name: str, default=None):
    return _alert_read().get("state", {}).get(name, default)


def alert_state_set(name: str, value):
    with _ALERT_LOCK:
        state = _alert_read()
        state.setdefault("state", {})[name] = value
        _alert_write(state)



# ----------------------------------------------------------------------------
# 主力（外資+投信）逐日買賣超歷史：FinMind 公開 API（上市櫃皆有）。
# 有歷史才能「回測」主力因子。可選 FINMIND_TOKEN 提高呼叫上限；本機 CSV 快取、每日增量更新。
# ----------------------------------------------------------------------------
INST_FILE = STATE_DIR / "inst_history.csv"
_FM_URL = "https://api.finmindtrade.com/api/v4/data"


def load_inst_history(codes, years: int = 4, token: str = "", max_fetch: int = 400):
    msgs = []
    cols = ["date", "code", "foreign", "trust"]
    start = (now_tw().date() - dt.timedelta(days=365 * years + 10)).isoformat()
    try:
        cache = (pd.read_csv(INST_FILE, dtype={"code": str}, parse_dates=["date"])
                 if INST_FILE.exists() else pd.DataFrame(columns=cols))
    except Exception:
        cache = pd.DataFrame(columns=cols)
    last = cache.groupby("code")["date"].max().to_dict() if len(cache) else {}
    marker = STATE_DIR / "inst_refresh.txt"
    today_s = now_tw().strftime("%Y-%m-%d")
    try:
        done_today = marker.exists() and marker.read_text().strip() == today_s
    except Exception:
        done_today = False
    todo = [c for c in codes if not (done_today and c in last)]
    if len(todo) > max_fetch:
        msgs.append(f"股票池大於 {max_fetch} 檔：只更新成交值前 {max_fetch} 檔的主力逐日資料，其餘用量價足跡代理。")
        todo = todo[:max_fetch]
    state = {"limit": False}

    def fetch(code):
        if state["limit"]:
            return code, None
        s0 = (last[code] - pd.Timedelta(days=5)).date().isoformat() if code in last else start
        try:
            params = dict(dataset="TaiwanStockInstitutionalInvestorsBuySell", data_id=code, start_date=s0)
            if token:
                params["token"] = token
            js = requests.get(_FM_URL, params=params, headers=UA, timeout=30).json()
            if js.get("status") != 200:
                if js.get("status") in (402, 429):
                    state["limit"] = True
                return code, None
            df = pd.DataFrame(js.get("data", []))
            if df.empty:
                return code, pd.DataFrame(columns=cols)
            df["net"] = pd.to_numeric(df["buy"], errors="coerce") - pd.to_numeric(df["sell"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            f = df[df["name"].isin(["Foreign_Investor", "Foreign_Dealer_Self"])].groupby("date")["net"].sum()
            t = df[df["name"] == "Investment_Trust"].groupby("date")["net"].sum()
            out = pd.DataFrame({"foreign": f, "trust": t}).reset_index()
            out.insert(1, "code", code)
            return code, out
        except Exception:
            return code, None

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, todo))
    new = [r for _, r in res if r is not None and len(r)]
    ok = sum(1 for _, r in res if r is not None)
    if new:
        cache = pd.concat([cache] + new, ignore_index=True).drop_duplicates(["date", "code"], keep="last")
        try:
            cache.to_csv(INST_FILE, index=False)
        except Exception:
            pass
    if ok and not state["limit"]:
        try:
            marker.write_text(today_s)
        except Exception:
            pass
    if state["limit"]:
        msgs.append("FinMind 已達呼叫上限：請設定環境變數 FINMIND_TOKEN 或稍後再試；先使用已快取的資料。")
    msgs.append(f"主力逐日資料（FinMind 外資/投信）：本次更新 {ok}/{len(todo)} 檔，快取共 {cache['code'].nunique() if len(cache) else 0} 檔")
    out = {}
    if len(cache):
        for code, g in cache[cache["code"].isin(list(codes))].groupby("code"):
            out[code] = g.set_index("date")[["foreign", "trust"]].sort_index()
    return out, msgs


# ----------------------------------------------------------------------------
# 新聞：Google News RSS（免金鑰），以關鍵字做簡單多空計分。僅作『避雷與輔助』，無法回測。
# ----------------------------------------------------------------------------
NEWS_POS = ["創高", "創新高", "大漲", "漲停", "營收創", "獲利成長", "轉盈", "調升", "上修", "看好", "利多", "大單",
            "擴產", "接單", "訂單", "需求強", "旺季", "法說會報喜", "目標價上調", "買超", "策略合作", "認證"]
NEWS_NEG = ["下修", "調降", "虧損", "衰退", "裁員", "處置", "跌停", "重挫", "大跌", "利空", "遭調查", "下調", "砍單",
            "示警", "出貨", "賣超", "停工", "違約", "掏空", "爆雷", "財測不如", "減產", "遭罰", "訴訟"]


def load_news(query: str, n: int = 8) -> list:
    url = ("https://news.google.com/rss/search?q=" + quote(query + " when:7d")
           + "&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    root = ET.fromstring(_get(url, timeout=15).content)
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate")).astimezone(TZ)
        except Exception:
            pub = None
        items.append(dict(title=title, link=(it.findtext("link") or "").strip(), time=pub,
                          source=(it.findtext("source") or "").strip()))
        if len(items) >= n:
            break
    return items


def news_counts(items: list):
    pos = sum(any(k in i["title"] for k in NEWS_POS) for i in items)
    neg = sum(any(k in i["title"] for k in NEWS_NEG) for i in items)
    return pos, neg



# ----------------------------------------------------------------------------
# 官方收盤價（交叉驗證 yfinance）、盤中成交量外推、資料品質檢查
# ----------------------------------------------------------------------------
def load_official_close():
    out, msgs = {}, []
    try:
        for r in _get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL").json():
            out[str(r.get("Code", "")).strip()] = _num(r.get("ClosingPrice"))
    except Exception as e:
        msgs.append(f"官方收盤價（上市）抓取失敗：{e}")
    try:
        for r in _get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes").json():
            ck = next((k for k in r if "Code" in k), None)
            vk = "Close" if "Close" in r else next((k for k in r if "Close" in k), None)
            if ck and vk:
                out[str(r[ck]).strip()] = _num(r[vk])
    except Exception as e:
        msgs.append(f"官方收盤價（上櫃）抓取失敗：{e}")
    return pd.Series(out, dtype=float).dropna(), msgs


DATA_DELAY_MIN = 20     # yfinance 台股報價約延遲 15~20 分鐘
_VOL_CURVE = [(540, 0.0), (570, 0.22), (600, 0.34), (630, 0.44), (660, 0.54), (690, 0.63),
              (720, 0.72), (750, 0.82), (780, 0.92), (810, 1.0)]     # 台股日內累積成交量比例（經驗曲線）


def session_fraction(now: dt.datetime | None = None) -> float:
    n = (now or now_tw()) - dt.timedelta(minutes=DATA_DELAY_MIN)
    m = n.hour * 60 + n.minute
    xs, ys = zip(*_VOL_CURVE)
    return float(np.clip(np.interp(m, xs, ys), 0.12, 1.0))


def project_partial_volume(df: pd.DataFrame) -> pd.DataFrame:
    """盤中最後一根 K 棒的成交量只是『到目前為止』；依日內曲線外推成全日量，量能因子才不會被低估。"""
    if not is_partial(df):
        return df
    df = df.copy()
    c = df.columns.get_loc("Volume")
    df.iloc[-1, c] = df.iloc[-1, c] / session_fraction()
    return df


def validate_prices(prices: dict, official, bench_last):
    """剔除：資料過舊、成交量過薄、異常跳價(疑似未調整的分割/壞資料)、與官方收盤價不符。"""
    keep, rep = {}, dict(ok=0, stale=0, thin=0, jump=0, mismatch=0, skipped=False)
    mism = []
    for t, df in prices.items():
        code = t.split(".")[0]
        if t.startswith("^") or code.startswith("00"):
            keep[t] = df
            continue
        if df.index[-1] < bench_last - pd.Timedelta(days=5):
            rep["stale"] += 1
            continue
        if (df["Volume"].tail(60) == 0).mean() > 0.2:
            rep["thin"] += 1
            continue
        if df["Close"].pct_change().abs().tail(250).max() > 0.25:
            rep["jump"] += 1
            continue
        keep[t] = df
        if official is not None and code in official.index and pd.notna(official[code]) and official[code] > 0:
            recent = df["Close"].tail(3).to_numpy(dtype=float)
            if not np.any(np.abs(recent / float(official[code]) - 1) < 0.015):
                mism.append(t)
    checked = max(len(keep), 1)
    if mism and len(mism) / checked > 0.25:      # 大量不符＝兩邊日期不同步，不是壞資料
        rep["skipped"] = True
    else:
        for t in mism:
            keep.pop(t, None)
        rep["mismatch"] = len(mism)
    rep["ok"] = sum(1 for t in keep if not t.startswith("^") and not t.startswith("00"))
    return keep, rep


# ============================================================================
# B. 量化引擎
# ============================================================================
TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# 台股升降單位
# ----------------------------------------------------------------------------
def tick_size(p: float) -> float:
    if p < 10:
        return 0.01
    if p < 50:
        return 0.05
    if p < 100:
        return 0.1
    if p < 500:
        return 0.5
    if p < 1000:
        return 1.0
    return 5.0


def tick_round(p: float, mode: str = "nearest") -> float:
    t = tick_size(p)
    n = p / t
    if mode == "down":
        n = np.floor(n + 1e-9)
    elif mode == "up":
        n = np.ceil(n - 1e-9)
    else:
        n = np.round(n)
    return round(float(n * t), 2)


# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Costs:
    fee_rate: float = 0.001425   # 公定手續費
    fee_disc: float = 0.6        # 券商折扣（0.6 = 6 折）
    tax: float = 0.003           # 證交稅（賣出）
    slip: float = 0.001          # 單邊滑價

    @property
    def fee(self) -> float:
        return self.fee_rate * self.fee_disc

    @property
    def round_trip(self) -> float:
        return 2 * self.fee + self.tax + 2 * self.slip


@dataclass(frozen=True)
class Filters:
    min_price: float = 35.0
    min_turnover: float = 1.5e8   # 5 日均成交值 (TWD)
    min_amp: float = 8.0          # 20 日振幅 %
    bias5_max: float = 3.5
    bias20_max: float = 10.0


HEATS = {"嚴": (3.5, 10.0), "中": (6.0, 15.0), "寬": (9.0, 25.0)}   # (5MA乖離上限, 20MA乖離上限)：越寬越能買到強勢領漲股
OBJ_MODES = {"excess": "超額報酬（對大盤，含回撤懲罰）", "sharpe": "Sharpe（風險調整後）",
             "winrate": "勝率優先（須贏過大盤且期望值>0）", "tstat": "每筆期望值 t 值"}


@dataclass(frozen=True)
class Params:
    preset: str = "均衡"
    min_score: float = 65.0
    stop_atr: float = 3.0
    tp_r: float = 0.0          # 0 = 不設固定停利，用移動停損讓獲利奔跑
    hold: int = 30
    trail_atr: float = 4.5
    heat: str = "中"
    active: float = 1.0        # 主動選股部位占資金比例；其餘閒置資金持有大盤(0050)。0 = 純大盤擇時
    bear_cash: bool = False    # True=大盤空頭時全部空手；False=空頭仍持有大盤(不輸大盤，但回撤較大)

    def label(self) -> str:
        if self.active <= 0:
            return f"純大盤擇時｜空頭{'空手' if self.bear_cash else '續抱'}"
        tp = f"停利{self.tp_r}R" if self.tp_r > 0 else "移動停利"
        return (f"{self.preset}｜門檻{self.min_score:.0f}｜乖離{self.heat}｜停損{self.stop_atr}ATR｜{tp}｜追蹤{self.trail_atr}ATR｜"
                f"持股≤{self.hold}日｜主動{self.active:.0%}｜空頭{'空手' if self.bear_cash else '續抱大盤'}")


# 八類特徵權重預設：由 walk-forward 在『樣本外』挑選，不再拍腦袋固定 75/20/5
PRESETS = {
    "均衡":       {"trend": .14, "compress": .08, "mom": .10, "rs": .20, "main": .20, "setup": .08, "nh": .10, "sector": .10},
    "主力籌碼型": {"trend": .10, "compress": .05, "mom": .08, "rs": .15, "main": .35, "setup": .07, "nh": .08, "sector": .12},
    "強勢領漲型": {"trend": .10, "compress": .05, "mom": .10, "rs": .30, "main": .10, "setup": .05, "nh": .20, "sector": .10},
}
DEFAULT_GRID = dict(
    preset=list(PRESETS),
    min_score=[60, 70],
    heat=["中", "寬"],
    stop_atr=[2.5, 3.5],
    tp_r=[0.0],
    trail_atr=[3.0, 4.5],
    hold=[20, 40],
    active=[0.5, 1.0],
    bear_cash=[False, True],
)


# ----------------------------------------------------------------------------
# 指標（單一標的）
# ----------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d["Volume"] = d["Volume"].fillna(0.0)
    o, c, h, l, v = d["Open"], d["Close"], d["High"], d["Low"], d["Volume"]

    for n in (5, 10, 20, 60):
        d[f"MA{n}"] = c.rolling(n).mean()
    d["MA20_slope5"] = d["MA20"] / d["MA20"].shift(5) - 1
    d["EMA10"] = c.ewm(span=10, adjust=False).mean()
    d["Vol_MA5"] = v.rolling(5).mean()
    d["Vol_MA20"] = v.rolling(20).mean()
    d["Turnover_MA5"] = (c * v).rolling(5).mean()
    d["Amp20"] = (h.rolling(20).max() / l.rolling(20).min() - 1) * 100
    for n in (5, 10, 20, 60):
        d[f"Bias{n}"] = (c / d[f"MA{n}"] - 1) * 100

    std20 = c.rolling(20).std()
    d["BB_Width"] = 4 * std20 / d["MA20"]
    d["BB_Pct"] = d["BB_Width"].rolling(60).rank(pct=True)  # 帶寬在近60日的分位，越低越收斂

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()  # Wilder

    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["RSI"] = 100 - 100 / (1 + ag / (al + 1e-12))

    rng9 = h.rolling(9).max() - l.rolling(9).min()
    rsv = ((c - l.rolling(9).min()) / (rng9 + 1e-9) * 100).fillna(50)
    d["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["D"] = d["K"].ewm(alpha=1 / 3, adjust=False).mean()

    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACD_Hist"] = dif - dif.ewm(span=9, adjust=False).mean()

    # ---- 大戶/主力足跡（量價代理，可回測；真實法人/大戶資料只用於即時加減分）----
    clv = ((c - l) - (h - c)) / (h - l + 1e-9)
    d["CLV5"] = (clv * v).rolling(5).sum() / (v.rolling(5).sum() + 1e-9)
    d["CMF20"] = (clv * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)      # Chaikin 資金流
    big = v > 1.5 * d["Vol_MA20"].shift(1)
    acc = (big & (c > o) & (clv > 0.4)).astype(float)                            # 大量收高 = 吸籌日
    dist = (big & (c < o) & (clv < -0.4)).astype(float)                          # 大量收低 = 出貨日
    d["AD_net20"] = acc.rolling(20).sum() - dist.rolling(20).sum()
    upv = v.where(c > pc, 0.0).rolling(20).sum()
    dnv = v.where(c < pc, 0.0).rolling(20).sum()
    d["UDVR"] = upv / (dnv + 1.0)                                                # 上漲量/下跌量

    tp = (h + l + c) / 3
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    d["MFI"] = 100 - 100 / (1 + pos / (neg + 1e-9))

    d["Close_loc"] = (c - l) / (h - l + 1e-9)
    d["Range5"] = (h.rolling(5).max() - l.rolling(5).min()) / c * 100
    d["Low5"] = l.rolling(5).min()
    d["High20_prev"] = h.rolling(20).max().shift(1)
    d["pct"] = c.pct_change() * 100
    d["Ret20"] = c / c.shift(20) - 1
    d["Ret60"] = c / c.shift(60) - 1
    d["Ret120"] = c / c.shift(120) - 1
    d["Hi250"] = c / h.rolling(250, min_periods=120).max()          # 距52週高點
    for k in ("MF10", "TR10", "FR10"):                                # 主力(外資+投信)淨買超/成交量，需另外 attach_inst
        d[k] = np.nan
    return d


def attach_inst(d: pd.DataFrame, inst) -> pd.DataFrame:
    """inst：index=日期、欄位 foreign / trust（淨買賣超股數）。算 10 日淨買超占成交量比。"""
    if inst is None or len(inst) == 0:
        return d
    x = inst.reindex(d.index)
    have = (x["foreign"].notna() | x["trust"].notna()).astype(float).rolling(10, min_periods=1).max() > 0
    f, t = x["foreign"].fillna(0.0), x["trust"].fillna(0.0)
    v10 = d["Volume"].rolling(10).sum() + 1.0
    d["FR10"] = (f.rolling(10).sum() / v10).where(have)
    d["TR10"] = (t.rolling(10).sum() / v10).where(have)
    d["MF10"] = ((f + t).rolling(10).sum() / v10).where(have)
    return d


PANEL_FIELDS = [
    "Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "MA60", "MA20_slope5", "EMA10",
    "Vol_MA5", "Vol_MA20", "Turnover_MA5", "Amp20", "Bias5", "Bias10", "Bias20", "BB_Pct", "ATR", "RSI",
    "K", "D", "MACD_Hist", "MFI", "CLV5", "CMF20", "AD_net20", "UDVR", "Close_loc", "Range5", "Low5",
    "High20_prev", "pct", "Ret20", "Ret60", "Ret120", "Hi250", "MF10", "TR10", "FR10",
]



def build_panel_stream(dfs: dict, inst: dict, idx) -> dict:
    """逐檔計算指標、直接寫進 float32 陣列，不保留每檔的完整指標表（大幅降低尖峰記憶體）。"""
    codes = list(dfs)
    T, N = len(idx), len(codes)
    arrs = {f: np.full((T, N), np.nan, dtype=np.float32) for f in PANEL_FIELDS}
    for j, code in enumerate(codes):
        d = attach_inst(add_indicators(dfs[code]), inst.get(code)).reindex(idx)
        for f in PANEL_FIELDS:
            arrs[f][:, j] = d[f].to_numpy(dtype=np.float32)
    return {f: pd.DataFrame(a, index=idx, columns=codes) for f, a in arrs.items()}


def build_panel(stock_dfs: dict) -> dict:
    """code -> indicator df  ==>  field -> DataFrame(date x code)"""
    P = {}
    for f in PANEL_FIELDS:
        P[f] = pd.concat({code: df[f] for code, df in stock_dfs.items()}, axis=1).sort_index()
    return P


# ----------------------------------------------------------------------------
# 特徵 / 分數 / 過濾
# ----------------------------------------------------------------------------
def sector_strength(mb: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """族群強弱：各產業『成員動能平均』的橫斷面百分位，回填給每檔成員（≥3 檔的族群才計算）。"""
    T, N = mb.shape
    out = np.full((T, N), 0.5, dtype=np.float32)
    if groups:
        gmap = pd.Series({c: (groups.get(c, "其他") or "其他") for c in mb.columns})
        members = {g: [mb.columns.get_loc(c) for c in cols] for g, cols in gmap.groupby(gmap).groups.items() if len(cols) >= 3}
        if len(members) >= 3:
            arr = mb.to_numpy(dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gm = pd.DataFrame({g: np.nanmean(arr[:, ix], axis=1) for g, ix in members.items()}, index=mb.index)
            rk = gm.rank(axis=1, pct=True)
            for g, ix in members.items():
                out[:, ix] = rk[g].to_numpy(dtype=np.float32)[:, None]
    return pd.DataFrame(out, index=mb.index, columns=mb.columns)


def compute_features(P: dict, bench_close: pd.Series, groups: dict | None = None) -> dict:
    C = P["Close"]
    f = {}
    f["trend"] = 0.25 * ((C > P["MA20"]).astype(float) + (P["MA20"] > P["MA60"]).astype(float)
                         + (P["MA20_slope5"] > 0).astype(float) + (P["EMA10"] > P["MA20"]).astype(float))
    f["compress"] = 0.6 * (1 - P["BB_Pct"]) + 0.4 * (P["Range5"] < 6).astype(float)
    vol_up = ((P["Volume"] > P["Vol_MA5"] * 1.2) & (C > P["Open"])).astype(float)
    f["mom"] = (0.3 * ((P["RSI"] >= 50) & (P["RSI"] <= 70)).astype(float)
                + 0.3 * (P["MACD_Hist"] > 0).astype(float)
                + 0.2 * ((P["K"] > P["D"]) & (P["K"] >= 50) & (P["K"] <= 82)).astype(float)
                + 0.2 * vol_up)
    # 相對強度：橫斷面動能百分位（強者恆強；比『贏大盤多少』更能抓到領漲股）
    mb = 0.5 * P["Ret60"] + 0.3 * P["Ret120"].fillna(P["Ret60"]) + 0.2 * P["Ret20"]
    f["rs"] = mb.rank(axis=1, pct=True)
    f["nh"] = ((P["Hi250"] - 0.7) / 0.3).clip(0, 1)               # 逼近52週新高
    f["sector"] = sector_strength(mb, groups or {})
    # 主力：真實外資+投信買超（有資料時）融合量價足跡（吸籌日/資金流/上漲量）；沒資料時退回量價足跡
    proxy = (0.35 * ((P["CMF20"] + 0.05) / 0.25).clip(0, 1)
             + 0.35 * ((P["AD_net20"] + 1) / 4).clip(0, 1)
             + 0.30 * ((P["UDVR"] - 0.8) / 1.2).clip(0, 1))
    real = 0.5 * ((P["MF10"] + 0.01) / 0.06).clip(0, 1) + 0.5 * (P["TR10"] / 0.02).clip(0, 1)
    f["main"] = (0.55 * real + 0.45 * proxy).where(P["MF10"].notna(), proxy)
    breakout = ((C > P["High20_prev"]) & (P["Volume"] > 1.3 * P["Vol_MA20"]) & (P["Close_loc"] > 0.6)).astype(float)
    pullback = ((P["MA20_slope5"] > 0) & (P["Bias10"].abs() <= 2.0) & (P["RSI"] >= 42) & (P["RSI"] <= 62)
                & (P["Volume"] < P["Vol_MA5"] * 0.9) & (C > P["MA20"])).astype(float)
    f["setup"] = np.maximum(breakout, 0.8 * pullback)
    return {k: v.astype(np.float32) for k, v in f.items()}


def composite_score(feats: dict, preset: str) -> pd.DataFrame:
    w = PRESETS[preset]
    tot = sum(w.values())
    return sum(feats[k] * v for k, v in w.items()) / tot * 100


def universe_masks(P: dict, flt: Filters):
    C = P["Close"]
    liquid = ((C > flt.min_price) & (P["Turnover_MA5"] >= flt.min_turnover) & (P["Amp20"] >= flt.min_amp)
              & (P["Volume"] > 0) & P["MA60"].notna() & P["ATR"].notna())
    trend_ok = (C > P["MA20"]) & (C > P["MA60"]) & (P["MA20_slope5"] > -0.005)
    heat_ok = (P["Bias5"] <= flt.bias5_max) & (P["Bias20"] <= flt.bias20_max)
    return liquid, liquid & trend_ok & heat_ok


def regime_array(bench_close: pd.Series, idx) -> np.ndarray:
    b = bench_close.reindex(idx).ffill()
    ma20, ma60 = b.rolling(20).mean(), b.rolling(60).mean()
    ok = (~((b < ma20) & (b < ma60))) & ma60.notna()
    return ok.to_numpy()


# ----------------------------------------------------------------------------
# 交易計畫（回測與即時推薦共用同一套邏輯）
# ----------------------------------------------------------------------------
def plan_trade(fill: float, atr: float, low5: float, stop_atr: float, tp_r: float):
    stop_raw = fill - stop_atr * atr
    struct = low5 * 0.99
    stop = max(stop_raw, struct)
    risk = float(np.clip((fill - stop) / fill, 0.03, 0.10))
    stop = tick_round(fill * (1 - risk))
    R = fill - stop
    tp = tick_round(fill + tp_r * R) if tp_r > 0 else float("inf")
    return stop, tp, R


def size_position(capital: float, risk_pct: float, entry: float, stop: float, max_alloc: float = 0.25) -> dict:
    per_share = max(entry - stop, 1e-9)
    by_risk = int(capital * risk_pct / 100.0 / per_share)
    by_alloc = int(capital * max_alloc / entry)
    shares = max(0, min(by_risk, by_alloc))
    return dict(shares=shares, lots=shares // 1000, odd=shares % 1000, amount=shares * entry,
                pct_capital=shares * entry / capital * 100 if capital else 0.0,
                risk_amt=shares * per_share, binding="風險上限" if by_risk <= by_alloc else "單檔25%資金上限")


# ----------------------------------------------------------------------------
# 引擎：把面板、分數、遮罩與排名打包
# ----------------------------------------------------------------------------
def regime_levels(bench_close: pd.Series, idx, P: dict):
    """大盤環境分三級：2 多頭 / 1 中性 / 0 空頭（加權指數同時跌破月線與季線）；市場寬度<30% 再降一級。"""
    b = bench_close.reindex(idx).ffill()
    ma20, ma60 = b.rolling(20).mean(), b.rolling(60).mean()
    strong = (b > ma20) & (ma20 > ma60)
    weak = (b < ma20) & (b < ma60)
    lvl = pd.Series(1, index=idx)
    lvl[strong] = 2
    lvl[weak] = 0
    lvl[ma60.isna()] = 0
    C = P["Close"]
    breadth = ((C > P["MA20"]).sum(axis=1) / C.notna().sum(axis=1).replace(0, np.nan)).fillna(0.5)
    lvl = lvl.where(breadth >= 0.30, (lvl - 1).clip(lower=0))
    return lvl.to_numpy().astype(int), breadth


CORE_FIELDS = ("Open", "High", "Low", "Close", "Volume", "ATR", "Low5")


class Engine:
    """輕量化引擎：完整歷史只保留回測/預測力檢驗必需的欄位；其餘指標只留最近 130 根供顯示。"""

    def __init__(self, P: dict, bench_close: pd.Series, flt: Filters, costs: Costs, capital: float = 1_000_000.0,
                 risk_frac: float = 0.015, groups: dict | None = None, sleeve_close: pd.Series | None = None):
        self.costs, self.capital, self.risk_frac, self.flt = costs, capital, risk_frac, flt
        self.idx = P["Close"].index
        self.codes = list(P["Close"].columns)
        self.groups = groups or {}
        feats = compute_features(P, bench_close, self.groups)
        self.scores = {k: composite_score(feats, k).astype(np.float32) for k in PRESETS}
        self.feats = {k: v.tail(5) for k, v in feats.items()}
        del feats
        self.elig_by_heat = {}
        for h, (b5, b20) in HEATS.items():
            lq, el = universe_masks(P, Filters(flt.min_price, flt.min_turnover, flt.min_amp, b5, b20))
            self.liquid, self.elig_by_heat[h] = lq, el
        self.elig = self.elig_by_heat["中"]
        self.regime_lvl, self.breadth = regime_levels(bench_close, self.idx, P)
        self.regime = self.regime_lvl > 0
        self.bench_close = bench_close
        sc = sleeve_close if sleeve_close is not None else bench_close
        self.sleeve = sc.reindex(self.idx).ffill().bfill()
        self.sleeve_ret = self.sleeve.pct_change().fillna(0.0).to_numpy()
        f64 = lambda k: P[k].to_numpy(dtype=np.float64)
        self.arr = dict(O=f64("Open"), H=f64("High"), L=f64("Low"), C=f64("Close"),
                        Cf=P["Close"].ffill().to_numpy(dtype=np.float64), ATR=f64("ATR"), LOW5=f64("Low5"))
        self.P = {k: (v if k in CORE_FIELDS else v.tail(130)) for k, v in P.items()}
        self._rank_cache = {}
        self._memo = {}

    def ranks(self, preset: str, min_score: float, heat: str = "中", topk: int = 6):
        key = (preset, float(min_score), heat, topk)
        if key not in self._rank_cache:
            sc = self.scores[preset]
            S = sc.where(self.elig_by_heat[heat] & (sc >= min_score)).to_numpy()
            filled = np.where(np.isnan(S), -np.inf, S)
            order = np.argsort(-filled, axis=1)[:, :topk]
            self._rank_cache[key] = (order, np.take_along_axis(S, order, axis=1))
        return self._rank_cache[key]


def eng_memo(eng: Engine, key, fn):
    """重運算（快照、預測力、校準、族群表）只算一次，頁面重繪不再重算。"""
    if key not in eng._memo:
        eng._memo[key] = fn()
    return eng._memo[key]


def bench_window(eng: Engine, a: int, b: int):
    """同一視窗『大盤(0050)買進持有』的報酬% 與最大回撤%。"""
    s = eng.sleeve.iloc[max(a - 1, 0):b]
    if len(s) < 2:
        return np.nan, np.nan
    curve = s / s.iloc[0]
    return (curve.iloc[-1] - 1) * 100, abs((curve / curve.cummax() - 1).min()) * 100


# ----------------------------------------------------------------------------
# 模擬器：主動選股 + 閒置資金持有大盤(0050) + 大盤擇時
# ----------------------------------------------------------------------------
MAX_BY_LEVEL = {0: 0, 1: 2, 2: 4}
SLEEVE_FRIC = 0.0015


def simulate(eng: Engine, params: Params, start: int, end: int, max_pos: int = 4):
    A = eng.arr
    O, H, L, C, Cf, ATR, LOW5 = A["O"], A["H"], A["L"], A["C"], A["Cf"], A["ATR"], A["LOW5"]
    order, valid = eng.ranks(params.preset, params.min_score, params.heat)
    lvl, SR, costs = eng.regime_lvl, eng.sleeve_ret, eng.costs
    fee, slip, tax = costs.fee, costs.slip, costs.tax
    init = eng.capital
    cash, pos, trades, eq_vals, eq_idx = init, [], [], [], []
    expo, sleeve_days, lv0_days = [], 0, 0
    start, end = max(start, 1), min(end, len(eng.idx))
    sleeve_prev = None

    def close_pos(p, d, px, reason, sleeve_on):
        nonlocal cash
        proceeds = p["sh"] * px * (1 - slip - fee - tax)
        cash += proceeds - (proceeds * SLEEVE_FRIC if sleeve_on else 0.0)
        net = (px * (1 - slip - fee - tax) / (p["fill"] * (1 + fee)) - 1) * 100
        trades.append(dict(code=eng.codes[p["j"]], entry_date=eng.idx[p["d0"]], exit_date=eng.idx[d],
                           fill=p["fill"], exit=px, reason=reason, days=d - p["d0"] + 1, net_pct=net,
                           r_mult=(px - p["fill"]) / p["R"], mfe_r=(p["hh"] - p["fill"]) / p["R"],
                           mae_r=(p["fill"] - p["ll"]) / p["R"]))

    for d in range(start, end):
        lv = int(lvl[d - 1])
        sleeve_on = (not params.bear_cash) or lv >= 1
        if sleeve_prev is None:
            if sleeve_on:
                cash *= (1 - fee)
        elif sleeve_on != sleeve_prev:
            cash *= (1 - (fee + (0.001 if not sleeve_on else 0.0)))
        sleeve_prev = sleeve_on
        sleeve_days += sleeve_on
        lv0_days += (lv == 0)

        # 1) 開盤進場（訊號來自前一日收盤）；倉位 = min(主動額度/4, 風險預算/停損距離, 現金)
        allowed = 0 if params.active <= 0 else min(max_pos, MAX_BY_LEVEL[lv])
        if len(pos) < allowed:
            held = {p["j"] for p in pos}
            for k in range(order.shape[1]):
                if np.isnan(valid[d - 1, k]):
                    break
                j = int(order[d - 1, k])
                if j in held:
                    continue
                o, pc, atr, low5 = O[d, j], C[d - 1, j], ATR[d - 1, j], LOW5[d - 1, j]
                if np.isnan(o) or np.isnan(pc) or np.isnan(atr) or np.isnan(low5):
                    continue
                if o / pc - 1 >= 0.09:      # 開盤鎖漲停，買不到
                    continue
                fill = o * (1 + slip)
                stop, tp, R = plan_trade(fill, atr, low5, params.stop_atr, params.tp_r)
                equity = cash + sum(p["sh"] * Cf[d - 1, p["j"]] for p in pos)
                alloc = min(equity * params.active / max_pos, cash, equity * eng.risk_frac / max(R / fill, 1e-9))
                sh = int(alloc / (fill * (1 + fee)))
                if sh <= 0:
                    continue
                cost = sh * fill * (1 + fee)
                cash -= cost + (cost * SLEEVE_FRIC if sleeve_on else 0.0)
                pos.append(dict(j=j, d0=d, fill=fill, stop=stop, tp=tp, R=R, sh=sh, atr=atr, hh=fill, ll=fill))
                break

        # 2) 盤中出場（停損優先於停利；保守）
        for p in list(pos):
            j = p["j"]
            o, h, l, c = O[d, j], H[d, j], L[d, j], C[d, j]
            if np.isnan(c):
                continue
            p["hh"], p["ll"] = max(p["hh"], h), min(p["ll"], l)
            px = reason = None
            if o <= p["stop"]:
                px, reason = o, "跳空停損"
            elif l <= p["stop"]:
                px = p["stop"]
                reason = "停損" if p["stop"] < p["fill"] else ("移動停利" if p["stop"] > p["fill"] * 1.01 else "保本出場")
            elif o >= p["tp"]:
                px, reason = o, "跳空達標"
            elif h >= p["tp"]:
                px, reason = p["tp"], "達標停利"
            elif d - p["d0"] + 1 >= params.hold:
                px, reason = c, "期滿出場"
            if px is not None:
                close_pos(p, d, px, reason, sleeve_on)
                pos.remove(p)
            else:                               # 停損上移：隔日起生效
                if p["hh"] >= p["fill"] + 1.0 * p["R"]:
                    p["stop"] = max(p["stop"], tick_round(p["fill"] * 1.006))
                if p["hh"] >= p["fill"] + 1.5 * p["R"]:
                    p["stop"] = max(p["stop"], tick_round(p["hh"] - params.trail_atr * p["atr"]))

        if sleeve_on:                           # 閒置資金吃大盤報酬
            cash *= (1 + SR[d])
        stock_val = sum(p["sh"] * Cf[d, p["j"]] for p in pos)
        eq = cash + stock_val
        expo.append(stock_val / eq if eq > 0 else 0.0)
        eq_vals.append(eq)
        eq_idx.append(eng.idx[d])

    for p in list(pos):  # 視窗結束強制平倉
        close_pos(p, end - 1, Cf[end - 1, p["j"]], "視窗結束", bool(sleeve_prev))
    if eq_vals:
        cash *= (1 - (fee + 0.001)) if sleeve_prev else 1.0
        eq_vals[-1] = cash
    n = max(end - start, 1)
    info = dict(exposure=float(np.mean(expo)) if expo else 0.0, sleeve_frac=sleeve_days / n, lvl0_frac=lv0_days / n)
    return pd.DataFrame(trades), pd.Series(eq_vals, index=eq_idx, dtype=float) / init, info


# ----------------------------------------------------------------------------
# 績效統計
# ----------------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (ctr - half) * 100, (ctr + half) * 100


def summarize(trades: pd.DataFrame, equity: pd.Series | None = None) -> dict:
    s = dict(n=len(trades), win_rate=np.nan, avg_win=np.nan, avg_loss=np.nan, payoff=np.nan, expectancy=np.nan,
             profit_factor=np.nan, tstat=np.nan, wr_lo=np.nan, wr_hi=np.nan, max_consec_loss=0, avg_hold=np.nan,
             total_ret=np.nan, cagr=np.nan, mdd=np.nan, sharpe=np.nan, exp_lo=np.nan, exp_hi=np.nan)
    if len(trades):
        r = trades["net_pct"].to_numpy()
        n = len(r)
        wins, losses = r[r > 0], r[r <= 0]
        s["win_rate"] = len(wins) / n * 100
        s["avg_win"] = wins.mean() if len(wins) else 0.0
        s["avg_loss"] = losses.mean() if len(losses) else 0.0
        s["payoff"] = abs(s["avg_win"] / s["avg_loss"]) if s["avg_loss"] < 0 else np.nan
        s["expectancy"] = r.mean()
        s["profit_factor"] = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else np.nan
        sd = r.std(ddof=1) if n > 1 else np.nan
        s["tstat"] = r.mean() / (sd / np.sqrt(n)) if (n > 1 and sd and sd > 0) else np.nan
        s["wr_lo"], s["wr_hi"] = _wilson(len(wins), n)
        run = best = 0
        for x in r:
            run = run + 1 if x <= 0 else 0
            best = max(best, run)
        s["max_consec_loss"] = best
        s["avg_hold"] = trades["days"].mean()
        if n >= 5:
            rng = np.random.default_rng(7)
            bs = rng.choice(r, size=(2000, n), replace=True).mean(axis=1)
            s["exp_lo"], s["exp_hi"] = np.percentile(bs, [5, 95])
    if equity is not None and len(equity) > 1:
        s["total_ret"] = (equity.iloc[-1] - 1) * 100
        s["mdd"] = abs((equity / equity.cummax() - 1).min()) * 100
        if len(equity) >= 120:
            s["cagr"] = (equity.iloc[-1] ** (TRADING_DAYS / len(equity)) - 1) * 100
        dr = equity.pct_change().dropna()
        if dr.std() > 0:
            s["sharpe"] = dr.mean() / dr.std() * np.sqrt(TRADING_DAYS)
    return s


def bench_return(bench_close: pd.Series, index) -> tuple[pd.Series, float]:
    b = bench_close.reindex(index).ffill().dropna()
    if len(b) < 2:
        return pd.Series(dtype=float), np.nan
    curve = b / b.iloc[0]
    return curve, (curve.iloc[-1] - 1) * 100


def mc_ruin(trade_rets_pct, n_sims=2000, n_trades=60, frac=0.25, block=3, ruin=0.5, seed=42) -> dict:
    """區塊自助法（保留連續虧損的群聚性）；frac≈每筆佔資金比例。"""
    r = np.asarray(trade_rets_pct, dtype=float) / 100.0
    n = len(r)
    if n < 5:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n_trades / block))
    starts = rng.integers(0, n, size=(n_sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    seq = r[idx].reshape(n_sims, -1)[:, :n_trades]
    eq = np.concatenate([np.ones((n_sims, 1)), np.cumprod(1 + frac * seq, axis=1)], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = -(eq / peak - 1).min(axis=1) * 100
    return dict(curves=eq, end_mean=eq[:, -1].mean(), end_p5=np.percentile(eq[:, -1], 5),
                mdd95=np.percentile(mdd, 95), p_ruin=(eq.min(axis=1) <= ruin).mean() * 100)


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------
def _objective(st: dict, bw: tuple, params: Params, mode: str, min_trades: int) -> float:
    ret, mdd = st["total_ret"], st["mdd"]
    if np.isnan(ret):
        return -1e9
    if params.active > 0 and st["n"] < min_trades:
        return -1e9
    br, bd = bw
    ex = ret - (br if not np.isnan(br) else 0.0)
    dd_pen = max(0.0, (mdd if not np.isnan(mdd) else 0.0) - (bd if not np.isnan(bd) else 0.0))
    if mode == "sharpe":
        return st["sharpe"] if not np.isnan(st["sharpe"]) else -1e9
    if mode == "tstat":
        return st["tstat"] if not np.isnan(st["tstat"]) else -1e9
    if mode == "winrate":
        if ex <= 0 or st["n"] < min_trades or not st["expectancy"] > 0:
            return -1e9
        return st["win_rate"] - 0.2 * dd_pen
    return ex - 0.5 * dd_pen          # excess


_ORD_DIMS = ("min_score", "heat", "stop_atr", "tp_r", "trail_atr", "hold", "active")


def _ckey(c: Params):
    return (c.preset, c.min_score, c.heat, c.stop_atr, c.tp_r, c.trail_atr, c.hold, c.active, c.bear_cash)


def smooth_objectives(combos: list, objs: list, grid: dict) -> list:
    """參數高原平滑：組合分數 = 自己與『相鄰參數』的平均，避免挑到孤立的幸運尖峰。"""
    pos = {_ckey(c): i for i, c in enumerate(combos)}
    ok = [o for o in objs if o > -1e8]
    lo = min(ok) if ok else 0.0
    out = []
    for i, c in enumerate(combos):
        if objs[i] <= -1e8:
            out.append(-1e9)
            continue
        vals = [objs[i]]
        for name in _ORD_DIMS:
            lst = grid[name]
            cur = getattr(c, name)
            if cur not in lst:
                continue
            j = lst.index(cur)
            for dj in (-1, 1):
                if 0 <= j + dj < len(lst):
                    k = _ckey(Params(**{**c.__dict__, name: lst[j + dj]}))
                    if k in pos:
                        vals.append(max(objs[pos[k]], lo))
        out.append(float(np.mean(vals)))
    return out


def make_combos(grid: dict) -> list:
    combos = [Params(preset=a, min_score=b, heat=c, stop_atr=d, tp_r=e, trail_atr=f, hold=g, active=h, bear_cash=i)
              for a, b, c, d, e, f, g, h, i in product(grid["preset"], grid["min_score"], grid["heat"], grid["stop_atr"],
                                                       grid["tp_r"], grid["trail_atr"], grid["hold"], grid["active"],
                                                       grid["bear_cash"])]
    combos += [Params(active=0.0, bear_cash=b) for b in (False, True)]      # 純大盤擇時基準線
    return combos


def walk_forward(eng: Engine, grid: dict | None = None, train_days: int = 250, test_days: int = 80,
                 min_trades: int = 12, progress=None, warm: int = 70, obj_mode: str = "excess"):
    grid = grid or DEFAULT_GRID
    combos = make_combos(grid)
    T = len(eng.idx)
    folds, a = [], warm
    while a + train_days + test_days <= T:
        folds.append((a, a + train_days, a + train_days + test_days))
        a += test_days
    if not folds:
        return None

    total = (len(folds) * 2 + 1) * len(combos)
    step = 0
    rows, oos_trades, chained, fold_info, test_ranges = [], [], [], [], []
    level = 1.0
    sel_counter = {}

    for fi, (a, b, e) in enumerate(folds):
        tr_stats, te_stats, te_data = [], [], []
        bw_tr = bench_window(eng, a, b)
        for c in combos:
            t, q, _ = simulate(eng, c, a, b)
            tr_stats.append(summarize(t, q))
            t2, q2, inf2 = simulate(eng, c, b, e)
            te_stats.append(summarize(t2, q2))
            te_data.append((t2, q2, inf2))
            step += 2
            if progress:
                progress(min(step / total, 1.0))
        objs = [_objective(s, bw_tr, c, obj_mode, min_trades) for s, c in zip(tr_stats, combos)]
        sm = smooth_objectives(combos, objs, grid)
        bi = int(np.argmax(sm))
        if sm[bi] <= -1e8:
            rows.append(dict(fold=fi + 1, train=f"{eng.idx[a].date()}~{eng.idx[b-1].date()}",
                             test=f"{eng.idx[b].date()}~{eng.idx[e-1].date()}", params="訓練期條件不足，略過",
                             train_n=0, train_exp=np.nan, test_n=0, test_exp=np.nan, test_ret=np.nan,
                             bench_ret=np.nan, test_mdd=np.nan, rank_pct=np.nan))
            continue
        best = combos[bi]
        sel_counter[best.preset if best.active > 0 else "純大盤擇時"] = sel_counter.get(best.preset if best.active > 0 else "純大盤擇時", 0) + 1
        t2, q2, inf2 = te_data[bi]
        bw_te = bench_window(eng, b, e)
        # 過擬合診斷：所選參數在測試期的『超額報酬』排名百分位
        ex_all = np.array([(s["total_ret"] - bw_te[0]) if not np.isnan(s["total_ret"]) else np.nan for s in te_stats])
        me = ex_all[bi]
        valid = ~np.isnan(ex_all)
        rank_pct = (np.sum(ex_all[valid] < me) / valid.sum() * 100) if (valid.any() and not np.isnan(me)) else np.nan
        rows.append(dict(fold=fi + 1, train=f"{eng.idx[a].date()}~{eng.idx[b-1].date()}",
                         test=f"{eng.idx[b].date()}~{eng.idx[e-1].date()}", params=best.label(),
                         train_n=tr_stats[bi]["n"], train_exp=tr_stats[bi]["expectancy"],
                         test_n=te_stats[bi]["n"], test_exp=te_stats[bi]["expectancy"],
                         test_ret=te_stats[bi]["total_ret"], bench_ret=bw_te[0],
                         test_mdd=te_stats[bi]["mdd"], rank_pct=rank_pct))
        if len(t2):
            t2 = t2.copy()
            t2["fold"] = fi + 1
            oos_trades.append(t2)
        chained.append(q2 * level)
        level = chained[-1].iloc[-1]
        fold_info.append(dict(inf2, days=e - b, active=best.active))
        test_ranges.append((b, e))

    # 最新一個訓練視窗 → 用於「今日推薦」的參數
    a, b = T - train_days, T
    bw_f = bench_window(eng, a, b)
    fin_stats = []
    for c in combos:
        t, q, _ = simulate(eng, c, a, b)
        fin_stats.append(summarize(t, q))
        step += 1
        if progress:
            progress(min(step / total, 1.0))
    fobj = [_objective(s, bw_f, c, obj_mode, min_trades) for s, c in zip(fin_stats, combos)]
    fsm = smooth_objectives(combos, fobj, grid)
    fi_best = int(np.argmax(fsm))
    final_params, final_stats = combos[fi_best], fin_stats[fi_best]
    final_ok = fsm[fi_best] > -1e8

    oos_tr = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    oos_eq = pd.concat(chained) if chained else pd.Series(dtype=float)
    oos_stats = summarize(oos_tr, oos_eq if len(oos_eq) else None)
    fold_df = pd.DataFrame(rows)
    return dict(folds=fold_df, oos_trades=oos_tr, oos_equity=oos_eq, oos_stats=oos_stats,
                is_expectancy=fold_df["train_exp"].mean() if len(fold_df) else np.nan,
                rank_pct_median=fold_df["rank_pct"].median() if len(fold_df) else np.nan,
                preset_selection=sel_counter, n_folds=len(folds), n_combos=len(combos),
                final_params=final_params if final_ok else Params(), final_stats=final_stats, final_ok=final_ok,
                train_days=train_days, test_days=test_days, fold_info=fold_info, test_ranges=test_ranges,
                obj_mode=obj_mode)


def verdict(wf: dict, bench_ret_pct: float | None = None):
    """由數據動態產生結論。回傳 (level, headline, bullets)；level ∈ green / yellow / red。
    第一判準 = 樣本外是否贏過大盤；其次才是期望值、顯著性、過擬合。"""
    o = wf["oos_stats"]
    bullets, level = [], "green"
    order = {"green": 0, "yellow": 1, "red": 2}

    def down(new):
        nonlocal level
        if order[new] > order[level]:
            level = new

    n = o["n"]
    if np.isnan(o["total_ret"]):
        return "red", "樣本外沒有可評估的資料，無法驗證策略。", ["請放寬篩選條件、增加股票池或拉長資料期間。"]
    if bench_ret_pct is not None and not np.isnan(bench_ret_pct):
        diff = o["total_ret"] - bench_ret_pct
        if diff < 0:
            down("red" if diff < -10 else "yellow")
            bullets.append(f"樣本外策略 {o['total_ret']:+.1f}% vs 大盤買進持有 {bench_ret_pct:+.1f}%：落後 {abs(diff):.1f} 個百分點。")
        else:
            bullets.append(f"樣本外策略 {o['total_ret']:+.1f}% vs 大盤 {bench_ret_pct:+.1f}%：超額 {diff:+.1f} 個百分點。")
    if n == 0:
        bullets.append("樣本外幾乎都是純大盤擇時（沒有選股交易）：代表主動選股在訓練期沒有證明優勢，系統自動退回大盤。")
    else:
        if n < 30:
            down("yellow")
            bullets.append(f"樣本外僅 {n} 筆交易（<30），績效數字可信度有限。")
        exp = o["expectancy"]
        if exp <= 0:
            down("yellow")
            bullets.append(f"樣本外每筆期望值 {exp:+.2f}%（扣成本後）≤ 0：選股交易本身沒有優勢。")
        else:
            bullets.append(f"樣本外每筆期望值 {exp:+.2f}%（90% 自助信賴區間 {o['exp_lo']:+.2f}% ~ {o['exp_hi']:+.2f}%）。")
            if not np.isnan(o["exp_lo"]) and o["exp_lo"] <= 0:
                down("yellow")
                bullets.append("信賴區間下緣 ≤ 0：無法排除『只是運氣』。")
        if not np.isnan(o["tstat"]) and o["tstat"] < 1.65:
            down("yellow")
            bullets.append(f"t 值 {o['tstat']:.2f} < 1.65，統計顯著性不足。")
        ise = wf.get("is_expectancy", np.nan)
        if not np.isnan(ise) and ise > 0 and not np.isnan(exp) and exp < 0.5 * ise:
            down("yellow")
            bullets.append(f"樣本外期望值（{exp:+.2f}%）不到訓練期平均（{ise:+.2f}%）的一半，疑似過擬合。")
        if o["max_consec_loss"] >= 6:
            bullets.append(f"最大連虧 {o['max_consec_loss']} 次，部位大小要能承受。")
    rp = wf.get("rank_pct_median", np.nan)
    if not np.isnan(rp):
        if rp < 55:
            down("yellow")
            bullets.append(f"訓練期選出的參數在測試期『超額報酬』僅排第 {rp:.0f} 百分位（≈50 代表挑參數沒有增值）。")
        else:
            bullets.append(f"所選參數在測試期超額報酬排第 {rp:.0f} 百分位，優於多數候選參數。")
    if not np.isnan(o["mdd"]) and o["mdd"] > 20:
        down("yellow")
        bullets.append(f"樣本外最大回撤 {o['mdd']:.1f}%，偏大。")
    heads = {"green": "樣本外驗證：通過基本檢查（仍不保證未來獲利）。",
             "yellow": "樣本外驗證：有疑慮，建議縮小主動部位或僅作觀察。",
             "red": "樣本外驗證：未通過，策略目前沒有可證實的優勢。"}
    return level, heads[level], bullets


def diagnose(wf: dict, eng: Engine):
    """依本次樣本外交易，自動找出『為什麼輸大盤 / 勝率低』的原因。回傳 (findings, exit_table)。"""
    tr, o, items = wf["oos_trades"], wf["oos_stats"], []
    fi = wf.get("fold_info", [])
    tot_days = sum(f["days"] for f in fi) or 1
    expo = sum(f["exposure"] * f["days"] for f in fi) / tot_days
    lv0 = sum(f["lvl0_frac"] * f["days"] for f in fi) / tot_days
    items.append(("info", f"樣本外平均曝險：股票 {expo*100:.0f}%（其餘為大盤或現金）；大盤空頭(等級0)天數占 {lv0*100:.0f}%。"))

    # 空頭濾網踏空
    miss = np.nan
    if wf.get("test_ranges"):
        parts = []
        for b, e in wf["test_ranges"]:
            d = np.arange(b, e)
            d = d[(d >= 1) & (d < len(eng.idx))]
            parts.append(eng.sleeve_ret[d][eng.regime_lvl[d - 1] == 0])
        r0 = np.concatenate(parts) if parts else np.array([])
        if len(r0):
            miss = (np.prod(1 + r0) - 1) * 100
    if not np.isnan(miss) and lv0 > 0.05:
        if miss > 3:
            items.append(("red", f"空頭濾網期間大盤其實上漲 {miss:+.1f}%：『同時跌破月線與季線就空手』造成踏空，"
                                 "牛市中這是落後大盤的主因之一（可用「空頭仍續抱大盤」參數避免）。"))
        else:
            items.append(("info", f"空頭濾網期間大盤報酬 {miss:+.1f}%，濾網沒有造成明顯踏空。"))

    if len(tr) == 0:
        items.append(("info", "沒有選股交易可分析（策略退回純大盤擇時）。"))
        return items, pd.DataFrame()

    n = len(tr)
    sl = eng.sleeve
    s0 = sl.reindex(pd.DatetimeIndex(tr["entry_date"]), method="ffill").to_numpy()
    s1 = sl.reindex(pd.DatetimeIndex(tr["exit_date"]), method="ffill").to_numpy()
    alpha = tr["net_pct"].to_numpy() - (s1 / s0 - 1) * 100
    beat = float(np.mean(alpha > 0) * 100)
    ma = float(np.nanmean(alpha))
    if ma < 0:
        items.append(("red", f"選股本身沒有超額報酬：每筆平均比大盤同期 {ma:+.2f}%，只有 {beat:.0f}% 的交易贏過大盤。"
                             "代表目前的選股因子在這段期間無效（或被過濾條件擋掉強勢股）——這是輸大盤的核心原因，不是停損參數能救的。"))
    else:
        items.append(("info", f"選股每筆平均比大盤同期多 {ma:+.2f}%，{beat:.0f}% 的交易贏過大盤。"))

    stop_share = float(tr["reason"].isin(["停損", "跳空停損"]).mean() * 100)
    avg_days = float(tr["days"].mean())
    if o["win_rate"] < 45:
        items.append(("yellow", f"勝率 {o['win_rate']:.0f}%、停損出場占 {stop_share:.0f}%、平均只持有 {avg_days:.1f} 天。"
                                + ("停損出場比例偏高且持有很短，常見原因是停損貼近日內雜訊，或進場在短線過熱處。" if stop_share > 50 else "")))
    los = tr[tr["net_pct"] <= 0]
    if len(los):
        giveback = float((los["mfe_r"] >= 1.0).mean() * 100)
        if giveback > 30:
            items.append(("yellow", f"{giveback:.0f}% 的虧損單曾先獲利超過 1R 才反轉：獲利沒鎖住，可考慮更早上移停損。"))
    cost = eng.costs.round_trip * 100
    aw = o["avg_win"]
    if aw and aw > 0 and cost / aw > 0.15:
        items.append(("yellow", f"每筆來回成本約 {cost:.2f}%，占平均獲利 {aw:.2f}% 的 {cost/aw*100:.0f}%，交易太頻繁會被成本吃掉。"))
    ex = tr.groupby("reason").agg(筆數=("net_pct", "size"), 平均損益=("net_pct", "mean"), 平均持有日=("days", "mean"))
    ex["占比%"] = ex["筆數"] / n * 100
    return items, ex.sort_values("筆數", ascending=False).round(2)


# ----------------------------------------------------------------------------
# 訊號品質（預測力檢驗）
# ----------------------------------------------------------------------------
def _fwd(eng: Engine, horizon: int) -> pd.DataFrame:
    P = eng.P
    return P["Close"].shift(-horizon) / P["Open"].shift(-1) - 1


def signal_quality(eng: Engine, preset: str, horizon: int = 10) -> dict:
    fwd, S, U = _fwd(eng, horizon), eng.scores[preset], eng.liquid
    valid = U & fwd.notna() & S.notna()
    s = S.where(valid).rank(axis=1)
    f = fwd.where(valid).rank(axis=1)
    n = valid.sum(axis=1)
    sm, fm = s.sub(s.mean(axis=1), axis=0), f.sub(f.mean(axis=1), axis=0)
    ic = (sm * fm).sum(axis=1) / np.sqrt((sm ** 2).sum(axis=1) * (fm ** 2).sum(axis=1))
    ic = ic[n >= 20].dropna()
    ic_s = ic.iloc[::horizon]   # 降採樣以減輕重疊報酬的自相關
    res = dict(horizon=horizon, days=len(ic_s), ic_mean=np.nan, ic_ir=np.nan, ic_t=np.nan, quintile=pd.DataFrame())
    if len(ic_s) >= 5:
        res["ic_mean"] = ic_s.mean()
        res["ic_ir"] = ic_s.mean() / ic_s.std() if ic_s.std() > 0 else np.nan
        res["ic_t"] = ic_s.mean() / (ic_s.std() / np.sqrt(len(ic_s))) if ic_s.std() > 0 else np.nan
    b = np.ceil(s.div(n, axis=0) * 5).clip(1, 5)
    df = pd.DataFrame({"bucket": b.stack(), "fwd": fwd.where(valid).stack()}).dropna()
    if len(df):
        q = df.groupby("bucket")["fwd"].agg(平均報酬="mean", 勝率=lambda x: (x > 0).mean(), 樣本數="size")
        q["平均報酬"] *= 100
        q["勝率"] *= 100
        q.index = [f"Q{int(i)}" + ("（最低分）" if i == 1 else "（最高分）" if i == 5 else "") for i in q.index]
        res["quintile"] = q
    return res


def score_calibration(eng: Engine, preset: str, horizon: int) -> pd.DataFrame:
    fwd, S = _fwd(eng, horizon), eng.scores[preset]
    E = eng.elig & fwd.notna() & S.notna()
    df = pd.DataFrame({"score": S.where(E).stack(), "fwd": fwd.where(E).stack()}).dropna()
    if df.empty:
        return pd.DataFrame()
    df["band"] = pd.cut(df["score"], [0, 55, 65, 75, 85, 101], right=False,
                        labels=["<55", "55-65", "65-75", "75-85", "85+"])
    out = df.groupby("band", observed=True)["fwd"].agg(平均報酬="mean", 勝率=lambda x: (x > 0).mean(), 樣本數="size")
    out["平均報酬"] *= 100
    out["勝率"] *= 100
    return out


# ----------------------------------------------------------------------------
# 即時推薦 / 快照 / 診斷
# ----------------------------------------------------------------------------
def make_chip_fn(chips, weight: float = 1.0):
    """主力（外資+投信）近5日買賣超占5日成交值比 → 最多約 ±10 分的『即時加減分』（外資±6、投信±4）。
    歷史逐日資料另走 FinMind，已進入回測的『主力』因子；這裡的加減分只反映最新幾天。"""
    def fn(code: str, close: float, turnover5: float):
        b, det = 0.0, {}
        if chips is not None and len(chips) and code in chips.index and turnover5 > 0:
            r = chips.loc[code]
            denom = turnover5 * 5
            f_ratio = float(r["foreign"]) * close / denom
            t_ratio = float(r["trust"]) * close / denom
            b += float(np.clip(f_ratio / 0.05, -1, 1)) * 6 + float(np.clip(t_ratio / 0.03, -1, 1)) * 4
            det.update(foreign=float(r["foreign"]), trust=float(r["trust"]), total=float(r["total"]))
        return float(b * weight), det
    return fn


def make_news_fn(names: dict, loader, weight: float = 1.0):
    """新聞加減分（避雷為主）：利空每則 -2、利多每則 +1，範圍 -6~+3。無法回測，只作輔助。"""
    def fn(code: str):
        if weight <= 0:
            return 0.0, {}
        try:
            items = loader(code, names.get(code, code))
        except Exception:
            return 0.0, {}
        pos, neg = news_counts(items)
        return float(np.clip(pos - 2 * neg, -6, 3)) * weight, dict(pos=pos, neg=neg, items=items[:5])
    return fn


def latest_candidates(eng: Engine, params: Params, top: int = 10, exclude: set | None = None,
                      chip_fn=None, news_fn=None) -> list:
    S = eng.scores[params.preset].iloc[-1]
    E = eng.elig_by_heat[params.heat].iloc[-1]
    m = E & (S >= params.min_score)
    if exclude:
        m &= ~S.index.isin(list(exclude))
    P = eng.P
    pool = S[m].sort_values(ascending=False).index[:max(top * 3, 30)]
    out = []
    for code in pool:
        close = float(P["Close"][code].iloc[-1])
        fill = close * (1 + eng.costs.slip)
        stop, tp, R = plan_trade(fill, float(P["ATR"][code].iloc[-1]), float(P["Low5"][code].iloc[-1]),
                                 params.stop_atr, params.tp_r)
        turn5 = float(P["Turnover_MA5"][code].iloc[-1])
        bonus, det = chip_fn(code, close, turn5) if chip_fn else (0.0, {})
        tech = float(S[code])
        out.append(dict(code=code, score=tech, bonus=bonus, final=tech + bonus, chip=det, news={}, news_bonus=0.0,
                        group=eng.groups.get(code, ""), close=close, pct=float(P["pct"][code].iloc[-1]),
                        stop=stop, tp=tp, risk_pct=(fill - stop) / fill * 100, turnover_yi=turn5 / 1e8,
                        volume_lots=float(P["Volume"][code].iloc[-1]) / 1000,
                        bias5=float(P["Bias5"][code].iloc[-1]), bias20=float(P["Bias20"][code].iloc[-1]),
                        k=float(P["K"][code].iloc[-1]), rsi=float(P["RSI"][code].iloc[-1]),
                        sub={k: float(eng.feats[k][code].iloc[-1]) * 100
                             for k in ("trend", "compress", "mom", "rs", "main", "setup", "nh", "sector")}))
    out.sort(key=lambda r: r["final"], reverse=True)
    out = out[:max(top + 3, 12)]
    if news_fn:
        for r in out:
            nb, nd = news_fn(r["code"])
            r["news_bonus"], r["news"], r["final"] = nb, nd, r["final"] + nb
        out.sort(key=lambda r: r["final"], reverse=True)
    return out[:top]


def sector_table(eng: Engine, names: dict) -> pd.DataFrame:
    """族群強弱排行（依產業別，由股票池真實價量統計）。"""
    P = eng.P
    C = P["Close"]
    last = lambda df: df.iloc[-1]
    g = pd.Series({c: (eng.groups.get(c) or "其他") for c in eng.codes})
    liq = last(eng.liquid)
    r20, r60 = last(P["Ret20"]) * 100, last(P["Ret60"]) * 100
    a20, a60 = last(C) > last(P["MA20"]), last(C) > last(P["MA60"])
    main = last(eng.feats["main"]) * 100
    rows = []
    for name, cols in g.groupby(g).groups.items():
        cols = [c for c in cols if bool(liq.get(c, False))]
        if len(cols) < 3:
            continue
        lead = sorted(cols, key=lambda c: -(r60[c] if pd.notna(r60[c]) else -999))[:3]
        rows.append({"族群": name, "檔數": len(cols), "20日平均%": r20[cols].mean(), "60日平均%": r60[cols].mean(),
                     "站上月線%": a20[cols].mean() * 100, "站上季線%": a60[cols].mean() * 100,
                     "主力足跡": main[cols].mean(), "領漲股": "、".join(names.get(c, c) for c in lead)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["強度"] = df["20日平均%"].rank(pct=True) * 0.4 + df["60日平均%"].rank(pct=True) * 0.4 + df["站上月線%"].rank(pct=True) * 0.2
    return df.sort_values("強度", ascending=False).round(1).drop(columns="強度").reset_index(drop=True)


def detect_events(d: pd.DataFrame) -> list:
    """最新一根 K 棒相對前一根的交易事件（自選股/持股推播用）。"""
    x, p = d.iloc[-1], d.iloc[-2]
    c = float(x["Close"])
    vr = float(x["Volume"] / (x["Vol_MA20"] + 1e-9))
    body = abs(c - x["Open"]) / (x["High"] - x["Low"] + 1e-9)
    ev = []
    if c > x["High20_prev"] and vr > 1.3: ev.append(("breakout20", "🚀 突破20日新高且放量"))
    if c > x["MA20"] and p["Close"] <= p["MA20"]: ev.append(("up_ma20", "⬆ 站上月線"))
    if c < x["MA20"] and p["Close"] >= p["MA20"]: ev.append(("dn_ma20", "⬇ 跌破月線"))
    if c > x["MA60"] and p["Close"] <= p["MA60"]: ev.append(("up_ma60", "⬆ 站上季線"))
    if c < x["MA60"] and p["Close"] >= p["MA60"]: ev.append(("dn_ma60", "⬇ 跌破季線"))
    if x["K"] > x["D"] and p["K"] <= p["D"] and x["K"] < 80: ev.append(("kd_gold", "KD 黃金交叉"))
    if x["K"] < x["D"] and p["K"] >= p["D"] and p["K"] > 70: ev.append(("kd_dead", "KD 高檔死亡交叉"))
    if x["MACD_Hist"] > 0 and p["MACD_Hist"] <= 0: ev.append(("macd_up", "MACD 翻紅"))
    if x["MACD_Hist"] < 0 and p["MACD_Hist"] >= 0: ev.append(("macd_dn", "MACD 翻綠"))
    if vr > 2 and c > x["Open"] and body > 0.6: ev.append(("big_red", "🔥 爆量長紅"))
    if vr > 2 and c < x["Open"] and body > 0.6: ev.append(("big_black", "⚠ 爆量長黑"))
    if x["RSI"] > 75 and p["RSI"] <= 75: ev.append(("rsi_hot", "RSI 過熱(>75)"))
    if x["RSI"] < 30 and p["RSI"] >= 30: ev.append(("rsi_cold", "RSI 超賣(<30)"))
    return ev


SCREEN_FLAGS = [
    "多頭排列 (收盤>20MA>60MA)", "空轉多 (3日內站上月線)", "多頭擴大 (5>10>20>60MA)", "今日強勢漲停 (≥9.5%)",
    "突破週線 (3日內站上5MA)", "突破月線 (3日內站上20MA)", "突破季線 (3日內站上60MA)", "創20日新高",
    "VCP 收斂 (帶寬低檔+5日振幅<6%)", "窒息量縮 (量<20日均量60%)", "爆量攻擊 (量>5日均量1.5倍)",
    "量價鎖碼 (CLV推力>0.15)", "MACD 紅柱", "RSI 50–70 健康動能", "KD 黃金交叉 (K>D 且 K<80)",
]


def snapshot(eng: Engine, preset: str) -> pd.DataFrame:
    P = {k: v.tail(130) for k, v in eng.P.items()}
    C = P["Close"]

    def cross(a, b, k=3):
        x = ((a > b) & (a.shift(1) <= b.shift(1))).astype(float)
        return x.rolling(k).max().iloc[-1] > 0

    last = lambda df: df.iloc[-1]
    flags = {
        SCREEN_FLAGS[0]: last((C > P["MA20"]) & (P["MA20"] > P["MA60"])),
        SCREEN_FLAGS[1]: cross(C, P["MA20"]),
        SCREEN_FLAGS[2]: last((P["MA5"] > P["MA10"]) & (P["MA10"] > P["MA20"]) & (P["MA20"] > P["MA60"])),
        SCREEN_FLAGS[3]: last(P["pct"]) >= 9.5,
        SCREEN_FLAGS[4]: cross(C, P["MA5"]),
        SCREEN_FLAGS[5]: cross(C, P["MA20"]),
        SCREEN_FLAGS[6]: cross(C, P["MA60"]),
        SCREEN_FLAGS[7]: last(C > P["High20_prev"]),
        SCREEN_FLAGS[8]: last((P["BB_Pct"] <= 0.25) & (P["Range5"] < 6)),
        SCREEN_FLAGS[9]: last(P["Volume"] < P["Vol_MA20"] * 0.6),
        SCREEN_FLAGS[10]: last(P["Volume"] > P["Vol_MA5"] * 1.5),
        SCREEN_FLAGS[11]: last(P["CLV5"] > 0.15),
        SCREEN_FLAGS[12]: last(P["MACD_Hist"] > 0),
        SCREEN_FLAGS[13]: last((P["RSI"] >= 50) & (P["RSI"] <= 70)),
        SCREEN_FLAGS[14]: last((P["K"] > P["D"]) & (P["K"] < 80)),
    }
    out = pd.DataFrame({
        "close": last(C), "pct": last(P["pct"]), "volume_lots": last(P["Volume"]) / 1000,
        "turnover_yi": last(P["Turnover_MA5"]) / 1e8, "bias5": last(P["Bias5"]), "bias20": last(P["Bias20"]),
        "K": last(P["K"]), "RSI": last(P["RSI"]), "score": last(eng.scores[preset]),
        "liquid": last(eng.liquid), "eligible": last(eng.elig),
    })
    for k, v in flags.items():
        out[k] = v.astype(bool)
    return out


def bull_bear_checks(d: pd.DataFrame, chip: dict | None = None, rev_yoy: float | None = None) -> dict:
    """只列出『真的算得出來』的多空項目；沒資料就不顯示，不再寫死。"""
    bull, bear = [], []
    x, p = d.iloc[-1], d.iloc[-2]
    c = x["Close"]
    if c > x["MA5"]: bull.append("站上週線 5MA")
    if c > x["MA20"]: bull.append("站上月線 20MA")
    if c > x["MA60"]: bull.append("站上季線 60MA")
    if (d["Close"].tail(5) > d["MA20"].tail(5)).all(): bull.append("連5日站穩月線")
    if x["MA5"] > x["MA10"] > x["MA20"]: bull.append("短線多頭排列")
    if x["MA10"] > x["MA20"] > x["MA60"]: bull.append("長線多頭排列")
    if 50 <= x["K"] <= 80 and x["K"] > x["D"]: bull.append("KD 多方強勢")
    if x["MACD_Hist"] > 0: bull.append("MACD 紅柱")
    if 50 <= x["RSI"] <= 70: bull.append("RSI 健康動能")
    if x["Volume"] > x["Vol_MA5"] * 1.3 and c > p["Close"]: bull.append("價漲量增")
    if c < x["MA20"]: bear.append("跌破月線")
    if c < x["MA60"]: bear.append("跌破季線")
    if x["MA20"] < x["MA60"] and c < x["MA20"]: bear.append("均線空頭排列")
    if x["Bias5"] > 3.5: bear.append("短線乖離過大")
    if x["Bias20"] > 12: bear.append("月線乖離過熱")
    if x["RSI"] > 75: bear.append("RSI 過熱")
    if x["MACD_Hist"] < 0: bear.append("MACD 綠柱")
    if x["Volume"] > x["Vol_MA5"] * 1.3 and c < p["Close"]: bear.append("價跌量增")
    if chip:
        if chip.get("foreign", 0) > 0: bull.append("外資近5日買超")
        elif chip.get("foreign", 0) < 0: bear.append("外資近5日賣超")
        if chip.get("trust", 0) > 0: bull.append("投信近5日買超")
        elif chip.get("trust", 0) < 0: bear.append("投信近5日賣超")
    if rev_yoy is not None and not np.isnan(rev_yoy):
        if rev_yoy >= 20: bull.append(f"月營收年增 {rev_yoy:.0f}%")
        elif rev_yoy > 0: bull.append(f"月營收年增 {rev_yoy:.0f}%")
        else: bear.append(f"月營收年減 {rev_yoy:.0f}%")
    return dict(bull=bull, bear=bear)


def holding_health(d: pd.DataFrame, params: Params) -> dict:
    x, p = d.iloc[-1], d.iloc[-2]
    close = float(x["Close"])
    stop, tp, R = plan_trade(close, float(x["ATR"]), float(x["Low5"]), params.stop_atr, params.tp_r)
    flags, weak = [], 0
    if close < x["MA20"]: flags.append("跌破月線"); weak += 2
    if x["MA20_slope5"] < 0: flags.append("月線下彎"); weak += 1
    if close < x["MA60"]: flags.append("跌破季線"); weak += 2
    if x["RSI"] < 45: flags.append("RSI<45 動能轉弱"); weak += 1
    if close <= stop: flags.append("觸及防守價"); weak += 3
    return dict(close=close, stop=stop, tp=tp, weak=weak, flags=flags,
                ret20=float(close / d["Close"].iloc[-21] - 1) * 100 if len(d) > 21 else np.nan)


# ============================================================================
# 共用：題材庫、檔案存取、引擎建構、訊號與推播、背景模式
# ============================================================================
SECTORS = {
    "重電綠能與強韌電網": [("1513", "中興電", ["儲能", "離岸風電", "GIS"]), ("1609", "大亞", ["電纜", "儲能"]),
                          ("1519", "華城", ["變壓器", "外銷"]), ("1503", "士電", ["重電", "車用電裝"]),
                          ("1514", "亞力", ["變配電", "半導體廠務"]), ("2371", "大同", ["重電", "資產開發"])],
    "CPO 矽光子與光通訊": [("3450", "聯鈞", ["光收發", "CPO"]), ("3363", "上詮", ["光纖陣列", "CPO"]),
                          ("3081", "聯亞", ["磊晶", "InP"]), ("4979", "華星光", ["AOC", "光模組"]),
                          ("6442", "光聖", ["光被動"]), ("4977", "眾達-KY", ["CPO", "光模組"])],
    "散熱模組與水冷系統": [("3017", "奇鋐", ["水冷板", "散熱模組"]), ("3324", "雙鴻", ["液冷", "CDU"]),
                          ("3653", "健策", ["均熱片"]), ("8996", "高力", ["熱交換器"]),
                          ("6230", "尼得科超眾", ["熱導管"]), ("3483", "力致", ["散熱模組"])],
    "AI 伺服器與硬體代工": [("2382", "廣達", ["AI伺服器"]), ("3231", "緯創", ["GPU基板"]), ("2376", "技嘉", ["AI伺服器"]),
                           ("6669", "緯穎", ["雲端ODM"]), ("2059", "川湖", ["伺服器滑軌"]), ("8210", "勤誠", ["伺服器機殼"])],
    "PCB、載板與 CCL": [("2383", "台光電", ["CCL"]), ("3037", "欣興", ["ABF載板"]), ("2368", "金像電", ["高層PCB"]),
                       ("6274", "台燿", ["CCL"]), ("3189", "景碩", ["載板"]), ("8046", "南電", ["載板"])],
    "記憶體模組與控制晶片": [("8299", "群聯", ["NAND控制IC"]), ("3260", "威剛", ["記憶體模組"]), ("4967", "十銓", ["記憶體模組"]),
                            ("2408", "南亞科", ["DRAM"]), ("2344", "華邦電", ["NOR/利基DRAM"]), ("3006", "晶豪科", ["利基DRAM"])],
    "半導體製造與設備": [("2330", "台積電", ["晶圓代工"]), ("2303", "聯電", ["成熟製程"]), ("3583", "辛耘", ["濕製程設備"]),
                        ("3131", "弘塑", ["濕製程設備"]), ("6223", "旺矽", ["探針卡"]), ("1560", "中砂", ["鑽石碟"])],
    "機器人自動化與智慧工具機": [("2359", "所羅門", ["機器視覺"]), ("2365", "昆盈", ["感測"]), ("6188", "廣明", ["協作機器人"]),
                                ("8374", "羅昇", ["傳動控制"]), ("2049", "上銀", ["滾珠螺桿"]), ("4583", "台灣精銳", ["精密減速機"])],
    "AI 晶片與 IP": [("2454", "聯發科", ["手機AP", "ASIC"]), ("3661", "世芯-KY", ["ASIC"]), ("3443", "創意", ["ASIC", "IP"]),
                    ("3529", "力旺", ["矽智財"]), ("6117", "迎廣", ["機殼", "水冷"])],
}
NAME_MAP = {c: n for v in SECTORS.values() for c, n, _ in v}
TAG_MAP = {c: t for v in SECTORS.values() for c, _, t in v}

PORTFOLIO_FILE = STATE_DIR / "portfolio.json"
DEFAULT_PORTFOLIO = [
    {"code": "2330", "cost": 950.0, "shares": 1000, "date": "2026-09-15"},
    {"code": "3189", "cost": 115.0, "shares": 5000, "date": "2026-09-20"},
    {"code": "3653", "cost": 820.0, "shares": 1000, "date": "2026-09-22"},
    {"code": "1513", "cost": 165.0, "shares": 6000, "date": "2026-09-25"},
]


def load_portfolio() -> list:
    env = os.environ.get("PORTFOLIO_JSON", "").strip()
    if env:
        try:
            return json.loads(env)
        except Exception:
            pass
    try:
        return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    except Exception:
        return [] if os.environ.get("TQ_HEADLESS") else [dict(x) for x in DEFAULT_PORTFOLIO]


def save_portfolio(rows: list):
    PORTFOLIO_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


WATCHLIST_FILE = STATE_DIR / "watchlist.json"
BEST_PARAMS_FILE = STATE_DIR / "best_params.json"


def load_watchlist() -> list:
    env = os.environ.get("WATCHLIST", "").strip()
    if env:
        return [c for c in re.split(r"[\s,，、]+", env) if c]
    try:
        return [str(x).strip() for x in json.loads(WATCHLIST_FILE.read_text(encoding="utf-8")) if str(x).strip()]
    except Exception:
        return []


def save_watchlist(codes: list):
    WATCHLIST_FILE.write_text(json.dumps(codes, ensure_ascii=False), encoding="utf-8")


def save_best_params(p: Params, level: str = ""):
    try:
        BEST_PARAMS_FILE.write_text(json.dumps(dict(asdict(p), verdict=level, saved=now_tw().isoformat()),
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def fetch_state_json(name: str):
    """本機狀態檔 → 失敗則讀 STATE_BASE_URL（例如 state 分支的 raw 網址）。"""
    try:
        return json.loads((STATE_DIR / name).read_text(encoding="utf-8"))
    except Exception:
        pass
    base = get_secret("STATE_BASE_URL")
    if base:
        try:
            return requests.get(base.rstrip("/") + "/" + name, timeout=15).json()
        except Exception:
            pass
    return None


def load_best_params(fetch=None):
    raw = fetch("best_params.json") if fetch else fetch_state_json("best_params.json")
    try:
        p = Params(**{k: raw[k] for k in Params.__dataclass_fields__ if k in raw})
        return p if (p.preset in PRESETS and p.heat in HEATS) else None   # 舊版存檔的參數名稱不相容時忽略
    except Exception:
        return None


def load_meta_core(hour_key: str) -> dict:
    uni, m1 = load_universe()
    turn, m2 = load_daily_turnover()
    names, markets, m6 = load_names()
    chips, chip_days, m3 = load_institutional(5)
    rev, m4 = load_revenue_yoy()
    punish, notice, m5 = load_flags()
    official, m8 = load_official_close()
    if uni.empty and markets:     # 官方 ISIN 清單失敗時，用名稱來源重建清單
        uni = pd.DataFrame([dict(code=c, name=names.get(c, c), market=m, industry="") for c, m in markets.items()])
    return dict(uni=uni, turn=turn, names=names, markets=markets, chips=chips, chip_days=chip_days,
                rev=rev, punish=punish, notice=notice, official=official,
                msgs=m1 + m6 + m2 + m3 + m4 + m5 + m8)


def build_name_market(meta: dict):
    uni = meta["uni"]
    names = dict(NAME_MAP)                                  # 最低優先：題材庫
    names.update(dict(zip(uni["code"], uni["name"])))       # ISIN
    names.update(meta["names"])                             # 公司簡稱（最高優先）
    market = {**meta["markets"], **dict(zip(uni["code"], uni["market"]))}
    return names, market


def build_engine_core(meta, n, excl, min_price, min_turn_yi, min_amp, b5, b20, fee_disc, slip_pct,
                      risk_frac, use_intraday, get_prices_fn, inst_fn=None, keep_rows=None):
    uni, notes = meta["uni"].copy(), []
    if not uni.empty:
        uni = uni[~uni["code"].str.startswith("00")]      # 排除 ETF
        if excl:
            uni = uni[~uni["industry"].isin(list(excl))]
        if len(meta["turn"]):
            uni["turn"] = uni["code"].map(meta["turn"])
            uni = uni.dropna(subset=["turn"]).sort_values("turn", ascending=False).head(n)
        else:
            notes.append("無法取得成交值排名，改用題材清單內的股票。")
            uni = uni[uni["code"].isin(NAME_MAP)]
    else:
        notes.append("官方股票清單抓取失敗，改用題材清單並猜測市場別（可能有誤）。")
        uni = pd.DataFrame([dict(code=c, name=NAME_MAP[c], market="TW", industry="") for c in NAME_MAP])
    tickers = [f"{r.code}.{r.market}" for r in uni.itertuples()]
    prices = get_prices_fn(tuple(tickers + ["^TWII", "0050.TW"]))
    bench = prices.pop("^TWII", None)
    sleeve = prices.pop("0050.TW", None)
    if bench is None or not prices:
        raise RuntimeError("價格資料下載失敗（yfinance）。請稍後重試或檢查網路。")
    prices, rep = validate_prices(prices, meta.get("official"), bench.index[-1])
    notes.append(f"資料品質檢查：通過 {rep['ok']} 檔；剔除 過舊{rep['stale']}／成交量過薄{rep['thin']}／異常跳價{rep['jump']}／"
                 f"與官方收盤價不符{rep['mismatch']}" + ("（官方與 yfinance 日期不同步，已略過官方比對）" if rep["skipped"] else ""))
    if use_intraday:
        prices = {t: project_partial_volume(df) for t, df in prices.items()}
        notes.append(f"盤中模式：最後一根 K 棒成交量已依日內曲線外推（目前約 {session_fraction()*100:.0f}% 的全日量）。")
    else:
        prices = {t: closed_only(df) for t, df in prices.items()}
        bench = closed_only(bench)
        sleeve = closed_only(sleeve) if sleeve is not None else None
    if sleeve is None:
        notes.append("抓不到 0050 價格，閒置資金改用加權指數代替（略為高估）。")
    inst, imsgs = ({}, [])
    if inst_fn is not None:
        try:
            inst, imsgs = inst_fn(tuple(t.split(".")[0] for t in prices))
        except Exception as e:
            imsgs = [f"主力逐日資料取得失敗：{e}（改用量價足跡代理）"]
    notes += imsgs
    idx = bench.index[-keep_rows:] if keep_rows else bench.index
    dfs = {t.split(".")[0]: df for t, df in prices.items()}
    P = build_panel_stream(dfs, inst, idx)
    groups = dict(zip(uni["code"], uni["industry"])) if "industry" in uni else {}
    flt = Filters(min_price, min_turn_yi * 1e8, min_amp, b5, b20)
    eng = Engine(P, bench["Close"], flt, Costs(fee_disc=fee_disc, slip=slip_pct / 100.0), risk_frac=risk_frac,
                 groups=groups, sleeve_close=sleeve["Close"] if sleeve is not None else None)
    del prices, dfs, P
    trim_memory()
    return eng, bench, notes


def get_ind_map(codes: list, market: dict, get_prices_fn) -> dict:
    """任意代碼 → 含最新（含盤中）K 線的指標表；市場別猜錯會自動換另一個後綴重試。"""
    codes = [c for c in dict.fromkeys(str(c).strip() for c in codes) if c]
    if not codes:
        return {}
    out, miss = {}, []
    first = {c: f"{c}.{market.get(c, 'TW')}" for c in codes}
    px = get_prices_fn(tuple(first.values()))
    for c, t in first.items():
        if t in px:
            out[c] = add_indicators(px[t])
        else:
            miss.append(c)
    if miss:
        alt = {c: f"{c}.{'TWO' if market.get(c, 'TW') == 'TW' else 'TW'}" for c in miss}
        px2 = get_prices_fn(tuple(alt.values()))
        for c, t in alt.items():
            if t in px2:
                out[c] = add_indicators(px2[t])
    return out


def assemble_ctx(meta, market, names, eng, params, get_prices_fn, capital, risk_pct, chip_weight,
                 portfolio_rows, watch_codes, news_fn=None) -> dict:
    hold_codes = [str(r.get("code", "")).strip() for r in portfolio_rows]
    ind_map = get_ind_map(hold_codes + list(watch_codes), market, get_prices_fn)
    return dict(meta=meta, names=names, market=market, capital=capital, risk_pct=risk_pct,
                portfolio=portfolio_rows, watch=list(watch_codes), ind_map=ind_map,
                chip_fn=make_chip_fn(meta["chips"], chip_weight), news_fn=news_fn,
                punish=meta["punish"], notice=meta["notice"])


# ----------------------------------------------------------------------------
# 訊號建立與推播
# ----------------------------------------------------------------------------
@dataclass
class Alert:
    key: str
    cat: str       # regime / cand / hold / watch
    text: str


ALERT_CATS = {"regime": "大盤環境轉折", "cand": "新進推薦", "hold": "持股警示", "watch": "自選股事件"}
LVL_TXT = {2: "🟢 多頭（最多4檔）", 1: "🟡 中性（最多2檔）", 0: "🔴 空頭（不新進場）"}


def build_alerts(eng: Engine, params: Params, ctx: dict) -> list:
    names = ctx["names"]
    lab = lambda c: f"{names.get(c, '')}({c})" if names.get(c) else str(c)
    out = []
    date_s = eng.idx[-1].strftime("%Y-%m-%d")
    today_s = now_tw().strftime("%Y-%m-%d")
    lvl = int(eng.regime_lvl[-1])

    prev = alert_state_get("regime_lvl")
    if prev is not None and int(prev) != lvl:
        out.append(Alert(f"regime|{today_s}|{lvl}", "regime",
                         f"🧭 大盤環境轉變：{LVL_TXT[int(prev)]} → {LVL_TXT[lvl]}（加權指數 {float(eng.bench_close.iloc[-1]):,.0f}）"))
    alert_state_set("regime_lvl", lvl)

    if lvl > 0:
        for i, c in enumerate(latest_candidates(eng, params, top=5, exclude=ctx["punish"], chip_fn=ctx["chip_fn"],
                                                 news_fn=ctx.get("news_fn")), 1):
            size = size_position(ctx["capital"], ctx["risk_pct"], c["close"], c["stop"])
            tp_txt = "移動停利" if np.isinf(c["tp"]) else f"{c['tp']:.2f}"
            det = c["chip"]
            chip_txt = ""
            if "foreign" in det:
                chip_txt += f"外資5日{det['foreign']/1000:+,.0f}張 投信{det['trust']/1000:+,.0f}張 "
            nw = c.get("news") or {}
            if nw.get("neg"):
                chip_txt += f"⚠負面新聞{nw['neg']}則 "
            elif nw.get("pos"):
                chip_txt += f"正面新聞{nw['pos']}則 "
            out.append(Alert(f"cand|{date_s}|{c['code']}", "cand",
                             f"🆕 推薦#{i} {lab(c['code'])}［{c['group'] or '—'}］最終分{c['final']:.0f}（技術{c['score']:.0f} 主力{c['bonus']:+.0f} 新聞{c['news_bonus']:+.0f}）\n"
                             f"收 {c['close']:.2f}｜停損 {c['stop']:.2f}｜停利 {tp_txt}｜風險 {c['risk_pct']:.1f}%\n"
                             f"建議 {size['lots']}張{size['odd']}股 ≈ {size['amount']:,.0f}元\n"
                             f"{chip_txt}\nhttps://tw.stock.yahoo.com/quote/{c['code']}"))

    for r in ctx["portfolio"]:
        code = str(r.get("code", "")).strip()
        d = ctx["ind_map"].get(code)
        if not code or d is None or len(d) < 65:
            continue
        ddate = d.index[-1].strftime("%Y-%m-%d")
        h = holding_health(d, params)
        close = h["close"]
        try:
            cost = float(r.get("cost") or 0)
        except Exception:
            cost = 0.0
        if cost > 0:
            st_c, tp_c, _ = plan_trade(cost, float(d["ATR"].iloc[-1]), float(d["Low5"].iloc[-1]),
                                       params.stop_atr, params.tp_r if params.tp_r > 0 else 2.0)
            pnl = (close / cost - 1) * 100
            if close <= st_c:
                out.append(Alert(f"hold_stop|{ddate}|{code}", "hold",
                                 f"🚨 持股 {lab(code)} 現價 {close:.2f} 已跌破以成本計算的防守價 {st_c:.2f}（成本 {cost:.2f}，{pnl:+.1f}%）"))
            elif close >= tp_c:
                out.append(Alert(f"hold_tp|{ddate}|{code}", "hold",
                                 f"🎯 持股 {lab(code)} 現價 {close:.2f} 達停利參考 {tp_c:.2f}（成本 {cost:.2f}，{pnl:+.1f}%），可考慮分批/上移停損。"))
        if h["weak"] >= 2:
            out.append(Alert(f"hold_weak|{ddate}|{code}|{h['weak']}", "hold",
                             f"⚠ 持股 {lab(code)} 轉弱（弱勢分{h['weak']}）：{'、'.join(h['flags'])}｜現價 {close:.2f}"))
        for k, txt in detect_events(d):
            if k in ("dn_ma20", "dn_ma60", "big_black", "kd_dead", "macd_dn"):
                out.append(Alert(f"hold_ev|{ddate}|{code}|{k}", "hold", f"⚠ 持股 {lab(code)} {txt}（收 {close:.2f}）"))

    for code in ctx["watch"]:
        d = ctx["ind_map"].get(code)
        if d is None or len(d) < 65:
            continue
        ddate = d.index[-1].strftime("%Y-%m-%d")
        close, pct = float(d["Close"].iloc[-1]), float(d["pct"].iloc[-1])
        for k, txt in detect_events(d):
            out.append(Alert(f"watch|{ddate}|{code}|{k}", "watch", f"👀 {lab(code)} {txt}（收 {close:.2f}，{pct:+.2f}%）"))
    return out


def dispatch_alerts(alerts: list, cfg: dict, enabled=None):
    enabled = set(enabled or ALERT_CATS)
    sel = [a for a in alerts if a.cat in enabled]
    if not sel:
        return "本次沒有符合條件的訊號。", []
    if not ((cfg.get("tg_token") and cfg.get("tg_chat")) or cfg.get("line_token")):
        return "尚未設定 Telegram / LINE，訊號未推送。", sel
    fresh_keys = set(alert_claim([a.key for a in sel]))
    fresh = [a for a in sel if a.key in fresh_keys]
    if not fresh:
        return "沒有新的訊號（已推送過的不會重複）。", []
    head = f"📡 台股量化訊號 {now_tw():%m/%d %H:%M}\n\n"
    pub = "\n\n".join(a.text for a in fresh if a.cat not in PRIVATE_CATS)
    pri = "\n\n".join(a.text for a in fresh if a.cat in PRIVATE_CATS)
    res = notify_routed(head + pub if pub else "", ("🔒 持股警示\n\n" + pri) if pri else "", cfg)
    if not any(ok for ok, _ in res):
        alert_release(list(fresh_keys))
    return "；".join(m for _, m in res) or "無推播結果（LINE 未設定 User ID 時，持股警示不會經 LINE 發送）", fresh


# ----------------------------------------------------------------------------
# 排程 / 常駐模式（不啟動網頁，可在 GitHub Actions 或任何主機跑，記憶體不受 1GB 限制）
#   python tw_quant_app.py --scan-once   掃描寬股票池、推播新訊號、寫 latest_scan.json
#   python tw_quant_app.py --wf-once     重跑 Walk-Forward，寫 best_params.json / wf_summary.json
#   python tw_quant_app.py --daemon      本機常駐：交易時段每 20 分鐘 scan
# 環境變數：TELEGRAM_BOT_TOKEN/CHAT_ID、LINE_CHANNEL_TOKEN/LINE_USER_ID/LINE_BROADCAST、FINMIND_TOKEN、
#          N_UNIVERSE(預設1200)、N_BT(預設300)、CAPITAL、RISK_PCT、CHIP_WEIGHT、NEWS_WEIGHT、OBJ_MODE、
#          STATE_DIR（狀態檔資料夾）、WATCHLIST（逗號分隔代碼）、PORTFOLIO_JSON
# ----------------------------------------------------------------------------
SCAN_EXCL = ["金融保險業", "航運業", "水泥工業"]


def in_trading_window(now: dt.datetime) -> bool:
    return now.weekday() < 5 and dt.time(8, 50) <= now.time() <= dt.time(14, 30)


def in_session(now: dt.datetime) -> bool:
    return now.weekday() < 5 and dt.time(8, 55) <= now.time() < dt.time(14, 10)


def write_latest_scan(eng, params, ctx, names, notes):
    cands = latest_candidates(eng, params, top=30, exclude=ctx["punish"], chip_fn=ctx["chip_fn"])
    sc = eng.scores[params.preset].iloc[-1]
    n_ok = int((eng.elig_by_heat[params.heat].iloc[-1] & (sc >= params.min_score)).sum())
    rows = [dict(code=c["code"], name=names.get(c["code"], ""), group=c["group"], final=round(c["final"], 1),
                 tech=round(c["score"], 1), main=round(c["bonus"], 1), close=round(c["close"], 2), pct=round(c["pct"], 2),
                 stop=c["stop"], tp=None if np.isinf(c["tp"]) else c["tp"], risk=round(c["risk_pct"], 2),
                 turnover=round(c["turnover_yi"], 2)) for c in cands]
    payload = dict(updated=now_tw().isoformat(timespec="minutes"), data_date=eng.idx[-1].strftime("%Y-%m-%d"),
                   universe=len(eng.codes), eligible=n_ok, regime=int(eng.regime_lvl[-1]), params=params.label(),
                   notes=notes[:6], candidates=rows)
    try:
        (STATE_DIR / "latest_scan.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def run_scan_cycle(get_prices_fn, meta_fn, inst_fn, news_loader):
    now = now_tw()
    cfg = notify_cfg()
    n = int(os.environ.get("N_UNIVERSE", "1200"))
    cap = float(os.environ.get("CAPITAL", "1000000"))
    rp = float(os.environ.get("RISK_PCT", "1.5"))
    mode = os.environ.get("USE_INTRADAY", "auto")
    use_intraday = in_session(now) if mode == "auto" else mode == "1"
    meta = meta_fn(now.strftime("%Y-%m-%d-%H"))
    names, market = build_name_market(meta)
    params = load_best_params() or Params()
    eng, _, notes = build_engine_core(meta, n, SCAN_EXCL, 35.0, 1.5, 8.0, 6.0, 15.0, 0.6, 0.10, rp / 100.0,
                                      use_intraday, get_prices_fn, inst_fn=inst_fn, keep_rows=340)
    nf = make_news_fn(names, news_loader, float(os.environ.get("NEWS_WEIGHT", "1")))
    ctx = assemble_ctx(meta, market, names, eng, params, get_prices_fn, cap, rp,
                       float(os.environ.get("CHIP_WEIGHT", "1")), load_portfolio(), load_watchlist(), news_fn=nf)
    alerts = build_alerts(eng, params, ctx)
    msg, fresh = dispatch_alerts(alerts, cfg)
    write_latest_scan(eng, params, ctx, names, notes)
    print(f"[{now:%m-%d %H:%M}] 股票池 {len(eng.codes)} 檔｜資料日 {eng.idx[-1]:%Y-%m-%d}｜"
          f"新訊號 {len(fresh)} 則｜{msg}｜記憶體 {rss_mb():.0f}MB", flush=True)
    for m in notes:
        print("  •", m, flush=True)


def run_wf_cycle():
    now = now_tw()
    cfg = notify_cfg()
    n = int(os.environ.get("N_BT", "300"))
    meta = load_meta_core(now.strftime("%Y-%m-%d-%H"))
    names, market = build_name_market(meta)
    inst_fn = lambda codes: load_inst_history(list(codes), token=get_secret("FINMIND_TOKEN"), max_fetch=400)
    eng, _, notes = build_engine_core(meta, n, SCAN_EXCL, 35.0, 1.5, 8.0, 6.0, 15.0, 0.6, 0.10,
                                      float(os.environ.get("RISK_PCT", "1.5")) / 100.0, False,
                                      lambda tk: download_prices(list(tk), "4y"), inst_fn=inst_fn)
    wf = walk_forward(eng, train_days=250, test_days=80, obj_mode=os.environ.get("OBJ_MODE", "excess"))
    if wf is None:
        print("資料不足，無法 walk-forward", flush=True)
        return
    bq = eng.sleeve.reindex(wf["oos_equity"].index)
    bret = (bq.iloc[-1] / bq.iloc[0] - 1) * 100 if len(bq) > 1 else np.nan
    lv, hd, bl = verdict(wf, bret)
    save_best_params(wf["final_params"], lv)
    o = wf["oos_stats"]
    summary = dict(updated=now.isoformat(timespec="minutes"), level=lv, headline=hd, bullets=bl, params=wf["final_params"].label(),
                   oos_return=o["total_ret"], bench_return=bret, trades=o["n"], win_rate=o["win_rate"], mdd=o["mdd"])
    try:
        (STATE_DIR / "wf_summary.json").write_text(json.dumps(summary, ensure_ascii=False, default=float), encoding="utf-8")
    except Exception:
        pass
    text = (f"🧪 Walk-Forward 週報 {now:%m/%d}\n{hd}\n策略 {o['total_ret']:+.1f}% vs 大盤 {bret:+.1f}%｜交易 {o['n']} 筆"
            f"｜最大回撤 {o['mdd']:.1f}%\n本週採用參數：{wf['final_params'].label()}")
    notify_routed("", text, cfg)
    print(text, flush=True)


def cli_main(flag: str):
    os.environ["TQ_HEADLESS"] = "1"
    inst_fn = lambda codes: load_inst_history(list(codes), token=get_secret("FINMIND_TOKEN"), max_fetch=400)
    news = lambda code, name: load_news(f"{name} {code}")
    if flag == "--wf-once":
        run_wf_cycle()
    elif flag == "--scan-once":
        if not in_trading_window(now_tw()) and os.environ.get("FORCE") != "1":
            print("非交易時段，略過（FORCE=1 可強制執行）。", flush=True)
            return
        run_scan_cycle(lambda tk: download_prices(list(tk), "2y", keep_rows=340), load_meta_core, inst_fn, news)
    elif flag == "--daemon":
        hist = lru_cache(maxsize=2)(lambda tk, day: download_prices(list(tk), "2y", keep_rows=340))
        rec = lru_cache(maxsize=2)(lambda tk, bk: download_prices(list(tk), "7d", min_len=2))
        meta_fn = lru_cache(maxsize=2)(load_meta_core)
        news_c = lru_cache(maxsize=1024)(lambda code, name, day: load_news(f"{name} {code}"))
        print("daemon 啟動；交易時段每 20 分鐘掃描一次。", flush=True)
        while True:
            now = now_tw()
            if in_trading_window(now):
                bk = now.strftime("%Y-%m-%d-%H-") + str(now.minute // 20)
                try:
                    run_scan_cycle(lambda tk, bk=bk: merge_recent(hist(tuple(tk), bk[:10]), rec(tuple(tk), bk)),
                                   meta_fn, inst_fn, lambda code, name, d=bk[:10]: news_c(code, name, d))
                except Exception as e:
                    print(f"[{now:%m-%d %H:%M}] 掃描失敗：{e}", flush=True)
                trim_memory()
            now = now_tw()
            time.sleep(max(30, 1200 - (now.minute % 20) * 60 - now.second))


_CLI = [a for a in sys.argv[1:] if a in ("--scan-once", "--wf-once", "--daemon")]
if __name__ == "__main__" and _CLI:
    cli_main(_CLI[0])
    sys.exit(0)


# ============================================================================
# C. 介面
# ============================================================================
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="台股量化操盤決策系統", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {background:#080c14 !important; color:#fff !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {background:#0b0f19 !important; border-right:1px solid rgba(79,172,254,.25) !important;}
p, span, label, h1, h2, h3, h4, h5, h6, li {color:#fff !important;}
a {color:#38bdf8 !important;}
div[data-baseweb="select"] > div {background:#0f172a !important; border:1px solid #334155 !important; border-radius:8px !important;}
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[data-baseweb="menu"] {background:#0b1120 !important; border:1px solid #38bdf8 !important;}
li[data-baseweb="menu-item"] {background:#0b1120 !important; color:#fff !important;}
li[data-baseweb="menu-item"]:hover {background:#1e293b !important; color:#38bdf8 !important;}
div.stButton > button {background:#080c14 !important; color:#fff !important; border:2px solid transparent !important; border-radius:9999px !important;
  padding:.6rem 2rem !important; font-weight:700 !important;
  background-image:linear-gradient(#080c14,#080c14),linear-gradient(90deg,#00f2fe,#4facfe,#fa709a,#fee140) !important;
  background-origin:border-box !important; background-clip:padding-box,border-box !important;
  box-shadow:0 0 16px rgba(79,172,254,.45) !important; transition:all .25s ease-in-out !important;}
div.stButton > button:hover {transform:translateY(-2px) !important; box-shadow:0 0 26px rgba(250,112,154,.6) !important;}
div[data-testid="stMetric"] {background:#0f172a !important; border:1px solid #1e293b !important; border-radius:12px !important; padding:12px 16px !important;}
div[data-testid="stMetricLabel"] {color:#94a3b8 !important;}
.stock-card {background:#0b1120; border:1px solid #1e293b; border-radius:16px; padding:16px; margin-bottom:14px;}
.stock-card-top1 {background:#0f172a; border:2px solid #38bdf8; border-radius:18px; padding:20px; margin-bottom:18px; box-shadow:0 0 25px rgba(56,189,248,.25);}
.tag-badge {display:inline-block; background:#082f49; color:#38bdf8 !important; border:1px solid #0284c7; padding:2px 8px; border-radius:9999px; font-size:11px; font-weight:600; margin:0 4px 4px 0;}
.rating-box {display:flex; justify-content:space-around; background:#060a12; border:1px solid #1e293b; border-radius:10px; padding:10px 4px; margin-top:12px; text-align:center;}
.rl {font-size:12px; color:#94a3b8 !important;} .rv {font-size:17px; font-weight:700;}
.bb-bg {width:100%; height:10px; background:#22c55e; border-radius:9999px; overflow:hidden; margin:8px 0;}
.bb-fill {height:100%; background:#ef4444;}
.chk-bull {background:#450a0a; border:1px solid #b91c1c; color:#fca5a5 !important; padding:3px 8px; border-radius:6px; font-size:12px; display:inline-block; margin:3px 4px 3px 0;}
.chk-bear {background:#052e16; border:1px solid #15803d; color:#86efac !important; padding:3px 8px; border-radius:6px; font-size:12px; display:inline-block; margin:3px 4px 3px 0;}
.lnk {font-size:12px; margin-right:10px;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 側邊欄
# ============================================================================
sb = st.sidebar
sb.markdown("### 資金與風控")
capital = sb.number_input("總操作資金 (TWD)", 50_000, 50_000_000, 1_000_000, 50_000)
risk_pct = sb.slider("單筆最大承受風險 (%)", 0.5, 5.0, 1.5, 0.1)
sb.markdown("### 股票池與過濾")
n_universe = sb.select_slider("網站內分析股票池（當日成交值前 N 檔；更大的池請用「排程全市場掃描」）", [100, 150, 200, 300, 400, 500, 600], value=300)
excl = sb.multiselect("排除產業", ["金融保險業", "航運業", "水泥工業", "食品工業", "貿易百貨業", "觀光餐旅", "塑膠工業", "造紙工業"],
                      default=["金融保險業", "航運業", "水泥工業"])
min_price = sb.number_input("最低股價", 1.0, 500.0, 35.0, 1.0)
min_turn_yi = sb.number_input("5日均成交值下限（億）", 0.1, 20.0, 1.5, 0.1)
min_amp = sb.number_input("20日振幅下限 (%)", 0.0, 30.0, 8.0, 0.5)
bias5_max, bias20_max = 6.0, 15.0   # 乖離上限改由 walk-forward 在「嚴/中/寬」三檔中挑選
sb.markdown("### 交易成本")
fee_disc = sb.slider("手續費折扣（0.6 = 6折）", 0.2, 1.0, 0.6, 0.02)
slip_pct = sb.slider("單邊滑價 (%)", 0.0, 0.5, 0.10, 0.05)
sb.markdown("### 即時性、主力與新聞")
use_intraday = sb.checkbox("排名使用盤中未收盤K線", value=False,
                           help="關閉=盤中只用已收盤資料排名（與回測一致，較穩定）；持股與自選股警示永遠使用最新價格。")
chip_weight = sb.slider("主力（外資+投信近5日）即時加減分權重", 0.0, 1.5, 1.0, 0.1,
                        help="近5日買賣超占成交值比，最多約 ±10 分。歷史逐日主力資料已另外進入回測。")
news_weight = sb.slider("新聞加減分權重（避雷為主）", 0.0, 1.5, 1.0, 0.1,
                        help="Google News 近7日標題的利多/利空關鍵字，-6~+3 分；無法回測，只作輔助。")
obj_mode = sb.selectbox("Walk-forward 最佳化目標", list(OBJ_MODES), format_func=lambda k: OBJ_MODES[k],
                        help="預設以『贏過大盤』為第一目標。選『勝率優先』會偏好勝率高、但須贏大盤且期望值為正的參數。")
sb.markdown("### Walk-forward")
train_days = sb.slider("訓練視窗（交易日）", 150, 400, 250, 10)
test_days = sb.slider("測試視窗（交易日）", 40, 120, 80, 10)
if sb.button("🔄 重新抓取最新資料"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for k in ("wf", "final_params"):
        st.session_state.pop(k, None)
    st.rerun()

# ============================================================================
# 記憶體保護、資料載入（快取有上限）＋ 交易時段每 20 分鐘自動更新
# ============================================================================
REFRESH_TICK_SEC = 300      # 每 5 分鐘檢查一次是否進入新的更新時段
MEM_LIMIT_MB = 850          # 超過就自動釋放快取（Streamlit 免費方案上限約 1GB）

_rss = rss_mb()
if _rss > MEM_LIMIT_MB:
    st.cache_data.clear()
    st.cache_resource.clear()
    trim_memory()
    st.toast(f"記憶體 {_rss:,.0f} MB 偏高，已自動釋放快取並重新載入。")
    _rss = rss_mb()
sb.caption(f"🧠 記憶體 {_rss:,.0f} MB（上限約 1,000 MB，超過 {MEM_LIMIT_MB} MB 會自動清快取）")


def bucket_key() -> str:
    """交易時段：每 20 分鐘一個時段（盤中更新）；收盤後/假日：每天一個（只建一次，記憶體不再成長）。"""
    n = now_tw()
    if in_session(n):
        return n.strftime("%Y-%m-%d-%H-") + str(n.minute // 20)
    return n.strftime("%Y-%m-%d-F")


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def load_meta(hour_key: str):
    return load_meta_core(hour_key)


@st.cache_data(ttl=6 * 3600, max_entries=2, show_spinner=False)
def prices_hist(tickers: tuple, day_key: str):
    return download_prices(list(tickers), period="4y")


@st.cache_data(ttl=1200, max_entries=2, show_spinner=False)
def prices_recent(tickers: tuple, bucket: str):
    return download_prices(list(tickers), period="7d", min_len=2)


def get_prices(tickers, bucket: str) -> dict:
    tk = tuple(tickers)
    return merge_recent(prices_hist(tk, bucket[:10]), prices_recent(tk, bucket))


@st.cache_data(ttl=6 * 3600, max_entries=2, show_spinner=False)
def inst_cached(codes: tuple, day_key: str):
    return load_inst_history(list(codes), token=get_secret("FINMIND_TOKEN"), max_fetch=400)


@st.cache_data(ttl=1200, max_entries=400, show_spinner=False)
def news_cached(code: str, name: str, day_key: str):
    return load_news(f"{name} {code}")


@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def remote_state(name: str):
    return fetch_state_json(name)


@st.cache_resource(ttl=3 * 3600, max_entries=1, show_spinner=False)
def build_engine(bucket, n, excl_t, min_price, min_turn_yi, min_amp, b5, b20, fee_disc, slip_pct, risk_pct, use_intraday):
    meta_ = load_meta(bucket.rsplit("-", 1)[0])
    out = build_engine_core(meta_, n, list(excl_t), min_price, min_turn_yi, min_amp, b5, b20, fee_disc, slip_pct,
                            risk_pct / 100.0, use_intraday, lambda tk: get_prices(tk, bucket),
                            inst_fn=lambda codes: inst_cached(tuple(codes), bucket[:10]))
    trim_memory()
    return out


bucket = bucket_key()
try:
    with st.spinner("更新資料中（首次約 1~3 分鐘；之後交易時段每 20 分鐘自動增量更新，收盤後每天只更新一次）…"):
        eng, bench_df, eng_notes = build_engine(bucket, n_universe, tuple(excl), min_price, min_turn_yi, min_amp,
                                                bias5_max, bias20_max, fee_disc, slip_pct, risk_pct, use_intraday)
        meta = load_meta(bucket.rsplit("-", 1)[0])
except Exception as e:
    st.error(f"資料載入失敗：{e}")
    st.stop()
st.session_state["loaded_bucket"] = bucket

NAMES, ALL_MARKET = build_name_market(meta)
get_px = lambda tk: get_prices(tk, bucket)
data_date = eng.idx[-1].strftime("%Y-%m-%d")
bench_last = float(eng.bench_close.iloc[-1])
regime_lvl = int(eng.regime_lvl[-1])
regime_ok = regime_lvl > 0
chip_fn = make_chip_fn(meta["chips"], chip_weight)
news_fn = make_news_fn(NAMES, lambda code, name: news_cached(code, name, bucket), news_weight)

wf = st.session_state.get("wf")
params = st.session_state.get("final_params") or load_best_params(fetch=remote_state) or Params()
portfolio_rows = st.session_state.setdefault("portfolio", load_portfolio())
watch_codes = st.session_state.setdefault("watchlist", load_watchlist())


def label(code: str) -> str:
    n = NAMES.get(code)
    return f"{n} ({code})" if n else f"{code}"


def chip_dict(code: str):
    ch = meta["chips"]
    if code in ch.index:
        r = ch.loc[code]
        return dict(foreign=float(r["foreign"]), trust=float(r["trust"]), total=float(r["total"]))
    return None


def link_html(code: str) -> str:
    return "".join(f'<a class="lnk" href="{u}" target="_blank">{n}</a>'
                   for n, u in links(code, ALL_MARKET.get(code, "TW")).items())


def cur_cfg() -> dict:
    base, ov = notify_cfg(), st.session_state.get("cfg_override", {})
    out = {k: (ov.get(k) or base.get(k, "")) for k in ("tg_token", "tg_chat", "line_token", "line_user")}
    out["line_broadcast"] = ov["line_broadcast"] if "line_broadcast" in ov else base["line_broadcast"]
    return out


with st.expander("📡 資料狀態（來源、日期、資料品質、抓取失敗訊息）"):
    st.write(f"價格資料最新日期：**{data_date}**　｜　股票池：**{len(eng.codes)}** 檔　｜　系統時間：{now_tw():%Y-%m-%d %H:%M} (台北)　｜　記憶體 {rss_mb():,.0f} MB")
    for m in eng_notes + meta["msgs"]:
        st.caption("• " + m)
    st.caption("• 即時籌碼（三大法人近5日）只接上市；『主力』因子的歷史逐日資料來自 FinMind（上市櫃皆有，會進入回測）。新聞與營收沒有歷史資料，不在回測內，只作輔助。")


@st.fragment(run_every=REFRESH_TICK_SEC)
def auto_tick():
    cur, loaded = bucket_key(), st.session_state.get("loaded_bucket")
    mode = "盤中・含未收盤K線" if use_intraday else "僅用已收盤K線"
    st.caption(f"⏱ 交易時段每 20 分鐘自動更新（不需按重新整理）｜資料時段 {loaded}｜台北時間 {now_tw():%H:%M:%S}｜{mode}")
    if loaded != cur:
        st.rerun()


auto_tick()

ctx = assemble_ctx(meta, ALL_MARKET, NAMES, eng, params, get_px, capital, risk_pct, chip_weight,
                   portfolio_rows, watch_codes, news_fn=news_fn)
alerts = build_alerts(eng, params, ctx)

cfg_now = cur_cfg()
configured = bool((cfg_now["tg_token"] and cfg_now["tg_chat"]) or cfg_now["line_token"])
if st.session_state.get("_auto_push", configured) and st.session_state.get("alerted_bucket") != bucket:
    msg, fresh = dispatch_alerts(alerts, cfg_now, st.session_state.get("_alert_cats", list(ALERT_CATS)))
    st.session_state["alerted_bucket"] = bucket
    st.session_state.setdefault("alert_log", []).insert(0, f"{now_tw():%m-%d %H:%M} ｜ 新訊號 {len(fresh)} 則 ｜ {msg}")
    st.session_state["alert_log"] = st.session_state["alert_log"][:30]
    if fresh:
        st.toast(f"已推送 {len(fresh)} 則新訊號")

PAGES = ["🎯 每日決策與推薦", "💼 持股健檢與換股", "🌐 族群多空儀表板", "🧪 Walk-Forward 驗證與風險",
         "🛠️ 15 項全景篩選器", "🔭 排程全市場掃描", "🔔 推播與自選股監控"]
page = st.radio("頁面", PAGES, horizontal=True, label_visibility="collapsed", key="page")   # 一次只執行一頁，省記憶體與運算


def candle_fig(df: pd.DataFrame, n: int, stop=None, tp=None, height=380, title=""):
    d = df.tail(n)
    cats = d.index.strftime("%m/%d").tolist()
    fig = go.Figure(go.Candlestick(x=cats, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                   increasing_line_color="#ef4444", decreasing_line_color="#22c55e", name="K"))
    fig.add_trace(go.Scatter(x=cats, y=d["MA20"], line=dict(color="#3b82f6", width=1.4), name="20MA"))
    fig.add_trace(go.Scatter(x=cats, y=d["MA60"], line=dict(color="#f59e0b", width=1.4), name="60MA"))
    if stop:
        fig.add_hline(y=stop, line_dash="dash", line_color="#22c55e", annotation_text=f"停損 {stop:.2f}")
    if tp:
        fig.add_hline(y=tp, line_dash="dash", line_color="#ef4444", annotation_text=f"目標 {tp:.2f}")
    fig.update_layout(height=height, title=title, xaxis=dict(type="category", rangeslider=dict(visible=False)),
                      margin=dict(l=5, r=5, t=40 if title else 10, b=5), paper_bgcolor="#080c14",
                      plot_bgcolor="#080c14", font=dict(color="#fff"), showlegend=False)
    return fig


def ind_df(code: str) -> pd.DataFrame:
    P = eng.P
    d = pd.DataFrame({f: P[f][code] for f in ("Open", "High", "Low", "Close", "MA20", "MA60")})
    return d.dropna(subset=["Close"])


def verdict_banner(level, head, bullets):
    fn = {"green": st.success, "yellow": st.warning, "red": st.error}[level]
    fn(head + "\n\n" + "\n".join("- " + b for b in bullets))


# ============================================================================
# Tab 1
# ============================================================================
def chip_line(code: str, det: dict) -> str:
    parts = []
    if "foreign" in det:
        parts.append(f"上市三大法人近5日：外資 {det['foreign']/1000:+,.0f} 張、投信 {det['trust']/1000:+,.0f} 張")
    else:
        parts.append("三大法人：無資料（上櫃或抓取失敗）")
    ry = meta["rev"].get(code, np.nan)
    parts.append(f"月營收年增 {ry:+.1f}%" if pd.notna(ry) else "月營收：無資料")
    return "　｜　".join(parts)


if page == PAGES[0]:
    st.markdown(f"### 盤勢與做多決策 ｜ 資料日期 {data_date}")
    breadth = float(eng.breadth.iloc[-1]) * 100
    (st.info if regime_ok else st.warning)(
        f"大盤環境：{LVL_TXT[regime_lvl]}｜加權指數 {bench_last:,.0f}｜股票池站上月線比例 {breadth:.0f}%"
        + ("" if regime_ok else "｜回測規則此時不新進場，以下僅供觀察"))

    if wf:
        _, bh = bench_return(eng.bench_close, wf["oos_equity"].index) if len(wf["oos_equity"]) else (None, np.nan)
        lv, hd, bl = verdict(wf, bh)
        verdict_banner(lv, hd, bl)
        st.caption(f"目前使用的參數（最近 {wf['train_days']} 日訓練視窗選出）：{params.label()}"
                   + ("" if wf["final_ok"] else "　⚠ 最近視窗交易數不足，改用預設參數"))
    else:
        st.warning("尚未執行 Walk-Forward 驗證，目前使用" + ("上次儲存的參數" if load_best_params() else "預設參數")
                   + "，**績效未經本次樣本外檢驗**。請到「🧪 Walk-Forward 驗證」分頁執行。")

    cands = latest_candidates(eng, params, top=10, exclude=meta["punish"], chip_fn=chip_fn, news_fn=news_fn)
    if not cands:
        st.warning("今日沒有標的同時通過流動性、趨勢、乖離與分數門檻。空手也是一種部位。")
    else:
        calib = eng_memo(eng, ("calib", params.preset, params.hold), lambda: score_calibration(eng, params.preset, params.hold))

        t1 = cands[0]
        code = t1["code"]
        tags = "".join(f'<span class="tag-badge">{html.escape(t)}</span>' for t in TAG_MAP.get(code, []))
        flag_txt = "　⚠ 注意股" if code in meta["notice"] else ""
        subs = t1["sub"]
        st.markdown("#### 👑 今日首選")
        st.markdown(f"""
        <div class="stock-card-top1">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <div><span style="font-size:30px;font-weight:900;color:#f59e0b !important;">{html.escape(label(code))}</span>{flag_txt}
              <div style="margin-top:8px;">{tags}</div><div style="margin-top:6px;">{link_html(code)}</div></div>
            <div style="text-align:right;">
              <div style="font-size:34px;font-weight:900;">{t1['close']:.2f}</div>
              <div style="font-size:17px;font-weight:700;color:{'#ef4444' if t1['pct']>=0 else '#22c55e'} !important;">{'▲' if t1['pct']>=0 else '▼'} {t1['pct']:+.2f}%</div>
              <div style="font-size:14px;color:#38bdf8 !important;font-weight:700;">最終分 {t1['final']:.1f}＝技術 {t1['score']:.1f} {t1['bonus']:+.1f} 主力 {t1['news_bonus']:+.1f} 新聞（{params.preset}）</div></div></div>
          <div class="rating-box">
            <div><div class="rl">趨勢</div><div class="rv">{subs['trend']:.0f}</div></div>
            <div><div class="rl">波動收斂</div><div class="rv">{subs['compress']:.0f}</div></div>
            <div><div class="rl">動能</div><div class="rv">{subs['mom']:.0f}</div></div>
            <div><div class="rl">相對強度</div><div class="rv">{subs['rs']:.0f}</div></div>
            <div><div class="rl">主力</div><div class="rv">{subs['main']:.0f}</div></div>
            <div><div class="rl">逼近新高</div><div class="rv">{subs['nh']:.0f}</div></div>
            <div><div class="rl">族群強度</div><div class="rv">{subs['sector']:.0f}</div></div>
            <div><div class="rl">進場型態</div><div class="rv">{subs['setup']:.0f}</div></div>
            <div><div class="rl">K / RSI</div><div class="rv">{t1['k']:.0f} / {t1['rsi']:.0f}</div></div></div></div>
        """, unsafe_allow_html=True)

        size = size_position(capital, risk_pct, t1["close"], t1["stop"])
        tp_show = None if np.isinf(t1["tp"]) else t1["tp"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("參考進場（隔日開盤附近）", f"{t1['close']:.2f}", f"成交 {int(t1['volume_lots']):,} 張")
        c2.metric("防守停損（已對齊升降單位）", f"{t1['stop']:.2f}", f"-{t1['risk_pct']:.2f}%", delta_color="inverse")
        c3.metric(f"停利目標 ({params.tp_r}R)" if tp_show else "停利方式",
                  f"{tp_show:.2f}" if tp_show else "移動停損", f"+{(tp_show/t1['close']-1)*100:.2f}%" if tp_show else f"獲利>1.5R 後追蹤{params.trail_atr}ATR")
        c4.metric("5日均成交值", f"{t1['turnover_yi']:.2f} 億")
        s1, s2, s3 = st.columns(3)
        s1.metric("建議張數", f"{size['lots']} 張 + {size['odd']} 股", f"依{size['binding']}")
        s2.metric("投入金額", f"{size['amount']:,.0f} 元", f"{size['pct_capital']:.1f}% 資金")
        s3.metric("停損時預估虧損", f"{size['risk_amt']:,.0f} 元", f"{size['risk_amt']/capital*100:.2f}% 資金", delta_color="inverse")
        st.caption(f"族群：{t1['group'] or '—'}　｜　" + chip_line(code, t1["chip"]))
        nw = t1.get("news") or {}
        if nw.get("items"):
            with st.expander(f"📰 近 7 日新聞（利多 {nw['pos']} / 利空 {nw['neg']}，關鍵字計分，僅供參考）"):
                for it in nw["items"]:
                    st.markdown(f"- [{it['title']}]({it['link']})")
        if len(calib):
            band = pd.cut([t1["score"]], [0, 55, 65, 75, 85, 101], right=False, labels=["<55", "55-65", "65-75", "75-85", "85+"])[0]
            if band in calib.index:
                r = calib.loc[band]
                st.caption(f"📊 歷史校準：過去股票池內技術分落在 {band} 且通過過濾的訊號共 {int(r['樣本數']):,} 次，"
                           f"隔日開盤買進持有 {params.hold} 日，勝率 {r['勝率']:.1f}%、平均報酬 {r['平均報酬']:+.2f}%（未扣成本，訊號高度重疊，僅供參考）。")
        st.plotly_chart(candle_fig(ind_df(code), 60, t1["stop"], tp_show, title=f"{label(code)} 日K（連續交易日）"),
                        use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 第 2～10 名")
        rows = []
        for i, c in enumerate(cands[1:], start=2):
            det = c["chip"]
            rows.append({"名次": i, "標的": label(c["code"]), "最終分": round(c["final"], 1), "技術分": round(c["score"], 1),
                         "主力加減": round(c["bonus"], 1), "新聞加減": round(c["news_bonus"], 1), "族群": c["group"], "收盤": c["close"], "漲跌%": round(c["pct"], 2), "停損": c["stop"],
                         "停利": "移動" if np.isinf(c["tp"]) else c["tp"], "風險%": round(c["risk_pct"], 2),
                         "外資5日(張)": round(det["foreign"] / 1000) if "foreign" in det else None,
                         "投信5日(張)": round(det["trust"] / 1000) if "trust" in det else None,
                         "注意股": "⚠" if c["code"] in meta["notice"] else ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        pick = st.selectbox("查看連結與K線", [c["code"] for c in cands[1:]], format_func=label) if len(cands) > 1 else None
        if pick:
            cc = next(c for c in cands if c["code"] == pick)
            st.markdown(link_html(pick), unsafe_allow_html=True)
            st.caption(chip_line(pick, cc["chip"]))
            st.plotly_chart(candle_fig(ind_df(pick), 60, cc["stop"], None if np.isinf(cc["tp"]) else cc["tp"], height=320),
                            use_container_width=True)
    st.caption("⚠ 以上為量化規則產出的輔助資訊，不是投資建議；實際下單前請自行確認消息面、處置/注意公告與流動性。")

# ============================================================================
# Tab 2：持股
# ============================================================================
if page == PAGES[1]:
    st.markdown("### 💼 持股健檢與換股（存於 portfolio.json）")
    base = pd.DataFrame(portfolio_rows) if portfolio_rows else pd.DataFrame(columns=["code", "cost", "shares", "date"])
    for col in ("code", "cost", "shares", "date"):
        if col not in base:
            base[col] = None
    base["code"] = base["code"].astype(str).str.strip()
    base.insert(1, "name", base["code"].map(lambda c: NAMES.get(c, "（查無名稱，請確認代碼）")))
    edited = st.data_editor(base[["code", "name", "cost", "shares", "date"]], num_rows="dynamic", use_container_width=True,
                            key="port_editor",
                            column_config={"code": "代碼", "name": st.column_config.TextColumn("名稱（自動帶入）", disabled=True),
                                           "cost": st.column_config.NumberColumn("成本", format="%.2f"),
                                           "shares": st.column_config.NumberColumn("股數", step=1000),
                                           "date": "買進日(YYYY-MM-DD)"})
    if st.button("💾 儲存持股"):
        rows_new = (edited[["code", "cost", "shares", "date"]].dropna(subset=["code"])
                    .replace({np.nan: None}).to_dict("records"))
        st.session_state["portfolio"] = rows_new
        save_portfolio(rows_new)
        st.success("已儲存。")

    recs = edited.replace({np.nan: None}).to_dict("records")
    codes_h = [str(r.get("code") or "").strip() for r in recs if str(r.get("code") or "").strip()]
    need = [c for c in codes_h if c not in ctx["ind_map"]]
    ind_h = {**ctx["ind_map"], **(get_ind_map(need, ALL_MARKET, get_px) if need else {})}
    hold_rows, health = [], {}
    for r in recs:
        c = str(r.get("code") or "").strip()
        if not c:
            continue
        d = ind_h.get(c)
        if d is None or len(d) < 65:
            hold_rows.append({"標的": label(c), "警訊": "無法取得足夠的價格資料"})
            continue
        h = holding_health(d, params)
        cost = float(r.get("cost") or 0)
        shares = float(r.get("shares") or 0)
        pnl = (h["close"] / cost - 1) * 100 if cost else np.nan
        try:
            days_held = (pd.Timestamp(d.index[-1]) - pd.Timestamp(str(r.get("date"))[:10])).days
        except Exception:
            days_held = None
        health[c] = dict(h, pnl=pnl, days=days_held)
        cd = chip_dict(c) or {}
        hold_rows.append({"標的": label(c), "現價": round(h["close"], 2), "成本": cost, "未實現%": round(pnl, 2),
                          "損益(元)": round((h["close"] - cost) * shares), "弱勢分": h["weak"],
                          "警訊": "、".join(h["flags"]) or "—", "防守價(現價重算)": h["stop"], "持有天數": days_held,
                          "外資5日(張)": round(cd["foreign"] / 1000) if "foreign" in cd else None,
                          "投信5日(張)": round(cd["trust"] / 1000) if "trust" in cd else None})
    if hold_rows:
        st.dataframe(pd.DataFrame(hold_rows), use_container_width=True, hide_index=True)
        st.caption("現價含盤中最新成交（若盤中）；弱勢分：跌破月線+2、月線下彎+1、跌破季線+2、RSI<45 +1、觸及防守價+3。")

    st.markdown("#### 換股建議")
    cands_p = latest_candidates(eng, params, top=3, exclude=meta["punish"], chip_fn=chip_fn)
    if not health:
        st.info("尚無持股。")
    elif not cands_p:
        st.info("今日無合格新標的，不需要換股。")
    else:
        top = cands_p[0]
        if top["code"] in health:
            st.success(f"今日首選 {label(top['code'])} 已在持股中。")
        else:
            wc, wh = max(health.items(), key=lambda kv: (kv[1]["weak"], -(kv[1]["pnl"] if pd.notna(kv[1]["pnl"]) else 0)))
            rt = eng.costs.round_trip * 100
            if wh["weak"] >= 2 and regime_ok:
                st.error(f"建議評估：賣出 **{label(wc)}**（弱勢分 {wh['weak']}：{'、'.join(wh['flags'])}），"
                         f"隔日開盤換入 **{label(top['code'])}**（最終分 {top['final']:.1f}）。換股來回成本約 {rt:.2f}%，"
                         "請確認新標的預期優勢大於成本。")
                if st.button(f"🔄 記錄換股：賣 {label(wc)} → 買 {label(top['code'])}"):
                    rows_p = st.session_state["portfolio"]
                    for r in rows_p:
                        if str(r.get("code") or "").strip() == wc:
                            r["code"], r["cost"], r["date"] = top["code"], round(top["close"], 2), data_date
                    save_portfolio(rows_p)
                    st.success("已更新 portfolio.json（實際成交價請自行修正）。")
                    st.rerun()
            else:
                st.success(f"持股沒有明顯轉弱訊號（最弱：{label(wc)}，弱勢分 {wh['weak']}），"
                           f"不建議為了換股支付約 {rt:.2f}% 的來回成本。" + ("" if regime_ok else " 大盤偏弱，也不新進場。"))

# ============================================================================
# Tab 3：族群
# ============================================================================
if page == PAGES[2]:
    st.markdown("### 🌐 族群多空儀表板（真實價量 + 官方籌碼/營收；沒資料就不顯示）")
    st.markdown("##### 📊 族群強弱排行（依產業別，股票池真實價量統計；越上面越強）")
    stab = eng_memo(eng, "sector_table", lambda: sector_table(eng, NAMES))
    if len(stab):
        st.dataframe(stab, use_container_width=True, hide_index=True)
        st.caption("強度＝20日報酬、60日報酬、站上月線比例的綜合排名；此排名也已納入評分的『族群強度』特徵（可回測）。")
    st.markdown("##### 🧩 題材卡片")
    c_a, c_b = st.columns(2)
    sec = c_a.selectbox("題材族群", list(SECTORS))
    all_tags = sorted({t for v in SECTORS.values() for _, _, ts in v for t in ts})
    tag_pick = c_b.selectbox("概念標籤（選了會覆蓋族群）", ["—"] + all_tags)
    items = ([(c, n, t) for v in SECTORS.values() for c, n, t in v if tag_pick in t] if tag_pick != "—" else SECTORS[sec])
    seen = set()
    items = [i for i in items if not (i[0] in seen or seen.add(i[0]))]
    sec_ind = get_ind_map([c for c, _, _ in items], ALL_MARKET, get_px)
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for k in range(2):
            if i + k >= len(items):
                continue
            c, n0, tg = items[i + k]
            with cols[k]:
                d = sec_ind.get(c)
                if d is None or len(d) < 65:
                    st.warning(f"{label(c)}：無足夠價格資料")
                    continue
                last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
                diff = last - prev
                col = "#ef4444" if diff >= 0 else "#22c55e"
                diag = bull_bear_checks(d, chip_dict(c), meta["rev"].get(c, np.nan))
                nb, nr = len(diag["bull"]), max(1, len(diag["bear"]))
                pct_b = int(nb / (nb + nr) * 100)
                mk = "上櫃" if ALL_MARKET.get(c) == "TWO" else "上市"
                st.markdown(f"""
                <div class="stock-card">
                  <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <div><span style="font-size:22px;font-weight:800;color:#f59e0b !important;">{c} {html.escape(NAMES.get(c, n0))}</span>
                      <span style="font-size:12px;background:#334155;padding:2px 6px;border-radius:4px;margin-left:6px;">{mk}</span>
                      <div style="margin-top:6px;">{''.join(f'<span class="tag-badge">{html.escape(t)}</span>' for t in tg)}</div></div>
                    <div style="text-align:right;"><div style="font-size:26px;font-weight:800;">{last:.2f}</div>
                      <div style="color:{col} !important;font-weight:600;">{'▲' if diff>=0 else '▼'} {diff:+.2f} ({diff/prev*100:+.2f}%)</div></div></div>
                  <div style="margin-top:8px;">{link_html(c)}</div></div>""", unsafe_allow_html=True)
                st.plotly_chart(candle_fig(d, 35, height=210), use_container_width=True)
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;">
                  <span style="color:#ef4444 !important;">● 多方 {nb} 項</span><span style="color:#22c55e !important;">● 空方 {len(diag['bear'])} 項</span></div>
                <div class="bb-bg"><div class="bb-fill" style="width:{pct_b}%;"></div></div>
                <div style="font-size:12px;"><b style="color:#f87171 !important;">多方：</b>{''.join(f'<span class="chk-bull">{html.escape(x)}</span>' for x in diag['bull'][:8])}</div>
                <div style="font-size:12px;"><b style="color:#4ade80 !important;">空方：</b>{''.join(f'<span class="chk-bear">{html.escape(x)}</span>' for x in diag['bear'][:6]) or '—'}</div>
                """, unsafe_allow_html=True)

# ============================================================================
# Tab 4：驗證
# ============================================================================
if page == PAGES[3]:
    st.markdown("### 🧪 Walk-Forward 驗證與風險")
    st.caption("做法：滾動視窗——用訓練期挑參數（分數權重預設、進場門檻、ATR停損倍數、停利R、持股天數），"
               "再拿『沒看過的』測試期驗證，最後把所有測試期串起來看樣本外績效。回測含手續費、證交稅、滑價；"
               "隔日開盤進場；跳空以開盤價成交；同日停損優先於停利。")
    st.caption(f"候選參數組合共 {len(list(__import__('itertools').product(*DEFAULT_GRID.values())))} 組；"
               f"資料共 {len(eng.idx)} 個交易日（{eng.idx[0]:%Y-%m-%d} ~ {eng.idx[-1]:%Y-%m-%d}）。")
    if st.button("▶ 執行 Walk-Forward 驗證"):
        bar = st.progress(0.0, text="回測中…")
        res = walk_forward(eng, train_days=train_days, test_days=test_days, progress=lambda x: bar.progress(x), obj_mode=obj_mode)
        bar.empty()
        if res is None:
            st.error("資料天數不足以切出訓練+測試視窗，請縮短視窗長度或擴大股票池。")
        else:
            st.session_state["wf"] = res
            st.session_state["final_params"] = res["final_params"]
            _bh = bench_return(eng.bench_close, res["oos_equity"].index)[1] if len(res["oos_equity"]) else np.nan
            save_best_params(res["final_params"], verdict(res, _bh)[0])
            st.rerun()

    wf = st.session_state.get("wf")
    if wf:
        o = wf["oos_stats"]
        curve, bh = bench_return(eng.bench_close, wf["oos_equity"].index) if len(wf["oos_equity"]) else (pd.Series(dtype=float), np.nan)
        lv, hd, bl = verdict(wf, bh)
        verdict_banner(lv, hd, bl)

        def f(x, fmt="{:.2f}", na="—"):
            return na if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)

        m = st.columns(6)
        m[0].metric("樣本外交易筆數", f"{o['n']}", "n<30 不可靠" if o["n"] < 30 else "")
        m[1].metric("勝率", f(o["win_rate"], "{:.1f}%"), f"95%CI {f(o['wr_lo'], '{:.0f}')}~{f(o['wr_hi'], '{:.0f}')}%")
        m[2].metric("平均獲利 / 平均虧損", f"{f(o['avg_win'], '{:+.2f}')}% / {f(o['avg_loss'], '{:+.2f}')}%")
        m[3].metric("每筆期望值", f(o["expectancy"], "{:+.2f}%"), f"t={f(o['tstat'])}")
        m[4].metric("盈虧因子 / 賺賠比", f"{f(o['profit_factor'])} / {f(o['payoff'])}")
        m[5].metric("最大連虧", f"{o['max_consec_loss']} 次")
        m2 = st.columns(5)
        m2[0].metric("樣本外總報酬", f(o["total_ret"], "{:+.1f}%"), f"大盤 {f(bh, '{:+.1f}')}%")
        m2[1].metric("年化報酬", f(o["cagr"], "{:+.1f}%") if not np.isnan(o["cagr"]) else "期間過短")
        m2[2].metric("最大回撤", f(o["mdd"], "-{:.1f}%"))
        m2[3].metric("Sharpe", f(o["sharpe"]))
        m2[4].metric("平均持有", f(o["avg_hold"], "{:.1f} 日"))

        if len(wf["oos_equity"]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wf["oos_equity"].index, y=wf["oos_equity"].values, name="策略（樣本外串接）", line=dict(color="#f59e0b", width=2.2)))
            if len(curve):
                fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name="加權指數買進持有", line=dict(color="#38bdf8", width=1.6)))
            fig.update_layout(height=340, title="樣本外淨值（起點=1）", paper_bgcolor="#080c14", plot_bgcolor="#080c14", font=dict(color="#fff"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 🩺 失敗原因診斷（依本次樣本外交易自動分析）")
        dg_items, dg_exit = diagnose(wf, eng)
        for lv_, tx_ in dg_items:
            {"red": st.error, "yellow": st.warning}.get(lv_, st.info)(tx_)
        if len(dg_exit):
            st.dataframe(dg_exit, use_container_width=True)
        st.markdown("##### 各折明細（訓練期 vs 測試期）")
        fd = wf["folds"].copy()
        fd = fd.rename(columns={"fold": "折", "train": "訓練期", "test": "測試期", "params": "訓練期選出的參數", "train_n": "訓練筆數",
                                "train_exp": "訓練期望值%", "test_n": "測試筆數", "test_exp": "測試期望值%",
                                "test_ret": "測試報酬%", "bench_ret": "同期大盤%", "test_mdd": "測試MDD%", "rank_pct": "測試期排名百分位"})
        st.dataframe(fd.round(2), use_container_width=True, hide_index=True)
        st.caption(f"訓練期平均期望值 {f(wf['is_expectancy'], '{:+.2f}%')} → 樣本外 {f(o['expectancy'], '{:+.2f}%')}；"
                   f"所選參數在測試期的排名中位數 {f(wf['rank_pct_median'], '{:.0f}')} 百分位（≈50 代表挑參數沒有增值）。"
                   f"各折選到的權重預設：{wf['preset_selection']}（越分散代表參數越不穩定）。")

        st.markdown("##### 🎲 蒙地卡羅（區塊自助法）— 依樣本外交易損益")
        trs = wf["oos_trades"]["net_pct"].tolist() if len(wf["oos_trades"]) else []
        mc = mc_ruin(trs)
        if not mc:
            st.info("樣本外交易少於 5 筆，無法做蒙地卡羅。")
        else:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("60 筆後平均淨值", f"{mc['end_mean']:.2f}")
            a2.metric("5% 最差淨值", f"{mc['end_p5']:.2f}")
            a3.metric("95% 信賴最大回撤", f"-{mc['mdd95']:.1f}%")
            a4.metric("淨值腰斬機率", f"{mc['p_ruin']:.2f}%")
            fig2 = go.Figure()
            for row in mc["curves"][:60]:
                fig2.add_trace(go.Scatter(y=row, mode="lines", line=dict(width=.6, color="rgba(56,189,248,.18)"), showlegend=False))
            fig2.add_trace(go.Scatter(y=mc["curves"].mean(axis=0), line=dict(width=2.4, color="#f59e0b"), name="平均"))
            fig2.update_layout(height=320, paper_bgcolor="#080c14", plot_bgcolor="#080c14", font=dict(color="#fff"),
                               xaxis_title="交易筆數", yaxis_title="淨值（起點1，每筆約25%資金）")
            st.plotly_chart(fig2, use_container_width=True)
            if len(trs) < 30:
                st.caption("⚠ 樣本少於 30 筆，蒙地卡羅只是在重複抽你手上很少的資料，低估真實風險的可能性很高。")

        st.markdown("##### 🔬 訊號預測力檢驗（整個股票池、不限於實際下單的標的）")
        hz = st.select_slider("預測期間（交易日）", [5, 10, 15, 20], value=10)
        sq = eng_memo(eng, ("sq", params.preset, hz), lambda: signal_quality(eng, params.preset, hz))
        q1, q2, q3 = st.columns(3)
        q1.metric("Rank IC 平均", f(sq["ic_mean"], "{:+.3f}"), "分數 vs 未來報酬的排序相關")
        q2.metric("IC t 值", f(sq["ic_t"]), "|t|>2 才算顯著")
        q3.metric("IC IR", f(sq["ic_ir"]))
        if len(sq["quintile"]):
            st.dataframe(sq["quintile"].round(2), use_container_width=True)
            qq = sq["quintile"]
            top, bot = qq["平均報酬"].iloc[-1], qq["平均報酬"].iloc[0]
            mono = qq["平均報酬"].is_monotonic_increasing
            st.caption(("高分組報酬高於低分組（{:+.2f}% vs {:+.2f}%），".format(top, bot) if top > bot else
                        "高分組報酬並未高於低分組（{:+.2f}% vs {:+.2f}%），".format(top, bot))
                       + ("且五分位單調遞增。" if mono else "五分位並非單調，分數的排序能力有限。")
                       + " 若 IC 接近 0，代表這套分數對『預測』沒有實質幫助，別被漂亮的回測曲線騙了。")
        if len(wf["oos_trades"]):
            with st.expander("樣本外逐筆交易明細"):
                t = wf["oos_trades"].copy()
                t["code"] = t["code"].map(label)
                t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.strftime("%Y-%m-%d")
                t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(t.rename(columns={"code": "標的", "entry_date": "進場日(開盤)", "exit_date": "出場日", "fill": "成交價",
                                               "exit": "出場價", "reason": "出場原因", "days": "持有日", "net_pct": "淨損益%",
                                               "r_mult": "R倍數", "fold": "折"}).round(2), use_container_width=True, hide_index=True)
    else:
        st.info("按上方按鈕開始驗證。第一次執行約需數十秒到數分鐘（視股票池大小）。")

# ============================================================================
# Tab 5：篩選器
# ============================================================================
if page == PAGES[4]:
    st.markdown("### 🛠️ 15 項全景篩選器（對整個流動性股票池，不受推薦過濾影響）")
    snap = eng_memo(eng, ("snap", params.preset), lambda: snapshot(eng, params.preset))
    snap = snap[snap["liquid"]]
    chosen = []
    with st.expander("勾選條件（全部 AND）", expanded=True):
        cols = st.columns(3)
        for i, name in enumerate(SCREEN_FLAGS):
            if cols[i % 3].checkbox(name, key=f"scr_{i}"):
                chosen.append(name)
        s1, s2 = st.columns(2)
        mb5 = s1.slider("5MA 乖離上限 (%)", 1.0, 15.0, 15.0, 0.5, key="scr_b5")
        mb20 = s2.slider("20MA 乖離上限 (%)", 3.0, 40.0, 40.0, 1.0, key="scr_b20")
    res = snap.copy()
    for name in chosen:
        res = res[res[name]]
    res = res[(res["bias5"] <= mb5) & (res["bias20"] <= mb20)].sort_values("score", ascending=False)
    if res.empty:
        st.warning("沒有符合的標的，請放寬條件。")
    else:
        show = pd.DataFrame({"標的": [label(c) for c in res.index], "分數": res["score"].round(1), "收盤": res["close"].round(2),
                             "漲跌%": res["pct"].round(2), "成交(張)": res["volume_lots"].round(0).astype(int),
                             "5日均額(億)": res["turnover_yi"].round(2), "5MA乖離%": res["bias5"].round(2),
                             "20MA乖離%": res["bias20"].round(2), "K": res["K"].round(1), "RSI": res["RSI"].round(1),
                             "通過推薦過濾": np.where(res["eligible"], "✔", "")})
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(f"符合 {len(res)} 檔 / 流動性股票池 {len(snap)} 檔。")

# ============================================================================
# 排程全市場掃描（由 GitHub Actions 計算；本頁只讀結果，不佔網站記憶體）
# ============================================================================
@st.cache_data(ttl=300, max_entries=1, show_spinner=False)
def load_latest_scan():
    return fetch_state_json("latest_scan.json")


if page == PAGES[5]:
    st.markdown("### 🔭 排程全市場掃描（更大的股票池，盤中每 20 分鐘由 GitHub Actions 計算並推播）")
    scan = load_latest_scan()
    if not scan:
        st.info("尚未取得掃描結果。請依說明設定 GitHub Actions 排程，並在 Streamlit Secrets 設定 STATE_BASE_URL"
                "（指向 state 分支的 raw 網址）。這個頁面只讀取結果，所以不會增加網站記憶體。")
    else:
        lvl_ = int(scan.get("regime", 1))
        (st.info if lvl_ else st.warning)(f"掃描時間 {scan['updated']}｜資料日 {scan['data_date']}｜股票池 {scan['universe']} 檔｜"
                                          f"通過條件 {scan['eligible']} 檔｜大盤環境 {LVL_TXT[lvl_]}")
        st.caption("使用參數：" + scan.get("params", ""))
        if scan.get("candidates"):
            df_s = pd.DataFrame(scan["candidates"]).rename(columns={
                "code": "代碼", "name": "名稱", "group": "族群", "final": "最終分", "tech": "技術分", "main": "主力加減",
                "close": "收盤", "pct": "漲跌%", "stop": "停損", "tp": "停利", "risk": "風險%", "turnover": "5日均額(億)"})
            st.dataframe(df_s, use_container_width=True, hide_index=True)
        else:
            st.warning("本次掃描沒有標的通過條件。")
        for m_ in scan.get("notes", []):
            st.caption("• " + m_)
# ============================================================================
# Tab 6：推播與自選股監控
# ============================================================================
if page == PAGES[6]:
    st.markdown("### 🔔 推播與自選股監控（Telegram / LINE）")
    st.caption("頁面開著時每 20 分鐘自動掃描；要『瀏覽器關閉也推播』請在主機另外執行 `python tw_quant_app.py --daemon`。"
               "同一則訊號不會重複推送（記錄在 alert_state.json）。")
    with st.expander("推播管道設定", expanded=not configured):
        st.markdown("""
**Telegram**：在 @BotFather 建立機器人取得 Token；對機器人傳一則訊息後，開啟
`https://api.telegram.org/bot<TOKEN>/getUpdates` 找到 `chat.id`。
**LINE**：LINE Notify 已於 2025/3/31 終止，請改用 LINE Official Account 的 Messaging API：取得 Channel access token；
填入自己的 User ID 會單獨推送給你，留空則會廣播給所有好友（免費方案每月訊息量有限）。
建議把金鑰放在環境變數或 `.streamlit/secrets.toml`（鍵名：TELEGRAM_BOT_TOKEN、TELEGRAM_CHAT_ID、LINE_CHANNEL_TOKEN、LINE_USER_ID），不要寫死在程式碼。
        """)
        ov = st.session_state.setdefault("cfg_override", {})
        ov["tg_token"] = st.text_input("Telegram Bot Token（僅存於本次連線）", value=ov.get("tg_token", ""), type="password")
        ov["tg_chat"] = st.text_input("Telegram Chat ID", value=ov.get("tg_chat", ""))
        ov["line_token"] = st.text_input("LINE Channel access token", value=ov.get("line_token", ""), type="password")
        ov["line_user"] = st.text_input("LINE User ID（你自己的；持股警示只會傳給這個 ID）", value=ov.get("line_user", ""))
        ov["line_broadcast"] = st.checkbox("LINE 廣播公開訊號給所有加好友的人（持股警示仍只傳給你）",
                                           value=bool(ov.get("line_broadcast", notify_cfg()["line_broadcast"])))
        st.caption("廣播只會送到『已加入該官方帳號為好友』的人；好友人數可在 LINE Official Account Manager 查看。"
                   "也可用環境變數/Secrets 設定 LINE_BROADCAST=1。")
    cfg_now = cur_cfg()
    configured = bool((cfg_now["tg_token"] and cfg_now["tg_chat"]) or cfg_now["line_token"])
    a1, a2 = st.columns(2)
    st.session_state["_auto_push"] = a1.checkbox("開啟自動推播（交易時段每 20 分鐘掃描一次）",
                                                 value=st.session_state.get("_auto_push", configured))
    st.session_state["_alert_cats"] = a2.multiselect("推播哪些類型", list(ALERT_CATS),
                                                     default=st.session_state.get("_alert_cats", list(ALERT_CATS)),
                                                     format_func=lambda k: ALERT_CATS[k])
    if st.button("📨 傳送測試訊息"):
        res = notify_all("✅ 台股量化系統推播測試成功。", cfg_now) if configured else []
        if not res:
            st.error("尚未設定任何推播管道。")
        for ok, m in res:
            (st.success if ok else st.error)(m)

    st.markdown("#### 自選股監控（任何上市櫃代碼；突破、跌破、爆量、KD/MACD 轉折都會通知）")
    wl_text = st.text_area("代碼（以逗號、空白或換行分隔）", value=", ".join(st.session_state["watchlist"]), height=70)
    if st.button("💾 儲存自選股"):
        codes = [c for c in re.split(r"[\s,，、]+", wl_text) if re.fullmatch(r"\d{4}", c)]
        st.session_state["watchlist"] = codes
        save_watchlist(codes)
        st.success(f"已儲存 {len(codes)} 檔：" + "、".join(label(c) for c in codes))
        st.rerun()
    if st.session_state["watchlist"]:
        st.caption("目前監控：" + "、".join(label(c) for c in st.session_state["watchlist"]))

    st.markdown("#### 目前偵測到的訊號（預覽；是否已推送看下方紀錄）")
    if alerts:
        st.dataframe(pd.DataFrame([{"類型": ALERT_CATS[a.cat], "內容": a.text.replace("\n", "  ")} for a in alerts]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("目前沒有新的訊號。")
    if st.session_state.get("alert_log"):
        st.markdown("#### 推播紀錄")
        for line in st.session_state["alert_log"][:15]:
            st.caption(line)

