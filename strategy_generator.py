#!/usr/bin/env python3
"""
strategy_generator.py - Systematic Strategy Generation
=======================================================
Instead of relying on LLM creativity (which keeps recycling same indicators),
systematically combine ALL available indicators in a structured way.

Generates strategy.py files from a template with different indicator combos.
"""
import itertools
import random
from pathlib import Path

# ALL available indicators with their code templates
TREND_INDICATORS = {
    "ema_cross": {
        "code": """
    ema_fast = close_s.ewm(span={fast}, min_periods={fast}, adjust=False).mean().values
    ema_slow = close_s.ewm(span={slow}, min_periods={slow}, adjust=False).mean().values
    trend = np.where(ema_fast > ema_slow, 1.0, -1.0)""",
        "params": {"fast": [8, 12, 21], "slow": [21, 34, 55]},
    },
    "sma_cross": {
        "code": """
    sma_fast = close_s.rolling({fast}, min_periods={fast}).mean().values
    sma_slow = close_s.rolling({slow}, min_periods={slow}).mean().values
    trend = np.where(sma_fast > sma_slow, 1.0, -1.0)""",
        "params": {"fast": [10, 20, 50], "slow": [50, 100, 200]},
    },
    "hma_slope": {
        "code": """
    def _wma(a, w):
        wts = np.arange(1, w+1, dtype=np.float64); wts /= wts.sum()
        out = np.full(len(a), np.nan)
        for j in range(w-1, len(a)): out[j] = np.dot(a[j-w+1:j+1], wts)
        return out
    hma = _wma(2*_wma(close, {period}//2) - _wma(close, {period}), max(1,int(np.sqrt({period}))))
    trend = np.zeros(n)
    for i in range(2, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-1]):
            trend[i] = 1.0 if hma[i] > hma[i-1] else -1.0""",
        "params": {"period": [16, 21, 34]},
    },
    "supertrend": {
        "code": """
    tr = np.zeros(n)
    for i in range(1, n): tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr_st = pd.Series(tr).rolling({atr_period}, min_periods={atr_period}).mean().values
    hl2 = (high + low) / 2; upper = hl2 + {mult}*atr_st; lower = hl2 - {mult}*atr_st
    trend = np.zeros(n); fu = np.full(n, np.nan); fl = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(upper[i]): trend[i]=trend[i-1]; continue
        fl[i] = max(lower[i], fl[i-1]) if not np.isnan(fl[i-1]) and close[i-1]>fl[i-1] else lower[i]
        fu[i] = min(upper[i], fu[i-1]) if not np.isnan(fu[i-1]) and close[i-1]<fu[i-1] else upper[i]
        if close[i]>fu[i]: trend[i]=1
        elif close[i]<fl[i]: trend[i]=-1
        else: trend[i]=trend[i-1]""",
        "params": {"atr_period": [10, 14], "mult": [2.0, 3.0]},
    },
    "kama_direction": {
        "code": """
    kama = np.zeros(n); kama[10] = close[10]
    for i in range(11, n):
        direction_k = abs(close[i] - close[i-10])
        volatility_k = sum(abs(close[j]-close[j-1]) for j in range(i-9, i+1))
        er = direction_k / volatility_k if volatility_k > 0 else 0
        sc = (er * (2/3 - 2/31) + 2/31) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    trend = np.zeros(n)
    for i in range(12, n): trend[i] = 1.0 if close[i] > kama[i] and kama[i] > kama[i-1] else (-1.0 if close[i] < kama[i] and kama[i] < kama[i-1] else 0.0)""",
        "params": {},
    },
    "donchian": {
        "code": """
    don_h = pd.Series(high).rolling({period}, min_periods={period}).max().values
    don_l = pd.Series(low).rolling({period}, min_periods={period}).min().values
    trend = np.zeros(n)
    for i in range({period}+1, n):
        if close[i] > don_h[i-1]: trend[i] = 1.0
        elif close[i] < don_l[i-1]: trend[i] = -1.0
        else: trend[i] = trend[i-1]""",
        "params": {"period": [15, 20, 30]},
    },
    "ichimoku": {
        "code": """
    tenkan = (pd.Series(high).rolling(20).max().values + pd.Series(low).rolling(20).min().values) / 2
    kijun = (pd.Series(high).rolling(60).max().values + pd.Series(low).rolling(60).min().values) / 2
    trend = np.zeros(n)
    for i in range(60, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            trend[i] = 1.0 if tenkan[i] > kijun[i] and close[i] > kijun[i] else (-1.0 if tenkan[i] < kijun[i] and close[i] < kijun[i] else 0.0)""",
        "params": {},
    },
    "parabolic_sar": {
        "code": """
    sar = np.zeros(n); sar[0] = low[0]; af = 0.02; ep = high[0]; is_long = True
    trend = np.zeros(n)
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if is_long:
            if low[i] < sar[i]: is_long = False; sar[i] = ep; ep = low[i]; af = 0.02
            else:
                if high[i] > ep: ep = high[i]; af = min(af + 0.02, 0.20)
        else:
            if high[i] > sar[i]: is_long = True; sar[i] = ep; ep = high[i]; af = 0.02
            else:
                if low[i] < ep: ep = low[i]; af = min(af + 0.02, 0.20)
        trend[i] = 1.0 if is_long else -1.0""",
        "params": {},
    },
}

ENTRY_FILTERS = {
    "rsi": {
        "code": """
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta>0, delta, 0.0); loss = np.where(delta<0, -delta, 0.0)
    ag = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    al = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(al>0, ag/al, 100.0); rsi = 100 - 100/(1+rs)
    entry_ok_long = rsi < {long_max}
    entry_ok_short = rsi > {short_min}""",
        "params": {"long_max": [55, 60, 65], "short_min": [35, 40, 45]},
    },
    "stochastic": {
        "code": """
    low_min = pd.Series(low).rolling(14, min_periods=14).min().values
    high_max = pd.Series(high).rolling(14, min_periods=14).max().values
    stoch_k = np.where(high_max-low_min > 0, (close-low_min)/(high_max-low_min)*100, 50)
    stoch_d = pd.Series(stoch_k).rolling(3, min_periods=3).mean().values
    entry_ok_long = stoch_k < {long_max}
    entry_ok_short = stoch_k > {short_min}""",
        "params": {"long_max": [30, 40], "short_min": [60, 70]},
    },
    "williams_r": {
        "code": """
    high_max = pd.Series(high).rolling(14, min_periods=14).max().values
    low_min = pd.Series(low).rolling(14, min_periods=14).min().values
    willr = np.where(high_max-low_min > 0, (high_max-close)/(high_max-low_min)*(-100), -50)
    entry_ok_long = willr < {long_max}
    entry_ok_short = willr > {short_min}""",
        "params": {"long_max": [-70, -80], "short_min": [-20, -30]},
    },
    "cci": {
        "code": """
    tp = (high + low + close) / 3
    tp_sma = pd.Series(tp).rolling(20, min_periods=20).mean().values
    tp_mad = pd.Series(tp).rolling(20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean()))).values
    cci = np.where(tp_mad > 0, (tp - tp_sma) / (0.015 * tp_mad), 0)
    entry_ok_long = cci < {long_max}
    entry_ok_short = cci > {short_min}""",
        "params": {"long_max": [-100, -50], "short_min": [50, 100]},
    },
    "mfi": {
        "code": """
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = np.where(np.diff(tp, prepend=tp[0]) > 0, mf, 0)
    neg_mf = np.where(np.diff(tp, prepend=tp[0]) < 0, mf, 0)
    pos_sum = pd.Series(pos_mf).rolling(14, min_periods=14).sum().values
    neg_sum = pd.Series(neg_mf).rolling(14, min_periods=14).sum().values
    mfi = np.where(neg_sum > 0, 100 - 100/(1 + pos_sum/neg_sum), 50)
    entry_ok_long = mfi < {long_max}
    entry_ok_short = mfi > {short_min}""",
        "params": {"long_max": [30, 40], "short_min": [60, 70]},
    },
    "obv_trend": {
        "code": """
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]: obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]: obv[i] = obv[i-1] - volume[i]
        else: obv[i] = obv[i-1]
    obv_ema = pd.Series(obv).ewm(span=21, min_periods=21, adjust=False).mean().values
    entry_ok_long = np.array([obv[i] > obv_ema[i] if not np.isnan(obv_ema[i]) else False for i in range(n)])
    entry_ok_short = np.array([obv[i] < obv_ema[i] if not np.isnan(obv_ema[i]) else False for i in range(n)])""",
        "params": {},
    },
    "macd_hist": {
        "code": """
    macd_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    macd_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    macd_line = macd_fast - macd_slow
    macd_signal = pd.Series(macd_line).ewm(span=9, min_periods=9, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    entry_ok_long = np.array([macd_hist[i] > 0 for i in range(n)])
    entry_ok_short = np.array([macd_hist[i] < 0 for i in range(n)])""",
        "params": {},
    },
    "volume_spike": {
        "code": """
    vol_avg = pd.Series(volume).rolling(20, min_periods=20).mean().values
    vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 1.0)
    entry_ok_long = np.array([vol_ratio[i] > {threshold} for i in range(n)])
    entry_ok_short = entry_ok_long.copy()""",
        "params": {"threshold": [1.2, 1.5, 2.0]},
    },
}

REGIME_FILTERS = {
    "none": {"code": "    regime_ok = np.ones(n, dtype=bool)", "params": {}},
    "choppiness": {
        "code": """
    _tr = np.zeros(n)
    for i in range(1, n): _tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _atr_sum = pd.Series(_tr).rolling(14, min_periods=14).sum().values
    _hh = pd.Series(high).rolling(14, min_periods=14).max().values
    _ll = pd.Series(low).rolling(14, min_periods=14).min().values
    _range = _hh - _ll
    chop = np.where(_range > 0, 100 * np.log10(_atr_sum / _range) / np.log10(14), 50)
    regime_ok = np.array([chop[i] < {trending_thresh} for i in range(n)])""",
        "params": {"trending_thresh": [45, 50, 55]},
    },
    "bbw_regime": {
        "code": """
    _sma = close_s.rolling(20, min_periods=20).mean().values
    _std = close_s.rolling(20, min_periods=20).std().values
    _bbw = np.where(_sma > 0, _std / _sma, 0)
    _bbw_pct = pd.Series(_bbw).rolling(100, min_periods=50).rank(pct=True).values
    regime_ok = np.array([not np.isnan(_bbw_pct[i]) and _bbw_pct[i] < {max_pct} and _bbw_pct[i] > {min_pct} for i in range(n)])""",
        "params": {"max_pct": [0.7, 0.8], "min_pct": [0.1, 0.2]},
    },
    "adx_filter": {
        "code": """
    _pdm = np.zeros(n); _ndm = np.zeros(n)
    for i in range(1, n):
        hd = high[i]-high[i-1]; ld = low[i-1]-low[i]
        if hd > ld and hd > 0: _pdm[i] = hd
        if ld > hd and ld > 0: _ndm[i] = ld
    _tr2 = np.zeros(n)
    for i in range(1, n): _tr2[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _atr2 = pd.Series(_tr2).ewm(span=14, min_periods=14, adjust=False).mean().values
    _pdi = np.where(_atr2>0, 100*pd.Series(_pdm).ewm(span=14,min_periods=14,adjust=False).mean().values/_atr2, 0)
    _ndi = np.where(_atr2>0, 100*pd.Series(_ndm).ewm(span=14,min_periods=14,adjust=False).mean().values/_atr2, 0)
    _dx = np.where(_pdi+_ndi>0, 100*np.abs(_pdi-_ndi)/(_pdi+_ndi), 0)
    adx = pd.Series(_dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    regime_ok = np.array([adx[i] > {min_adx} for i in range(n)])""",
        "params": {"min_adx": [20, 25, 30]},
    },
    "aroon_filter": {
        "code": """
    aroon_up = np.zeros(n); aroon_dn = np.zeros(n)
    for i in range(25, n):
        hh_idx = i - 25 + np.argmax(high[i-25:i])
        ll_idx = i - 25 + np.argmin(low[i-25:i])
        aroon_up[i] = (25 - (i - hh_idx)) / 25 * 100
        aroon_dn[i] = (25 - (i - ll_idx)) / 25 * 100
    regime_ok = np.array([abs(aroon_up[i] - aroon_dn[i]) > {threshold} for i in range(n)])""",
        "params": {"threshold": [30, 50]},
    },
}


def generate_strategy(trend_name, entry_name, regime_name, tf, size, trend_params, entry_params, regime_params):
    """Generate a complete strategy.py from component templates."""
    trend_info = TREND_INDICATORS[trend_name]
    entry_info = ENTRY_FILTERS[entry_name]
    regime_info = REGIME_FILTERS[regime_name]

    trend_code = trend_info["code"].format(**trend_params)
    entry_code = entry_info["code"].format(**entry_params)
    regime_code = regime_info["code"].format(**regime_params)

    name = f"gen_{trend_name}_{entry_name}_{regime_name}_{tf}_v1"

    return f'''#!/usr/bin/env python3
"""Auto-generated: {trend_name} trend + {entry_name} entry + {regime_name} regime on {tf}"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "{name}"
timeframe = "{tf}"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    close_s = pd.Series(close)

    # ATR for stoploss
    _tr = np.zeros(n)
    for i in range(1, n): _tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = pd.Series(_tr).rolling(14, min_periods=14).mean().values

    # TREND indicator
{trend_code}

    # ENTRY filter
{entry_code}

    # REGIME filter
{regime_code}

    signals = np.zeros(n)
    SIZE = {size}
    entry_price = 0.0
    in_trade = 0

    for i in range(100, n):
        if np.isnan(atr[i]) or atr[i] == 0: continue

        # Manage position
        if in_trade != 0:
            if in_trade == 1 and close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == -1 and close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == 1 and trend[i] < 0:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == -1 and trend[i] > 0:
                signals[i] = 0.0; in_trade = 0; continue
            signals[i] = SIZE * in_trade; continue

        if not regime_ok[i]: signals[i] = 0.0; continue

        if trend[i] > 0 and entry_ok_long[i]:
            signals[i] = SIZE; entry_price = close[i]; in_trade = 1
        elif trend[i] < 0 and entry_ok_short[i]:
            signals[i] = -SIZE; entry_price = close[i]; in_trade = -1
        else:
            signals[i] = 0.0

    return signals
'''


def get_all_combos():
    """Generate all possible strategy combinations."""
    combos = []
    for trend in TREND_INDICATORS:
        for entry in ENTRY_FILTERS:
            for regime in REGIME_FILTERS:
                for tf in ["15m", "30m", "1h", "4h", "12h", "1d"]:
                    combos.append((trend, entry, regime, tf))
    return combos


if __name__ == "__main__":
    combos = get_all_combos()
    print(f"Total possible combinations: {len(combos)}")
    print(f"  Trend indicators: {len(TREND_INDICATORS)}")
    print(f"  Entry filters: {len(ENTRY_FILTERS)}")
    print(f"  Regime filters: {len(REGIME_FILTERS)}")
    print(f"  Timeframes: 6 (15m, 30m, 1h, 4h, 12h, 1d)")
    print(f"\nSample combos:")
    for c in random.sample(combos, min(10, len(combos))):
        print(f"  {c[0]} + {c[1]} + {c[2]} on {c[3]}")
