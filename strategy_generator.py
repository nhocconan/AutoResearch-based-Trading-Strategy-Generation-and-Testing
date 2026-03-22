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
    "golden_cross": {
        "code": """
    sma50 = close_s.rolling(50, min_periods=50).mean().values
    sma200 = close_s.rolling(200, min_periods=200).mean().values
    trend = np.where(sma50 > sma200, 1.0, np.where(sma50 < sma200, -1.0, 0.0))""",
        "params": {},
    },
    "keltner_channel": {
        "code": """
    _kc_mid = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    _kc_tr = np.zeros(n)
    for i in range(1, n): _kc_tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _kc_atr = pd.Series(_kc_tr).rolling(20, min_periods=20).mean().values
    _kc_upper = _kc_mid + {mult} * _kc_atr
    _kc_lower = _kc_mid - {mult} * _kc_atr
    trend = np.zeros(n)
    for i in range(20, n):
        if close[i] > _kc_upper[i]: trend[i] = 1.0
        elif close[i] < _kc_lower[i]: trend[i] = -1.0
        else: trend[i] = trend[i-1]""",
        "params": {"mult": [1.5, 2.0, 2.5]},
    },
    "heikin_ashi": {
        "code": """
    ha_close = (prices["open"].values + high + low + close) / 4
    ha_open = np.zeros(n); ha_open[0] = (prices["open"].values[0] + close[0]) / 2
    for i in range(1, n): ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_ema = pd.Series(ha_close).ewm(span={period}, min_periods={period}, adjust=False).mean().values
    trend = np.where(ha_close > ha_ema, 1.0, np.where(ha_close < ha_ema, -1.0, 0.0))""",
        "params": {"period": [10, 21]},
    },
    "ma_ribbon": {
        "code": """
    _emas = [close_s.ewm(span=p, min_periods=p, adjust=False).mean().values for p in [8,13,21,34,55]]
    trend = np.zeros(n)
    for i in range(55, n):
        bullish = all(_emas[j][i] > _emas[j+1][i] for j in range(4))
        bearish = all(_emas[j][i] < _emas[j+1][i] for j in range(4))
        trend[i] = 1.0 if bullish else (-1.0 if bearish else 0.0)""",
        "params": {},
    },
    "trix": {
        "code": """
    _e1 = close_s.ewm(span={period}, min_periods={period}, adjust=False).mean()
    _e2 = _e1.ewm(span={period}, min_periods={period}, adjust=False).mean()
    _e3 = _e2.ewm(span={period}, min_periods={period}, adjust=False).mean().values
    _trix = np.zeros(n)
    for i in range(1, n): _trix[i] = (_e3[i] - _e3[i-1]) / _e3[i-1] * 10000 if _e3[i-1] != 0 else 0
    _trix_signal = pd.Series(_trix).rolling(9, min_periods=9).mean().values
    trend = np.where(_trix > _trix_signal, 1.0, np.where(_trix < _trix_signal, -1.0, 0.0))""",
        "params": {"period": [12, 15, 18]},
    },
    "pivot_breakout": {
        "code": """
    # Daily pivot points
    _prev_h = pd.Series(high).shift(1).rolling(6, min_periods=6).max().values
    _prev_l = pd.Series(low).shift(1).rolling(6, min_periods=6).min().values
    _prev_c = pd.Series(close).shift(1).values
    _pivot = (_prev_h + _prev_l + _prev_c) / 3
    _r1 = 2 * _pivot - _prev_l
    _s1 = 2 * _pivot - _prev_h
    trend = np.zeros(n)
    for i in range(10, n):
        if not np.isnan(_r1[i]):
            if close[i] > _r1[i]: trend[i] = 1.0
            elif close[i] < _s1[i]: trend[i] = -1.0
            else: trend[i] = trend[i-1]""",
        "params": {},
    },
    "darvas_box": {
        "code": """
    _box_high = np.zeros(n); _box_low = np.zeros(n)
    _box_high[0] = high[0]; _box_low[0] = low[0]
    _in_box = True; _box_start = 0
    trend = np.zeros(n)
    for i in range(1, n):
        if _in_box:
            if high[i] > _box_high[_box_start]:
                _box_high[i] = high[i]; _box_start = i; _box_low[i] = low[i]
            elif low[i] < _box_low[_box_start]:
                _box_low[i] = low[i]
            else:
                _box_high[i] = _box_high[i-1]; _box_low[i] = _box_low[i-1]
            if i - _box_start > {period}:
                _in_box = False
        else:
            _box_high[i] = _box_high[i-1]; _box_low[i] = _box_low[i-1]
            if close[i] > _box_high[i]: trend[i] = 1.0; _in_box = True; _box_start = i; _box_high[i] = high[i]; _box_low[i] = low[i]
            elif close[i] < _box_low[i]: trend[i] = -1.0; _in_box = True; _box_start = i; _box_high[i] = high[i]; _box_low[i] = low[i]
            else: trend[i] = trend[i-1]""",
        "params": {"period": [10, 20]},
    },
    "vwap_trend": {
        "code": """
    _cum_vol = np.cumsum(volume)
    _cum_vp = np.cumsum(close * volume)
    _vwap = np.where(_cum_vol > 0, _cum_vp / _cum_vol, close)
    _vwap_std = pd.Series(close - _vwap).rolling(20, min_periods=20).std().values
    trend = np.zeros(n)
    for i in range(20, n):
        if close[i] > _vwap[i] + _vwap_std[i]: trend[i] = 1.0
        elif close[i] < _vwap[i] - _vwap_std[i]: trend[i] = -1.0
        else: trend[i] = trend[i-1]""",
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
    "connors_rsi": {
        "code": """
    # Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    _d = np.diff(close, prepend=close[0])
    _g = np.where(_d>0,_d,0); _l = np.where(_d<0,-_d,0)
    _rsi3 = 100 - 100/(1+np.where(pd.Series(_l).ewm(span=3,min_periods=3,adjust=False).mean().values>0, pd.Series(_g).ewm(span=3,min_periods=3,adjust=False).mean().values/pd.Series(_l).ewm(span=3,min_periods=3,adjust=False).mean().values, 100))
    _streak = np.zeros(n)
    for i in range(1,n): _streak[i] = (_streak[i-1]+1 if close[i]>close[i-1] else (_streak[i-1]-1 if close[i]<close[i-1] else 0))
    _streak_rsi = 100 - 100/(1+np.where(pd.Series(np.where(np.diff(_streak,prepend=0)<0,-np.diff(_streak,prepend=0),0)).ewm(span=2,min_periods=2,adjust=False).mean().values>0, pd.Series(np.where(np.diff(_streak,prepend=0)>0,np.diff(_streak,prepend=0),0)).ewm(span=2,min_periods=2,adjust=False).mean().values/pd.Series(np.where(np.diff(_streak,prepend=0)<0,-np.diff(_streak,prepend=0),0)).ewm(span=2,min_periods=2,adjust=False).mean().values, 100))
    _pct_rank = pd.Series(close.astype(float)).rolling(100,min_periods=50).rank(pct=True).values * 100
    _crsi = (_rsi3 + _streak_rsi + np.nan_to_num(_pct_rank, nan=50)) / 3
    entry_ok_long = np.array([_crsi[i] < {long_max} for i in range(n)])
    entry_ok_short = np.array([_crsi[i] > {short_min} for i in range(n)])""",
        "params": {"long_max": [15, 25], "short_min": [75, 85]},
    },
    "fisher_transform": {
        "code": """
    _hl_mid = (pd.Series(high).rolling(9,min_periods=9).max().values + pd.Series(low).rolling(9,min_periods=9).min().values) / 2
    _hl_range = pd.Series(high).rolling(9,min_periods=9).max().values - pd.Series(low).rolling(9,min_periods=9).min().values
    _norm = np.where(_hl_range>0, 2*(close - _hl_mid)/_hl_range, 0)
    _norm = np.clip(_norm, -0.999, 0.999)
    _fisher = np.zeros(n)
    for i in range(1,n): _fisher[i] = 0.5*np.log((1+_norm[i])/(1-_norm[i])) * 0.5 + _fisher[i-1]*0.5
    entry_ok_long = np.array([_fisher[i] < {long_max} for i in range(n)])
    entry_ok_short = np.array([_fisher[i] > {short_min} for i in range(n)])""",
        "params": {"long_max": [-1.0, -1.5], "short_min": [1.0, 1.5]},
    },
    "awesome_osc": {
        "code": """
    _ao_fast = pd.Series((high+low)/2).rolling(5,min_periods=5).mean().values
    _ao_slow = pd.Series((high+low)/2).rolling(34,min_periods=34).mean().values
    _ao = _ao_fast - _ao_slow
    entry_ok_long = np.array([_ao[i]>0 and (i<1 or _ao[i]>_ao[i-1]) for i in range(n)])
    entry_ok_short = np.array([_ao[i]<0 and (i<1 or _ao[i]<_ao[i-1]) for i in range(n)])""",
        "params": {},
    },
    "rvi": {
        "code": """
    _rvi_num = pd.Series((close - prices["open"].values) if "open" in prices.columns else np.zeros(n)).rolling(10,min_periods=10).mean().values
    _rvi_den = pd.Series(high - low).rolling(10,min_periods=10).mean().values
    _rvi = np.where(_rvi_den>0, _rvi_num/_rvi_den, 0)
    _rvi_sig = pd.Series(_rvi).rolling(4,min_periods=4).mean().values
    entry_ok_long = np.array([_rvi[i]>_rvi_sig[i] for i in range(n)])
    entry_ok_short = np.array([_rvi[i]<_rvi_sig[i] for i in range(n)])""",
        "params": {},
    },
    "ad_line": {
        "code": """
    _clv = np.where(high-low>0, (2*close-low-high)/(high-low), 0)
    _ad = np.cumsum(_clv * volume)
    _ad_ema = pd.Series(_ad).ewm(span=21,min_periods=21,adjust=False).mean().values
    entry_ok_long = np.array([_ad[i]>_ad_ema[i] if not np.isnan(_ad_ema[i]) else False for i in range(n)])
    entry_ok_short = np.array([_ad[i]<_ad_ema[i] if not np.isnan(_ad_ema[i]) else False for i in range(n)])""",
        "params": {},
    },
    "pin_bar": {
        "code": """
    _body = np.abs(close - prices["open"].values) if "open" in prices.columns else np.ones(n)
    _range = high - low
    _upper_wick = high - np.maximum(close, prices["open"].values if "open" in prices.columns else close)
    _lower_wick = np.minimum(close, prices["open"].values if "open" in prices.columns else close) - low
    entry_ok_long = np.array([_lower_wick[i] > 2*_body[i] and _range[i]>0 for i in range(n)])
    entry_ok_short = np.array([_upper_wick[i] > 2*_body[i] and _range[i]>0 for i in range(n)])""",
        "params": {},
    },
    "engulfing": {
        "code": """
    _open = prices["open"].values if "open" in prices.columns else close
    entry_ok_long = np.zeros(n, dtype=bool)
    entry_ok_short = np.zeros(n, dtype=bool)
    for i in range(1, n):
        # Bullish engulfing: prev bearish + current bullish + current body engulfs prev
        if _open[i-1]>close[i-1] and close[i]>_open[i] and close[i]>_open[i-1] and _open[i]<close[i-1]:
            entry_ok_long[i] = True
        # Bearish engulfing
        if close[i-1]>_open[i-1] and _open[i]>close[i] and _open[i]>close[i-1] and close[i]<_open[i-1]:
            entry_ok_short[i] = True""",
        "params": {},
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
    "keltner_squeeze": {
        "code": """
    _kc_mid = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    _kc_tr = np.zeros(n)
    for i in range(1, n): _kc_tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _kc_atr = pd.Series(_kc_tr).rolling(20, min_periods=20).mean().values
    _kc_upper = _kc_mid + 1.5 * _kc_atr; _kc_lower = _kc_mid - 1.5 * _kc_atr
    _bb_mid = close_s.rolling(20, min_periods=20).mean().values
    _bb_std = close_s.rolling(20, min_periods=20).std().values
    _bb_upper = _bb_mid + 2 * _bb_std; _bb_lower = _bb_mid - 2 * _bb_std
    regime_ok = np.array([not np.isnan(_kc_upper[i]) and _bb_upper[i] < _kc_upper[i] for i in range(n)])""",
        "params": {},
    },
    "sma200_regime": {
        "code": """
    _sma200 = close_s.rolling(200, min_periods=200).mean().values
    regime_ok = np.array([not np.isnan(_sma200[i]) and close[i] > _sma200[i] for i in range(n)])""",
        "params": {},
    },
    "vol_regime": {
        "code": """
    _vol_tr = np.zeros(n)
    for i in range(1, n): _vol_tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _vol_atr = pd.Series(_vol_tr).rolling(14, min_periods=14).mean().values
    _vol_pct = np.where(close > 0, _vol_atr / close, 0)
    _vol_pct_median = pd.Series(_vol_pct).rolling(100, min_periods=50).median().values
    regime_ok = np.array([not np.isnan(_vol_pct_median[i]) and _vol_pct[i] < _vol_pct_median[i] * {max_ratio} for i in range(n)])""",
        "params": {"max_ratio": [1.5, 2.0]},
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
