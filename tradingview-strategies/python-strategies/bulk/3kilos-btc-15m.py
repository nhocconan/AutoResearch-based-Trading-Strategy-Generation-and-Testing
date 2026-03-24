#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "3kilos BTC 15m"
timeframe = "15m"
leverage = 1

def _ema(src, length):
    alpha = 2.0 / (length + 1.0)
    out = np.empty_like(src)
    out[0] = src[0]
    for i in range(1, len(src)):
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i-1]
    return out

def _tema(src, length):
    ema1 = _ema(src, length)
    ema2 = _ema(ema1, length)
    ema3 = _ema(ema2, length)
    return 3.0 * (ema1 - ema2) + ema3

def _atr(high, low, close, length):
    tr = np.empty_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return _ema(tr, length)

def _supertrend(high, low, close, atr, mult):
    n = len(close)
    up = np.empty(n)
    dn = np.empty(n)
    uptrend = np.empty(n, dtype=bool)
    hl2 = (high + low) / 2.0
    
    up[0] = hl2[0] - mult * atr[0]
    dn[0] = hl2[0] + mult * atr[0]
    uptrend[0] = True
    
    for i in range(1, n):
        basic_up = hl2[i] - mult * atr[i]
        basic_dn = hl2[i] + mult * atr[i]
        
        if uptrend[i-1]:
            up[i] = max(basic_up, up[i-1])
            dn[i] = basic_dn
        else:
            up[i] = basic_up
            dn[i] = min(basic_dn, dn[i-1])
            
        if close[i-1] > dn[i-1]:
            uptrend[i] = True
        elif close[i-1] < up[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            
    return uptrend

def generate_signals(prices):
    if isinstance(prices, pd.DataFrame):
        high = prices['high'].to_numpy()
        low = prices['low'].to_numpy()
        close = prices['close'].to_numpy()
        open_time = prices['open_time'].to_numpy()
    else:
        high = prices['high']
        low = prices['low']
        close = prices['close']
        open_time = prices['open_time']

    n = len(close)
    signals = np.zeros(n, dtype=int)
    
    short_len = 30
    long_len = 50
    long2_len = 140
    atr_len = 10
    st_mult = 2.0
    tp_pct = 0.01
    sl_pct = 0.01
    tema1 = _tema(high, short_len)
    tema2 = _tema(low, long_len)
    tema3 = _tema(close, long2_len)
    atr = _atr(high, low, close, atr_len)
    uptrend = _supertrend(high, low, close, atr, st_mult)
    
    position = 0
    entry_price = 0.0
    tp_level = 0.0
    sl_level = 0.0
    
    for i in range(n):
        if position != 0:
            exited = False
            if position == 1:
                if high[i] >= tp_level or low[i] <= sl_level:
                    exited = True
            elif position == -1:
                if low[i] <= tp_level or high[i] >= sl_level:
                    exited = True
            
            if exited:
                position = 0
                signals[i] = 0
                continue
            else:
                signals[i] = position
        
        if position == 0:
            if i > 0:
                t1_prev = tema1[i-1]
                t2_prev = tema2[i-1]
                t1_curr = tema1[i]
                t2_curr = tema2[i]
                t3_curr = tema3[i]
                
                if np.isnan(t1_curr) or np.isnan(t2_curr) or np.isnan(t3_curr):
                    signals[i] = 0
                    continue

                long_c = (t2_curr > t1_curr) and (t2_prev <= t1_prev)
                short_c = (t1_curr > t2_curr) and (t1_prev <= t2_prev)
                
                long_cond = long_c and (t2_curr > t3_curr) and uptrend[i] and (t1_curr != 0.0)
                short_cond = short_c and (t2_curr < t3_curr) and (not uptrend[i]) and (t1_curr != 0.0)
                
                if long_cond:
                    position = 1
                    entry_price = close[i]
                    tp_level = entry_price * (1.0 + tp_pct)
                    sl_level = entry_price * (1.0 - sl_pct)
                    signals[i] = 1
                elif short_cond:
                    position = -1
                    entry_price = close[i]
                    tp_level = entry_price * (1.0 - tp_pct)
                    sl_level = entry_price * (1.0 + sl_pct)
                    signals[i] = -1
                else:
                    signals[i] = 0
            else:
                signals[i] = 0
                
    return signals
