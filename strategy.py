#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_And_Chop_Regime_v2
Hypothesis: Kaufman Adaptive Moving Average (KAMA) trend filter with volume confirmation and choppiness regime filter.
Long when price > KAMA + volume spike + chop < 61.8 (trending), short when price < KAMA + volume spike + chop < 61.8.
Uses weekly EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Designed for 30-100 total trades over 4 years (7-25/year) with discrete position sizing (0.0, ±0.30).
Works in both bull and bear markets by combining adaptive trend (KAMA) with regime filter (chop) and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Kaufman Adaptive Moving Average (KAMA) - primary trend
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_len] = close[er_len]
        for i in range(er_len + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    
    # Choppiness Index - regime filter
    def choppiness_index(high, low, close, cp_len=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        for i in range(cp_len, len(close)):
            if i == cp_len:
                atr_sum = np.sum(atr[i-cp_len+1:i+1])
            else:
                atr_sum = atr_sum - atr[i-cp_len] + atr[i]
            highest_high = np.max(high[i-cp_len+1:i+1])
            lowest_low = np.min(low[i-cp_len+1:i+1])
            if atr_sum != 0:
                chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(cp_len)
            else:
                chop = 50
            atr[i] = chop  # reuse array for chop values
        return atr
    
    chop_vals = choppiness_index(high, low, close)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Higher timeframe trend filter: weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(chop_vals[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Discrete position sizing
        base_size = 0.30
        
        # Trending regime: chop < 61.8
        trending_regime = chop_vals[i] < 61.8
        
        # Long logic: price > KAMA + volume spike + trending + price > weekly EMA50 (uptrend filter)
        if close[i] > kama_vals[i] and volume_spike[i] and trending_regime and close[i] > ema_50_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price < KAMA + volume spike + trending + price < weekly EMA50 (downtrend filter)
        elif close[i] < kama_vals[i] and volume_spike[i] and trending_regime and close[i] < ema_50_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: loss of trend or volume
        elif position == 1 and (close[i] <= kama_vals[i] or not volume_spike[i] or chop_vals[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= kama_vals[i] or not volume_spike[i] or chop_vals[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_With_Volume_And_Chop_Regime_v2"
timeframe = "1d"
leverage = 1.0