#!/usr/bin/env python3
"""
1D_Weekly_Keltner_Breakout
Hypothesis: In the daily timeframe, price breaks above/below the weekly Keltner Channel
with volume confirmation and ADX trend filter. Weekly trend direction from EMA200 filters
trades to align with higher timeframe momentum. Works in bull/bear by following the
weekly trend and using volatility-based breakouts with volume confirmation to avoid
false signals. Target: 10-20 trades/year per symbol.
"""

name = "1D_Weekly_Keltner_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # ATR for Keltner Channel (10-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    atr = np.concatenate([np.full(1, np.nan), atr])
    
    # EMA20 for basis line
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # ADX for trend strength (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr_atr = np.zeros_like(tr)
    if len(tr) > 0:
        tr_atr[0] = tr[0]
        for i in range(1, len(tr)):
            tr_atr[i] = (tr_atr[i-1] * 13 + tr[i]) / 14
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                     pd.Series(tr_atr).ewm(alpha=1/14, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                      pd.Series(tr_atr).ewm(alpha=1/14, adjust=False).mean())
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx = np.concatenate([np.full(1, np.nan), adx])
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = close_1w > ema200_1w
    weekly_downtrend = close_1w < ema200_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        strong_trend = adx[i] > 25
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + strong daily trend + price breaks above Keltner upper + volume
            if weekly_up and strong_trend and volume_confirm:
                if close[i] > keltner_upper[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: weekly downtrend + strong daily trend + price breaks below Keltner lower + volume
            elif weekly_down and strong_trend and volume_confirm:
                if close[i] < keltner_lower[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: weekly trend changes OR price returns to EMA20
            if not weekly_up or close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly trend changes OR price returns to EMA20
            if not weekly_down or close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals