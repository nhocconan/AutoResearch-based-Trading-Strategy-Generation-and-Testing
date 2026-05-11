#!/usr/bin/env python3
"""
6h_Keltner_MeanReversion_1wTrend
Hypothesis: Mean revert off Keltner lower/upper bands in ranging markets (identified by weekly ADX < 20), with weekly trend filter (price > weekly EMA50 for longs, < for shorts). Works in bull/bear by using weekly trend to avoid counter-trend trades. Target: 15-30 trades/year on 6h.
"""

name = "6h_Keltner_MeanReversion_1wTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # === Weekly Data for Trend and Regime ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly ADX for ranging market detection (ADX < 20 = range)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    adx_6h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === 6h Keltner Channel (20, 2.0) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean()
    upper_keltner = ema20 + 2.0 * atr
    lower_keltner = ema20 - 2.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_6h[i]) or np.isnan(adx_6h[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (weekly ADX < 20)
        if adx_6h[i] >= 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches lower Keltner band and weekly uptrend
            if (low[i] <= lower_keltner[i] and 
                close[i] > ema50_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Keltner band and weekly downtrend
            elif (high[i] >= upper_keltner[i] and 
                  close[i] < ema50_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above EMA20 (mean reversion complete) or hits upper band
            if close[i] >= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses below EMA20 (mean reversion complete) or hits lower band
            if close[i] <= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals