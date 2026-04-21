#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_R3S3_Fade_V1
Hypothesis: Fade extreme weekly Camarilla R3/S3 levels on 6h timeframe with 1d trend filter.
In ranging markets, price reverts from extreme levels (R3/S3). In trending markets,
only trade with the 1d trend to avoid counter-trend fading. Uses 1w for Camarilla calculation
to capture major weekly extremes, reducing false signals. Target: 12-37 trades/year per symbol.
Works in both bull and bear markets via trend filter and mean reversion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for Camarilla pivot calculation (primary HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot point and R3/S3 levels (extreme levels for fading)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 1d EMA50 for trend filter (avoid fading against strong trend)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 24-period average (approx 6 days on 6h)
    vol_ma = prices['volume'].rolling(window=24, min_periods=24).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.3 * vol_ma[i]
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long fade: price reaches S3 extreme in uptrend (mean reversion up)
            # or price reaches S3 extreme in ranging market (no strong trend)
            if volume_ok and price <= s3_aligned[i]:
                if uptrend or (not uptrend and not downtrend):  # uptrend or ranging
                    signals[i] = 0.25
                    position = 1
            # Short fade: price reaches R3 extreme in downtrend (mean reversion down)
            # or price reaches R3 extreme in ranging market (no strong trend)
            elif volume_ok and price >= r3_aligned[i]:
                if downtrend or (not uptrend and not downtrend):  # downtrend or ranging
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches pivot (mean reversion target) or stoploss
            if price >= pivot_aligned[i] * 0.998 or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches pivot (mean reversion target) or stoploss
            if price <= pivot_aligned[i] * 1.002 or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_1d_Camarilla_R3S3_Fade_V1"
timeframe = "6h"
leverage = 1.0