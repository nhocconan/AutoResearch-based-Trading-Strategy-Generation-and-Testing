#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Camarilla R1/S1 levels from 12h act as strong support/resistance in trending markets.
Price breaking above R1 or below S1 with volume confirms trend continuation.
In ranging markets, reversals occur at these levels.
Uses 12h trend filter to avoid counter-trend trades.
Works in bull/bear: trend filter adapts to market direction.
Timeframe: 4h balances trade frequency with signal quality.
"""
name = "4h_12h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # === 12H DATA FOR CAMARILLA PIVOTS ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and Camarilla levels for 12h
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r1_12h = close_12h + (range_12h * 1.1 / 12)  # R1 = C + (H-L)*1.1/12
    s1_12h = close_12h - (range_12h * 1.1 / 12)  # S1 = C - (H-L)*1.1/12
    r2_12h = close_12h + (range_12h * 1.1 / 6)   # R2 = C + (H-L)*1.1/6
    s2_12h = close_12h - (range_12h * 1.1 / 6)   # S2 = C - (H-L)*1.1/6
    
    # Align 12h levels to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # === 12H TREND FILTER (EMA 50) ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # For EMA 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(r2_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with volume, only if above 12h EMA50 (uptrend)
            if close[i] > r1_12h_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume, only if below 12h EMA50 (downtrend)
            elif close[i] < s1_12h_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or reaches R2
            if close[i] < s1_12h_aligned[i] or close[i] > r2_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or reaches S2
            if close[i] > r1_12h_aligned[i] or close[i] < s2_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals