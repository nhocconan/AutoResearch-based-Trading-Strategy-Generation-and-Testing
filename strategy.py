#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1D_Trend_Force_v2 (Adapted for 4h)
Hypothesis: Camarilla pivot levels (R1, S1) from daily timeframe act as significant support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and filtered by daily EMA50 trend provide high-probability entries.
In bull markets, buying R1 breakouts captures momentum; in bear markets, selling S1 breakouts captures reversals.
Designed for 4h timeframe to target 20-50 trades/year, avoiding excessive frequency and fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_1D_Trend_Force_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day's range
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R1 = Close + (Range * 1.1 / 12)
    # S1 = Close - (Range * 1.1 / 12)
    # Using previous day's data (already shifted by align_htf_to_ltf)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (waits for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for volume spike confirmation on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R1 with volume spike and price above daily EMA50 (bullish trend)
            if close[i] > r1_aligned[i] and vol_spike and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike and price below daily EMA50 (bearish trend)
            elif close[i] < s1_aligned[i] and vol_spike and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks back below R1 or breaks below EMA50
            if close[i] < r1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks back above S1 or breaks above EMA50
            if close[i] > s1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals