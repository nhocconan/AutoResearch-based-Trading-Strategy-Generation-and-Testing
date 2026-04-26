#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_v1
Hypothesis: Donchian(20) breakouts on 6h filtered by weekly Camarilla pivot direction (R3/S3) and volume confirmation.
In bull markets (price above weekly R3), long breakouts are favored; in bear markets (price below weekly S3), short breakouts are favored.
Weekly pivots provide structural support/resistance that adapts to long-term trends, reducing false breakouts in chop.
Targeting 50-120 total trades over 4 years (12-30/year) with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivots (based on prior week's OHLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    H_1w = df_1w['high'].values
    L_1w = df_1w['low'].values
    C_1w = df_1w['close'].values
    
    # Calculate pivot levels
    rng = H_1w - L_1w
    R3_1w = C_1w + (rng * 1.1 / 4)
    S3_1w = C_1w - (rng * 1.1 / 4)
    
    # Align weekly pivots to 6h (no extra delay needed as pivots are based on completed weekly bar)
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    
    # Calculate ATR(14) on 6h for Donchian bands and volume normalization
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Donchian(20) channels on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_1w_aligned[i]) or np.isnan(S3_1w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly pivot direction filter
        price_above_R3 = close[i] > R3_1w_aligned[i]
        price_below_S3 = close[i] < S3_1w_aligned[i]
        
        # Long logic: Donchian breakout above upper band + volume spike + price above weekly R3 (bullish bias)
        if close[i] > highest_high[i] and volume_spike[i] and price_above_R3:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Donchian breakdown below lower band + volume spike + price below weekly S3 (bearish bias)
        elif close[i] < lowest_low[i] and volume_spike[i] and price_below_S3:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Donchian band or weekly pivot level
        elif position == 1 and (close[i] < lowest_low[i] or close[i] < S3_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > highest_high[i] or close[i] > R3_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_v1"
timeframe = "6h"
leverage = 1.0