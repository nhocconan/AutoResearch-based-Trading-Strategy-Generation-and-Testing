#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume spike
# Camarilla R3/S3 levels provide institutional support/resistance from 1d price action
# Breakout continuation (price > R4 or < S4) captures strong momentum moves
# 1d EMA34 trend filter ensures trades align with higher timeframe direction
# Volume spike (>2.0x 20-period EMA volume) confirms participation
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels from prior completed 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1/2
    # S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + (range_1d * 1.1 / 2)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2)
    r4_1d = pivot_1d + (range_1d * 1.1)
    s4_1d = pivot_1d - (range_1d * 1.1)
    
    # Shift by 1 to use only completed 1d bar
    r3_1d_shifted = np.roll(r3_1d, 1)
    s3_1d_shifted = np.roll(s3_1d, 1)
    r4_1d_shifted = np.roll(r4_1d, 1)
    s4_1d_shifted = np.roll(s4_1d, 1)
    r3_1d_shifted[0] = np.nan
    s3_1d_shifted[0] = np.nan
    r4_1d_shifted[0] = np.nan
    s4_1d_shifted[0] = np.nan
    
    # Align HTF Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_shifted)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_shifted)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d_shifted)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d_shifted)
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4 AND 1d EMA34 uptrend AND volume spike
            if close[i] > r4_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < s4_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 OR 1d EMA34 turns down
            if close[i] < r3_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 OR 1d EMA34 turns up
            if close[i] > s3_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals