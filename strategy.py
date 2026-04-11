#!/usr/bin/env python3
# 6h_12h_1d_camarilla_pivot_volume_v1
# Strategy: 6h Camarilla pivot breakout with volume confirmation and 12h trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) act as institutional support/resistance.
# Breakout above R4 or below S4 with volume confirmation and aligned 12h trend gives high-probability moves.
# Works in bull/bear markets by following breakouts in direction of higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H - L) * 1.1/2
    # R3 = C + (H - L) * 1.1/4
    # S3 = C - (H - L) * 1.1/4
    # S4 = C - (H - L) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend alignment
        if (close[i] > r4_1d_aligned[i] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < s4_1d_aligned[i] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price retracement to R3/S3 or trend change
        elif position == 1 and (close[i] < r3_1d_aligned[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > s3_1d_aligned[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals