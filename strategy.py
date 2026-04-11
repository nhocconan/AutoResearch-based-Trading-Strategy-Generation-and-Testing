#!/usr/bin/env python3
# 6h_1w_camarilla_volume_v1
# Strategy: 6s Camarilla pivot levels from 1d, with 1w trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) work better when aligned with weekly trend and volume spikes.
# In bull/bear markets, weekly trend filters out counter-trend trades. Volume confirms institutional interest.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_camarilla_volume_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (H, L, C)
    # Using yesterday's H, L, C for today's levels (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: multiplier = 1.1/12
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    camarilla_multiplier = 1.1 / 12
    r4 = close_1d + (high_1d - low_1d) * camarilla_multiplier * 6  # *6 because 1.1/2 = 6*(1.1/12)
    r3 = close_1d + (high_1d - low_1d) * camarilla_multiplier * 3  # *3 because 1.1/4 = 3*(1.1/12)
    s3 = close_1d - (high_1d - low_1d) * camarilla_multiplier * 3
    s4 = close_1d - (high_1d - low_1d) * camarilla_multiplier * 6
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Mean reversion at S3/R3 with volume spike
        long_mr = (close[i] <= s3_6h[i]) and vol_spike.iloc[i] and uptrend and position != 1
        short_mr = (close[i] >= r3_6h[i]) and vol_spike.iloc[i] and downtrend and position != -1
        
        # Breakout continuation at S4/R4 with volume spike
        long_break = (close[i] >= s4_6h[i]) and vol_spike.iloc[i] and uptrend and position != 1
        short_break = (close[i] <= r4_6h[i]) and vol_spike.iloc[i] and downtrend and position != -1
        
        # Entry logic
        if long_mr or long_break:
            position = 1
            signals[i] = 0.25
        elif short_mr or short_break:
            position = -1
            signals[i] = -0.25
        # Exit when price reverts to mean (opposite S3/R3) or volume dries up
        elif position == 1 and (close[i] >= r3_6h[i] or not vol_spike.iloc[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= s3_6h[i] or not vol_spike.iloc[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals