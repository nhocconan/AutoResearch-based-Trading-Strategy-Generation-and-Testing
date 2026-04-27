#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, trade Camarilla R3/S3 breakouts in the direction of the weekly trend (EMA34 on 1w) with volume spike confirmation. 
Weekly trend filter ensures we only take breakouts aligned with the higher timeframe momentum, reducing whipsaws in counter-trend moves.
Volume spike confirms institutional participation. Designed for low trade frequency (12-37/year) to minimize fee drag on 6h.
Works in bull markets (breakouts with trend) and bear markets (avoids false breakouts via trend filter).
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
    
    # Calculate weekly Camarilla pivot levels (R3, S3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    PP = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    R3 = PP + range_1w * 1.1 / 4.0
    S3 = PP - range_1w * 1.1 / 4.0
    
    # Align weekly Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Weekly trend: EMA34 on weekly close
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for weekly EMA34 and volume average
    start_idx = max(100, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for entry: breakout in direction of weekly trend with volume spike
            weekly_uptrend = close[i] > ema34_1w_aligned[i]  # price above weekly EMA34 = uptrend
            weekly_downtrend = close[i] < ema34_1w_aligned[i]  # price below weekly EMA34 = downtrend
            
            long_entry = (close_val > R3_aligned[i]) and weekly_uptrend and volume_spike[i]
            short_entry = (close_val < S3_aligned[i]) and weekly_downtrend and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit on weekly trend reversal or S3 retracement (stop)
            weekly_downtrend = close[i] < ema34_1w_aligned[i]
            if weekly_downtrend or close_val < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on weekly trend reversal or R3 retracement (stop)
            weekly_uptrend = close[i] > ema34_1w_aligned[i]
            if weekly_uptrend or close_val > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0