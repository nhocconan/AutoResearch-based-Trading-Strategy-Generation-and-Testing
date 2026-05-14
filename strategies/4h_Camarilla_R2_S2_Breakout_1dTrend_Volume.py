#!/usr/bin/env python3
# 4h_Camarilla_R2_S2_Breakout_1dTrend_Volume
# Hypothesis: 4-hour Camarilla R2/S2 breakouts with 1-day trend filter and volume spikes.
# Uses tighter R2/S2 levels (less frequent than R1/S1) to reduce trade frequency while maintaining edge.
# Long: price breaks above R2 with daily uptrend (price>EMA34) and volume spike (>2x 20-period avg).
# Short: price breaks below S2 with daily downtrend (price<EMA34) and volume spike.
# Exit: price returns to Pivot Point (PP). Designed for ~25-40 trades/year to avoid fee drag.

name = "4h_Camarilla_R2_S2_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Classic Camarilla: R2 = C + 1.1*(H-L)/6, S2 = C - 1.1*(H-L)/6
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot components
    hl_range = high_1d - low_1d
    r2_1d = close_1d + 1.1 * hl_range / 6
    s2_1d = close_1d - 1.1 * hl_range / 6
    pp_1d = (high_1d + low_1d + close_1d) / 3  # Pivot Point
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate EMA34 for trend filter (daily)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2.0x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # Ensure we have Camarilla, EMA34, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R2, price above EMA34 (uptrend), volume spike
            if (close[i] > r2_1d_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2, price below EMA34 (downtrend), volume spike
            elif (close[i] < s2_1d_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below Pivot Point (PP)
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above Pivot Point (PP)
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals