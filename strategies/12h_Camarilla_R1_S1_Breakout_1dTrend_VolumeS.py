#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: 12-hour Camarilla R1/S1 breakouts with 1-day trend filter and volume spikes.
# Uses proven Camarilla pivot levels from daily data (proven in DB) to capture institutional levels.
# Long: price breaks above R1 with daily uptrend and volume spike. Short: breaks below S1 with daily downtrend and volume spike.
# Exit: price returns to Pivot Point (PP) level. Designed for fewer trades (15-25/year) with high win rate.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    # Classic Camarilla: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot components
    hl_range = high_1d - low_1d
    r1_1d = close_1d + 1.1 * hl_range / 12
    s1_1d = close_1d - 1.1 * hl_range / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3  # Pivot Point
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
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
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, price above EMA34 (uptrend), volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, price below EMA34 (downtrend), volume spike
            elif (close[i] < s1_1d_aligned[i] and 
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