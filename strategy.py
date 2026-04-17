#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_VolumeConfirm_v2
Hypothesis: Daily Camarilla pivot levels (R1, S1) act as key support/resistance. 
Weekly trend filter ensures trades align with higher timeframe momentum.
Long when price breaks above daily R1 with volume confirmation AND weekly close above weekly open (bullish week).
Short when price breaks below daily S1 with volume confirmation AND weekly close below weekly open (bearish week).
Exit at opposite daily level (S1 for longs, R1 for shorts).
Designed for low trade frequency (<25/year) to minimize fee drag while capturing sustained moves.
Works in bull/bear because pivots adapt to recent price action and weekly filter avoids counter-trend trades.
"""

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
    
    # === 1-day Data (for Camarilla pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each day
    hl_range = high_1d - low_1d
    R1 = close_1d + 1.1 * hl_range / 12
    S1 = close_1d - 1.1 * hl_range / 12
    
    # Align daily levels to 1d timeframe (no additional delay needed for pivot levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily volume for confirmation (20-day average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1-week Data (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly bullish/bearish determination (close vs open)
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align weekly trend to 1d timeframe (wait for weekly close)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period (ensure sufficient data for indicators)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.5 * vol_ma_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation AND weekly bullish
            if close[i] > R1_aligned[i] and vol_confirmed and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation AND weekly bearish
            elif close[i] < S1_aligned[i] and vol_confirmed and weekly_bearish_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to opposite level
        elif position == 1:
            # Exit long: price returns to S1 or below
            if close[i] <= S1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to R1 or above
            if close[i] >= R1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1S1_Breakout_VolumeConfirm_v2"
timeframe = "1d"
leverage = 1.0