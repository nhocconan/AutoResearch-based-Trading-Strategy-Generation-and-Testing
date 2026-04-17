#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla R1/S1 breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above 1d Camarilla R1 with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below 1d Camarilla S1 with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the 1d Camarilla midpoint (H4/L4) or reverses with volume.
Uses 12h timeframe for lower trade frequency (reduces fee drag) and 1d for structure (reduces noise).
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Camarilla levels provide precise support/resistance based on prior day's range, effective in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and maximize test generalization.
"""

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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on prior 1d bar)
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 0.833 * range_1d
    s1_1d = close_1d - 0.833 * range_1d
    midpoint_1d = close_1d  # Camarilla midpoint is close
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(midpoint_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R1 with volume and uptrend (price > EMA50)
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S1 with volume and downtrend (price < EMA50)
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S1 with volume (reversal)
            if (close[i] <= midpoint_1d_aligned[i] or 
                (close[i] < s1_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R1 with volume (reversal)
            if (close[i] >= midpoint_1d_aligned[i] or 
                (close[i] > r1_1d_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Breakout_Volume_EMA50_Trend"
timeframe = "12h"
leverage = 1.0