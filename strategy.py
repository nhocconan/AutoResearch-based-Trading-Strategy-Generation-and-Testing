#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Camarilla R1/S1 breakout + volume confirmation + 1w EMA50 trend filter.
Long when price breaks above 1w Camarilla R1 with volume confirmation and price > 1w EMA50 (uptrend).
Short when price breaks below 1w Camarilla S1 with volume confirmation and price < 1w EMA50 (downtrend).
Exit when price returns to the 1w Camarilla midpoint (H4/L4) or reverses with volume.
Uses 1w timeframe for structure (reduces noise) and 1d for entry timing and volume confirmation.
Designed to capture weekly breakouts with institutional volume while avoiding false breakouts in choppy markets.
Camarilla levels provide precise support/resistance based on prior week's range, effective in both trending and ranging markets.
Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag and improve test generalization.
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
    
    # Get 1w data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (based on prior 1w bar)
    range_1w = high_1w - low_1w
    r1_1w = close_1w + 0.833 * range_1w
    s1_1w = close_1w - 0.833 * range_1w
    midpoint_1w = close_1w  # Camarilla midpoint is close
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    midpoint_1w_aligned = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(midpoint_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R1 with volume and uptrend (price > EMA50)
            if (close[i] > r1_1w_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S1 with volume and downtrend (price < EMA50)
            elif (close[i] < s1_1w_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below S1 with volume (reversal)
            if (close[i] <= midpoint_1w_aligned[i] or 
                (close[i] < s1_1w_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above R1 with volume (reversal)
            if (close[i] >= midpoint_1w_aligned[i] or 
                (close[i] > r1_1w_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wCamarilla_R1S1_Breakout_Volume_EMA50_Trend"
timeframe = "1d"
leverage = 1.0