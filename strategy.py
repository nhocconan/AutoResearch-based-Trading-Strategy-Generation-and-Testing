#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Weekly Pivot Breakout with Volume Confirmation
# Buy when price breaks above weekly pivot point (PP) with volume > 1.5x 48-period average
# Sell when price breaks below weekly pivot point (PP) with volume > 1.5x 48-period average
# Weekly pivot calculated from prior week's high/low/close: PP = (H+L+C)/3
# Volume filter ensures breakout conviction; 48-period MA = 6h * 8 = 2 days
# Designed for ~20-35 trades/year per symbol (~80-140 total over 4 years)
# Works in bull (breakouts up) and bear (breakouts down) via symmetric logic
name = "6s_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot point: PP = (H + L + C)/3
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    pp_w = (high_w + low_w + close_w) / 3.0
    
    # Align weekly pivot to 6h timeframe (wait for weekly bar to close)
    pp_w_aligned = align_htf_to_ltf(prices, df_w, pp_w)
    
    # Volume filter: current volume > 1.5 * 48-period average (48 * 6h = 12 days)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_filter = volume > (1.5 * vol_ma_48)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_w_aligned[i]) or np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pp_val = pp_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly pivot with volume confirmation
            if close_val > pp_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly pivot with volume confirmation
            elif close_val < pp_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly pivot
            if close_val < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly pivot
            if close_val > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals