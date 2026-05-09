#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot + Volume Confirmation
# Uses weekly pivot points (PP, R1, S1) as dynamic support/resistance.
# Long when price crosses above weekly R1 with volume confirmation.
# Short when price crosses below weekly S1 with volume confirmation.
# Weekly context ensures alignment with higher timeframe structure, reducing whipsaw.
# Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in both bull and bear markets by following weekly pivot structure.
name = "6h_WeeklyPivot_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H + L + C)/3
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pp = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pp - low_w
    s1 = 2 * pp - high_w
    
    # Use previous week's levels to avoid look-ahead
    pp_shifted = np.roll(pp, 1)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    pp_shifted[0] = np.nan
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    # Align to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_w, pp_shifted)
    r1_6h = align_htf_to_ltf(prices, df_w, r1_shifted)
    s1_6h = align_htf_to_ltf(prices, df_w, s1_shifted)
    
    # Volume filter: volume > 1.5 x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price crosses above weekly R1 with volume confirmation
            if price > r1_6h[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly S1 with volume confirmation
            elif price < s1_6h[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot point (mean reversion)
            if price < pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot point (mean reversion)
            if price > pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals