#!/usr/bin/env python3
# 1d_1W_Pivot_R1S1_Breakout_Volume
# Hypothesis: On daily chart, breakout above weekly pivot R1 with volume confirmation gives long signal; breakdown below weekly pivot S1 with volume confirmation gives short signal.
# Uses weekly pivot levels for structure, daily breakout for entry, volume to filter false breaks. Works in bull/bear by only taking breaks in direction of weekly trend (price above/below weekly pivot point).
# Targets 15-25 trades/year to minimize fee drag on 1d timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1W_Pivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly Pivot Points (standard calculation) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot Point = (H + L + C)/3
    pivot_point = (high_1w + low_1w + close_1w) / 3
    
    # Resistance 1 = (2*P) - L
    r1 = (2 * pivot_point) - low_1w
    
    # Support 1 = (2*P) - H
    s1 = (2 * pivot_point) - high_1w
    
    # Align weekly pivot levels to daily
    pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after first bar to have previous close
        # Get values
        close_val = prices['close'].iloc[i]
        prev_close = prices['close'].iloc[i-1]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_point_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(pivot_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume confirmation AND price above weekly pivot (bullish bias)
            if (close_val > r1_val and  # Break above R1
                prev_close <= r1_val and  # Was not above R1 previous bar (confirm break)
                pivot_val > 0 and  # Valid pivot level
                close_val > pivot_val and  # Price above weekly pivot (bullish bias)
                vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume confirmation AND price below weekly pivot (bearish bias)
            elif (close_val < s1_val and  # Break below S1
                  prev_close >= s1_val and  # Was not below S1 previous bar (confirm break)
                  pivot_val > 0 and  # Valid pivot level
                  close_val < pivot_val and  # Price below weekly pivot (bearish bias)
                  vol_ratio_val > 1.5):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below weekly R1 or breaks below weekly pivot
            if close_val < r1_val or close_val < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above weekly S1 or breaks above weekly pivot
            if close_val > s1_val or close_val > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals