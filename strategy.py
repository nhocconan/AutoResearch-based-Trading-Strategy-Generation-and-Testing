#!/usr/bin/env python3
# 12h_1w_Pivot_R3S4_Reversion_With_Volume_Confirmation
# Hypothesis: Fade at weekly R3/S4 levels with volume confirmation on 12h timeframe.
# Uses weekly trend filter (price vs weekly EMA50) to avoid counter-trend trades.
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull/bear via weekly trend filter - only trade with the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Pivot_R3S4_Reversion_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Calculate weekly pivot levels (R3, S4) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S4 = close - (range * 1.1/2)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # === Weekly EMA50 for trend filter ===
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all weekly levels to 12h
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r3_1w_val = r3_1w_aligned[i]
        s4_1w_val = s4_1w_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1w_val) or np.isnan(s4_1w_val) or np.isnan(ema50_1w_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S4 (bounces off support) with volume confirmation and above weekly EMA50
            if (close_val < s4_1w_val and  # Price touched or went below S4
                prices['low'].iloc[i] <= s4_1w_val and  # Confirmed touch of S4
                close_val > s4_1w_val and  # Now bouncing back above S4
                vol_ratio_val > 2.0 and  # Volume confirmation
                close_val > ema50_1w_val):  # Only long in weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R3 (bounces off resistance) with volume confirmation and below weekly EMA50
            elif (close_val > r3_1w_val and  # Price touched or went above R3
                  prices['high'].iloc[i] >= r3_1w_val and  # Confirmed touch of R3
                  close_val < r3_1w_val and  # Now falling back below R3
                  vol_ratio_val > 2.0 and  # Volume confirmation
                  close_val < ema50_1w_val):  # Only short in weekly downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R3 or shows weakness
            if close_val >= r3_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S4 or shows weakness
            if close_val <= s4_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals