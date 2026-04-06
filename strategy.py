#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume and Choppiness Filter
Hypothesis: Camarilla pivot levels on 1d timeframe act as strong reversal points.
Price retracing to L3 or H3 levels with volume confirmation and low choppiness
(indicating range-bound behavior) provides high-probability reversals.
Works in both bull and bear markets by fading extreme moves at key levels.
Designed for 75-200 trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_reversal_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot and choppiness (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    range_1d = prev_high - prev_low
    camarilla_h3 = prev_close + range_1d * 1.1 / 4
    camarilla_l3 = prev_close - range_1d * 1.1 / 4
    camarilla_h4 = prev_close + range_1d * 1.1 / 2
    camarilla_l4 = prev_close - range_1d * 1.1 / 2
    
    # Choppiness index (14-period) - range detection
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(np.maximum(high_1d - low_1d, 
                               np.abs(high_1d - np.roll(close_1d, 1))),
                    np.abs(low_1d - np.roll(close_1d, 1)))
    tr[0] = high_1d[0] - low_1d[0]
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(10)
    chop[~np.isfinite(chop)] = 50  # default to middle when invalid
    
    chop_range = chop > 61.8  # ranging market
    
    # Align 1d data to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Camarilla and chop calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ema[i]) or np.isnan(chop_range_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price moves back to center (P) or stoploss
        if position == 1:  # long position (entered at L3/L4)
            # Exit: price reaches midpoint between H3 and L3 or stoploss
            mid_point = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if (close[i] >= mid_point or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position (entered at H3/H4)
            # Exit: price reaches midpoint between H3 and L3 or stoploss
            mid_point = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if (close[i] <= mid_point or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price at L3/L4 or H3/H4 with volume and chop filter
            near_l3 = low[i] <= camarilla_l3_aligned[i] * 1.002  # within 0.2%
            near_l4 = low[i] <= camarilla_l4_aligned[i] * 1.002
            near_h3 = high[i] >= camarilla_h3_aligned[i] * 0.998  # within 0.2%
            near_h4 = high[i] >= camarilla_h4_aligned[i] * 0.998
            
            vol_ok = volume[i] > vol_ema[i] * 1.3
            chop_ok = chop_range_aligned[i]  # only trade in ranging markets
            
            # Long when price touches L3/L4 in ranging market with volume
            if ((near_l3 or near_l4) and vol_ok and chop_ok):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short when price touches H3/H4 in ranging market with volume
            elif ((near_h3 or near_h4) and vol_ok and chop_ok):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals