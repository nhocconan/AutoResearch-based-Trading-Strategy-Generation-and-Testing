#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Bounce_v1
Hypothesis: Use daily Camarilla pivot levels with volume confirmation on 12h.
Long when price bounces off daily S3/S4 with volume > 1.5x 20-period average,
short when price reverses from daily R3/R4 with volume > 1.5x 20-period average.
Camarilla levels provide strong intraday support/resistance; volume confirms
the bounce/reversal strength. Designed for low trade frequency (target: 50-150 total over 4 years)
to minimize fee drift. Works in bull via bounces from support, in bear via reversals from resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Bounce_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate daily Camarilla pivot levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Camarilla levels: H5/L5, H4/L4, H3/L3, H1/L1
    # H4 = Close + Range * 1.1/2, L4 = Close - Range * 1.1/2
    # H3 = Close + Range * 1.1/4, L3 = Close - Range * 1.1/4
    camarilla_h4 = prev_close + range_val * 1.1 / 2
    camarilla_l4 = prev_close - range_val * 1.1 / 2
    camarilla_h3 = prev_close + range_val * 1.1 / 4
    camarilla_l3 = prev_close - range_val * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_h4_array = np.full(len(df_1d), camarilla_h4)
    camarilla_l4_array = np.full(len(df_1d), camarilla_l4)
    camarilla_h3_array = np.full(len(df_1d), camarilla_h3)
    camarilla_l3_array = np.full(len(df_1d), camarilla_l3)
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_array)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_array)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_array)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_array)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Bounce/reversal conditions with volume filter
        long_bounce = (low[i] <= camarilla_l3_aligned[i] or low[i] <= camarilla_l4_aligned[i]) and vol_ratio[i] > 1.5
        short_reversal = (high[i] >= camarilla_h3_aligned[i] or high[i] >= camarilla_h4_aligned[i]) and vol_ratio[i] > 1.5
        
        # Exit conditions: price moves back toward the pivot (midpoint between H3 and L3)
        camarilla_pivot = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
        long_exit = close[i] >= camarilla_pivot
        short_exit = close[i] <= camarilla_pivot
        
        # Signal logic
        if long_bounce and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_reversal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals