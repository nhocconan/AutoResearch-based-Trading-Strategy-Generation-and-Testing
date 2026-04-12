#!/usr/bin/env python3
"""
1d_1w_Alligator_Momentum_v1
Hypothesis: Williams Alligator on weekly timeframe defines trend direction (bull/bear). 
On daily timeframe, enter long when price crosses above Alligator teeth (SMMA8) in bullish weekly trend, 
and short when price crosses below teeth in bearish weekly trend. Use volume confirmation (volume > 1.5x 20-day average) 
to avoid false breakouts. Exit when price crosses back over teeth or when weekly trend changes.
Designed for low trade frequency (10-25/year) to minimize fee drag. Works in bull (follow weekly trend) 
and bear (fade counter-trend moves at weekly extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Alligator_Momentum_v1"
timeframe = "1d"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Alligator (trend definition)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # === WEEKLY ALLIGATOR ===
    # Jaw: SMMA(13, 8)
    jaw = smma(weekly_close, 13)
    # Teeth: SMMA(8, 5) 
    teeth = smma(weekly_close, 8)
    # Lips: SMMA(5, 3)
    lips = smma(weekly_close, 5)
    
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Weekly trend: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    weekly_bullish = lips_aligned > teeth_aligned
    weekly_bearish = lips_aligned < teeth_aligned
    
    # === DAILY ENTRY CONDITIONS ===
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any data invalid
        if (np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Entry conditions based on weekly Alligator alignment
        long_entry = vol_confirm and weekly_bullish[i] and close[i] > teeth_aligned[i]
        short_entry = vol_confirm and weekly_bearish[i] and close[i] < teeth_aligned[i]
        
        # Exit conditions: price crosses back over teeth OR weekly trend changes
        long_exit = not weekly_bullish[i] or close[i] < teeth_aligned[i]
        short_exit = not weekly_bearish[i] or close[i] > teeth_aligned[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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