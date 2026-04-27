#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with daily volume confirmation and weekly trend filter.
Enters long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA) with above-average volume and weekly uptrend.
Enters short when jaws cross below teeth with above-average volume and weekly downtrend.
Uses weekly timeframe for trend structure and 12h for execution to reduce noise.
Williams Alligator is effective in trending markets and avoids whipsaws in ranging conditions.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 12h Williams Alligator components
    # Jaw: 13-period SMMA of median price
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)
    
    # Teeth: 8-period SMMA of median price
    teeth = smma(median_price, 8)
    
    # Lips: 5-period SMMA of median price (not used in signal but part of Alligator)
    lips = smma(median_price, 5)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator components, volume MA, and weekly EMA
    start_idx = max(13, 8, 5, 20, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_21_1w_aligned[i]
        
        # Current Alligator values
        jaw_now = jaw_aligned[i]
        teeth_now = teeth_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Alligator signals: jaws crossing teeth
        jaw_above_teeth = jaw_now > teeth_now
        jaw_below_teeth = jaw_now < teeth_now
        
        # Entry conditions
        if position == 0:
            # Long: jaws cross above teeth with volume + weekly uptrend
            if jaw_above_teeth and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: jaws cross below teeth with volume + weekly downtrend
            elif jaw_below_teeth and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: jaws cross below teeth or weekly trend turns down
            if jaw_below_teeth or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: jaws cross above teeth or weekly trend turns up
            if jaw_above_teeth or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0