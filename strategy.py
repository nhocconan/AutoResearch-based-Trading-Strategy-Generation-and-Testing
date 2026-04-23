#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
Long when Jaw < Teeth < Lips (bullish alignment) and price > Lips and volume > 1.5x average.
Short when Jaw > Teeth > Lips (bearish alignment) and price < Jaw and volume > 1.5x average.
Uses 12h timeframe to target 50-150 total trades over 4 years. Williams Alligator identifies
trend phases via smoothed moving averages. 1w trend filter ensures alignment with higher timeframe.
Volume confirmation reduces false signals. Works in both bull and bear markets by capturing
trends in either direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) / Wilder's MA"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    result = np.empty_like(series, dtype=np.float64)
    result[:] = np.nan
    # First value is simple SMA
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for Williams Alligator calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    median_1w = (high_1w + low_1w) / 2.0  # Typical price for Alligator
    
    # Calculate Williams Alligator lines (13,8,5 period SMMA with 8,5,3 offset)
    jaw = smma(median_1w, 13)  # Blue line (13-period)
    teeth = smma(median_1w, 8)  # Red line (8-period)
    lips = smma(median_1w, 5)   # Green line (5-period)
    
    # Apply offsets: Jaw offset 8, Teeth offset 5, Lips offset 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema34_val = ema34_1w_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Bullish alignment: Jaw < Teeth < Lips (Alligator sleeping then waking up)
            # Long when price > Lips (above green line) AND price > 1w EMA34 (uptrend) AND volume confirmation
            if (jaw_val < teeth_val and teeth_val < lips_val and price > lips_val and 
                price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Bearish alignment: Jaw > Teeth > Lips (Alligator sleeping then waking up)
            # Short when price < Jaw (below blue line) AND price < 1w EMA34 (downtrend) AND volume confirmation
            elif (jaw_val > teeth_val and teeth_val > lips_val and price < jaw_val and 
                  price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Teeth (red line) OR price breaks below 1w EMA34 (trend reversal)
                if price < teeth_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Teeth (red line) OR price breaks above 1w EMA34 (trend reversal)
                if price > teeth_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0