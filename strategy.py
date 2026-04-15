#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + ADX Trend Filter
# The Williams Alligator (Jaw/Teeth/Lips) identifies trend direction via SMAs.
# Trades only when price is outside the Alligator's mouth (trending) with volume spike and ADX > 25.
# Works in bull markets (price above Lips) and bear markets (price below Jaw).
# Target: 50-150 total trades over 4 years. Timeframe: 12h, HTF: 1d for Alligator.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with future shift)
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)  # Shift forward 8 bars
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)  # Shift forward 5 bars
    
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)  # Shift forward 3 bars
    
    # Align Alligator lines to 12h timeframe (wait for 1d bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(close_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(close_12h, 1)), 
                        np.maximum(np.roll(close_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price above Lips (bullish alignment) + volume spike + ADX > 25
        if (close[i] > lips_aligned[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below Jaw (bearish alignment) + volume spike + ADX > 25
        elif (close[i] < jaw_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price re-enters Alligator's mouth or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < teeth_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > teeth_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0