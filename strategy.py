#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d ADX trend filter + volume spike
# Long when: Green alignment (jaw<teeth<lips), ADX>25, volume>1.5x average
# Short when: Red alignment (jaw>teeth>lips), ADX>25, volume>1.5x average
# Williams Alligator uses SMAs of median price (hlc3) with specific periods
# ADX filters for trending markets only to avoid whipsaws in ranging conditions
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag

name = "6h_WilliamsAlligator_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data once for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h median price
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    median_price_6h = (high_6h + low_6h + close_6h) / 3.0
    
    # Jaw (blue line): 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price_6h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (red line): 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price_6h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (green line): 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price_6h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data once for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Alligator and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: Green alignment (jaw<teeth<lips), ADX>25, volume spike
            if jaw_val < teeth_val < lips_val and adx_val > 25 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Red alignment (jaw>teeth>lips), ADX>25, volume spike
            elif jaw_val > teeth_val > lips_val and adx_val > 25 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Red alignment or ADX<20
            if jaw_val > teeth_val > lips_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Green alignment or ADX<20
            if jaw_val < teeth_val < lips_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals