#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d ADX trend filter and volume confirmation.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend absence (all lines intertwined).
# Trade only when Alligator lines are separated (trending) AND 1d ADX>25 confirms trend strength.
# Volume spike (>1.5x average) adds confirmation. Designed to catch strong trends in both bull/bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_WilliamsAlligator_1dADX25_VolumeConfirm"
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
    
    # Get 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (14-period)
    tr14 = np.full_like(tr, np.nan)
    dm_plus14 = np.full_like(dm_plus, np.nan)
    dm_minus14 = np.full_like(dm_minus, np.nan)
    
    for i in range(14, len(tr)):
        if i == 14:
            tr14[i] = np.nansum(tr[1:15])
            dm_plus14[i] = np.nansum(dm_plus[1:15])
            dm_minus14[i] = np.nansum(dm_minus[1:15])
        else:
            tr14[i] = tr14[i-1] - (tr14[i-1]/14) + tr[i]
            dm_plus14[i] = dm_plus14[i-1] - (dm_plus14[i-1]/14) + dm_plus[i]
            dm_minus14[i] = dm_minus14[i-1] - (dm_minus14[i-1]/14) + dm_minus[i]
    
    # DI and DX
    di_plus = np.full_like(tr, np.nan)
    di_minus = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    for i in range(14, len(tr)):
        if tr14[i] != 0:
            di_plus[i] = 100 * dm_plus14[i] / tr14[i]
            di_minus[i] = 100 * dm_minus14[i] / tr14[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX (14-period smoothed DX)
    adx = np.full_like(dx, np.nan)
    for i in range(28, len(dx)):  # Need 28 periods: 14 for DX + 14 for smoothing
        if i == 28:
            valid_dx = dx[14:28]
            if len(valid_dx) > 0 and not np.all(np.isnan(valid_dx)):
                adx[i] = np.nanmean(valid_dx)
        else:
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 6h data
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 13)  # Need ADX and Alligator ready
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        j, t, l = jaw[i], teeth[i], lips[i]
        
        # Check if Alligator is sleeping (all lines intertwined) or awakening (separated)
        # Alligator sleeping: jaws, teeth, lips are close together (intertwined)
        # Alligator awakening: lines are separated, indicating trend
        max_val = max(j, t, l)
        min_val = min(j, t, l)
        spread = max_val - min_val
        # Normalize spread by price to make it scale-independent
        normalized_spread = spread / close[i] if close[i] != 0 else 0
        
        # Calculate 20-period volume average for spike confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:max(1,i)]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Alligator awakening (separated) AND ADX>25 AND lips > jaws (bullish alignment) AND volume confirmation
            if (normalized_spread > 0.001 and  # Alligator lines separated
                adx_val > 25 and 
                l > j and  # Lips above jaws = bullish
                volume[i] > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator awakening (separated) AND ADX>25 AND lips < jaws (bearish alignment) AND volume confirmation
            elif (normalized_spread > 0.001 and  # Alligator lines separated
                  adx_val > 25 and 
                  l < j and  # Lips below jaws = bearish
                  volume[i] > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator sleeping (lines intertwine) OR ADX weakens OR lips cross below jaws
            if (normalized_spread <= 0.0005 or  # Alligator sleeping
                adx_val < 20 or 
                l <= j):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator sleeping (lines intertwine) OR ADX weakens OR lips cross above jaws
            if (normalized_spread <= 0.0005 or  # Alligator sleeping
                adx_val < 20 or 
                l >= j):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals