#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with volume confirmation and ADX trend filter.
# Williams %R identifies overbought/oversold conditions; long when %R < -80 and ADX > 25 (trending),
# short when %R > -20 and ADX > 25. Uses 1-day timeframe for Williams %R to avoid noise.
# Volume confirmation requires 4h volume > 1.5x 20-period EMA. Designed for low trade frequency.
# Works in both bull and bear markets by following the trend via ADX filter.

name = "4h_1dWilliamsR_ADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Williams %R (14-period)
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 1-day ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
    
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])  # First value has no TR
    
    atr = np.zeros_like(high_1d)
    atr[0] = np.nan
    for i in range(1, len(high_1d)):
        if i < 14:
            atr[i] = np.nan
        else:
            atr[i] = np.nanmean(tr[i-13:i+1])  # Wilder's smoothing
    
    # Avoid division by zero in DI calculation
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    
    dx = np.zeros_like(high_1d)
    dx_sum = plus_di + minus_di
    dx = np.where(dx_sum == 0, 0, 100 * np.abs(plus_di - minus_di) / dx_sum)
    
    adx = np.zeros_like(high_1d)
    for i in range(len(high_1d)):
        if i < 27:  # Need 14 for TR + 14 for DX smoothing
            adx[i] = np.nan
        else:
            adx[i] = np.nanmean(dx[i-13:i+1])  # Wilder's smoothing for ADX
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 4h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for Williams %R and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) and ADX > 25 (strong trend) with volume confirmation
            if (williams_r_aligned[i] < -80 and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 (overbought) and ADX > 25 (strong trend) with volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or ADX weakens below 20
            if williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or ADX weakens below 20
            if williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals