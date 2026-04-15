#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volume Confirmation and ADX Trend Filter
# Combines price breakout (Donchian channel) with volume confirmation and trend strength (ADX).
# Long when price breaks above upper Donchian(20) with volume > 2x 20-day average and ADX > 25.
# Short when price breaks below lower Donchian(20) with volume > 2x 20-day average and ADX > 25.
# Works in bull markets (breakouts) and bear markets (breakdowns) by filtering with trend strength.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # ADX(14) for trend strength on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    dm_plus_14_safe = np.where(tr_14 == 0, 1, dm_plus_14)
    dm_minus_14_safe = np.where(tr_14 == 0, 1, dm_minus_14)
    tr_14_safe = np.where(tr_14 == 0, 1, tr_14)
    
    di_plus = 100 * dm_plus_14_safe / tr_14_safe
    di_minus = 100 * dm_minus_14_safe / tr_14_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian Channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Volume threshold: 2x 1-day average volume
        vol_threshold = 2 * vol_avg_1d_aligned[i]
        
        # Long: Price breaks above upper Donchian, volume spike, ADX > 25
        if (close[i] > highest_high[i] and 
            volume[i] > vol_threshold and 
            adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: Price breaks below lower Donchian, volume spike, ADX > 25
        elif (close[i] < lowest_low[i] and 
              volume[i] > vol_threshold and 
              adx_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: Price returns to middle of Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (highest_high[i] + lowest_low[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (highest_high[i] + lowest_low[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Volume_ADX"
timeframe = "4h"
leverage = 1.0