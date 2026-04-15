#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d Volume Spike + ADX Trend Filter
# Uses Donchian channel breakouts on 4h timeframe for trend following.
# Enters long when price breaks above 20-bar high, short when breaks below 20-bar low.
# Requires volume > 1.5x 20-bar median for confirmation.
# Uses 1d ADX > 25 to ensure we only trade in trending markets, avoiding chop.
# Designed to work in both bull and bear markets by following established trends.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ADX(14) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian Channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=1).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=1).min()
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(high_20.iloc[i]) or 
            np.isnan(low_20.iloc[i]) or 
            np.isnan(vol_threshold.iloc[i])):
            continue
        
        # Long: price breaks above 20-bar high, ADX > 25, volume spike
        if (close[i] > high_20.iloc[i] and 
            adx_1d_aligned[i] > 25 and 
            volume[i] > vol_threshold.iloc[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 20-bar low, ADX > 25, volume spike
        elif (close[i] < low_20.iloc[i] and 
              adx_1d_aligned[i] > 25 and 
              volume[i] > vol_threshold.iloc[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of channel or ADX weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < (high_20.iloc[i] + low_20.iloc[i]) / 2 or adx_1d_aligned[i] < 20)) or
               (signals[i-1] == -0.25 and (close[i] > (high_20.iloc[i] + low_20.iloc[i]) / 2 or adx_1d_aligned[i] < 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Volume_ADX"
timeframe = "4h"
leverage = 1.0