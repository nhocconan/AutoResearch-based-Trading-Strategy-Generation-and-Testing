#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d Volume Spike + 1d ADX Trend Filter
# Long when price breaks above 12h Donchian upper band (20-period high), volume > 2x 20-bar median, and 1d ADX > 25 (trending).
# Short when price breaks below 12h Donchian lower band (20-period low), volume > 2x 20-bar median, and 1d ADX > 25.
# Uses 1d ADX to filter for trending markets, avoiding whipsaws in ranging conditions.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.
# Designed to work in both bull and bear markets by following established trends with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ADX(14) for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
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
    tr_rolled = pd.Series(tr).rolling(window=14, min_periods=1).sum()
    dm_plus_rolled = pd.Series(dm_plus).rolling(window=14, min_periods=1).sum()
    dm_minus_rolled = pd.Series(dm_minus).rolling(window=14, min_periods=1).sum()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_rolled / (tr_rolled + 1e-10)
    di_minus = 100 * dm_minus_rolled / (tr_rolled + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=1).mean()
    adx_1d = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=1).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=1).min()
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(lookback, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(adx_1d_aligned[i])):
            continue
        
        # Long: Price breaks above Donchian upper band, volume spike, ADX > 25 (trending)
        if (close[i] > highest_high[i-1] and 
            volume[i] > vol_threshold[i] and 
            adx_1d_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian lower band, volume spike, ADX > 25 (trending)
        elif (close[i] < lowest_low[i-1] and 
              volume[i] > vol_threshold[i] and 
              adx_1d_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: Price returns to middle of Donchian channel or ADX weakens (< 20)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < (highest_high[i-1] + lowest_low[i-1]) / 2 or adx_1d_aligned[i] < 20)) or
               (signals[i-1] == -0.25 and (close[i] > (highest_high[i-1] + lowest_low[i-1]) / 2 or adx_1d_aligned[i] < 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Volume_ADX"
timeframe = "12h"
leverage = 1.0