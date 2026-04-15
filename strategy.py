#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Channel Breakout with Volume Spike and ADX Trend Filter
# Uses 20-period Donchian channels (highest high/lowest low) as breakout levels.
# Long when price breaks above upper band with volume > 2x median volume and ADX > 25.
# Short when price breaks below lower band with volume > 2x median volume and ADX > 25.
# Works in both bull and bear markets by trading breakouts in direction of trend.
# Includes ATR-based stoploss to limit drawdowns.
# Target: 20-50 trades/year (80-200 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on daily
    # True Range
    tr1 = high_1d - low_1d
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
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr_4h[i])):
            continue
        
        # Long entry: price breaks above Donchian upper band + volume spike + ADX > 25
        if (close[i] > highest_high[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower band + volume spike + ADX > 25
        elif (close[i] < lowest_low[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ATR-based stoploss
        elif position == 1:
            # Stoploss: 2 * ATR below entry price (tracked via highest high since entry)
            if close[i] < highest_high[i] - 2.0 * atr_4h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Stoploss: 2 * ATR above entry price (tracked via lowest low since entry)
            if close[i] > lowest_low[i] + 2.0 * atr_4h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_VolumeSpike_ADX_Stop"
timeframe = "4h"
leverage = 1.0