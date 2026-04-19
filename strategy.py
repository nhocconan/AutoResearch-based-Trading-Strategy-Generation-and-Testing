#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and 1-day ADX trend filter.
# Donchian breakouts capture momentum; volume confirms strength; ADX>25 filters for trending markets.
# Designed for 12h timeframe to capture medium-term trends with low frequency (<30 trades/year).
# Entry: Long when price breaks above upper Donchian(20) with volume spike and ADX>25.
# Short when price breaks below lower Donchian(20) with volume spike and ADX>25.
# Exit: Opposite Donchian touch or ADX<20 (trend weakening).
# Uses strict conditions to limit trades and avoid overtrading.

name = "12h_Donchian_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    def smooth(val, period):
        smoothed = np.zeros_like(val)
        smoothed[period-1] = np.mean(val[:period])
        for i in range(period, len(val)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + val[i]
        return smoothed
    
    atr = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) != 0, dx, 0)
    
    # ADX
    adx = smooth(dx, 14)
    # First 27 values will be invalid due to smoothing
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 12h data
    lookback = 20
    upper_dc = np.full(n, np.nan)
    lower_dc = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper_dc[i] = np.max(high[i-lookback+1:i+1])
        lower_dc[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        volume_ma[i] = np.mean(volume[i-20+1:i+1])
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20) + 27  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian with volume and strong trend
            if (close[i] > upper_dc[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and strong trend
            elif (close[i] < lower_dc[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower Donchian or trend weakens
            if (close[i] < lower_dc[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper Donchian or trend weakens
            if (close[i] > upper_dc[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals