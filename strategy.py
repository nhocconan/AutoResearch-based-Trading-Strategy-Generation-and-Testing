#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d/1w Donchian breakout with volume confirmation and ADX trend filter
# Uses weekly Donchian channels (20-period high/low) as support/resistance.
# Breakouts above weekly high or below weekly low are traded when confirmed by volume and ADX > 25.
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Timeframe: 1d, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian channels to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Load weekly data for ADX trend filter
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w_adx = df_1w['high'].values
    low_1w_adx = df_1w['low'].values
    close_1w_adx = df_1w['close'].values
    
    # Calculate ADX (14-period) on weekly
    tr1 = high_1w_adx - low_1w_adx
    tr2 = np.abs(high_1w_adx - np.roll(close_1w_adx, 1))
    tr3 = np.abs(low_1w_adx - np.roll(close_1w_adx, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1w_adx - np.roll(high_1w_adx, 1)) > (np.roll(low_1w_adx, 1) - low_1w_adx), 
                       np.maximum(high_1w_adx - np.roll(close_1w_adx, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w_adx, 1) - low_1w_adx) > (high_1w_adx - np.roll(close_1w_adx, 1)), 
                        np.maximum(np.roll(close_1w_adx, 1) - low_1w_adx, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly Donchian high + volume confirmation + ADX > 25
        if (close[i] > high_20_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly Donchian low + volume confirmation + ADX > 25
        elif (close[i] < low_20_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < low_20_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_20_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0