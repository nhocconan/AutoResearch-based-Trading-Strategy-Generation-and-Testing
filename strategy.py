#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1-day ADX > 25 trend filter + volume > 1.5x 20-period average.
# Donchian breakouts capture breakouts in both bull and bear markets. ADX filter ensures
# trades are taken only in trending markets to avoid whipsaws in ranges. Volume confirms
# breakout strength. Designed for low trade frequency (~20-30 trades/year) to minimize
# fee drag and improve generalization.
name = "4h_Donchian_ADX_Volume_v2"
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
    
    # Donchian channels (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX (14-period) from 1-day timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d),
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    minus_dm = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])),
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = 100 * (np.zeros_like(plus_dm))
    minus_di = 100 * (np.zeros_like(minus_dm))
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    for i in range(len(plus_dm)):
        if i == 0:
            plus_dm_smooth[i] = plus_dm[i]
            minus_dm_smooth[i] = minus_dm[i]
        else:
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    dx = np.zeros_like(atr)
    di_sum = plus_dm_smooth + minus_dm_smooth
    mask = di_sum != 0
    dx[mask] = 100 * np.abs(plus_dm_smooth[mask] - minus_dm_smooth[mask]) / di_sum[mask]
    
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < 13:
            adx[i] = dx[i]
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high + ADX > 25 + volume filter
            if (close[i] > donchian_high[i] and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + ADX > 25 + volume filter
            elif (close[i] < donchian_low[i] and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low OR ADX drops below 20
            if (close[i] < donchian_low[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high OR ADX drops below 20
            if (close[i] > donchian_high[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals