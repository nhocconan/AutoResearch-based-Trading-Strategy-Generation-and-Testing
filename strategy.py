#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) with 1d volume confirmation and 1d ADX trend filter
# Long when: price breaks above 20-bar Donchian high + volume > 1.5x 20-bar median + ADX > 25
# Short when: price breaks below 20-bar Donchian low + volume > 1.5x 20-bar median + ADX > 25
# Exit when: price crosses back through 10-bar Donchian midpoint
# Designed to capture strong trends with volume confirmation and avoid choppy markets
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag on 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d ADX for trend strength (14-period)
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
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume confirmation: current > 1.5x median of last 20 bars
    vol_1d = df_1d['volume'].values
    vol_median = pd.Series(vol_1d).rolling(window=20, min_periods=1).median().values
    vol_threshold = 1.5 * vol_median
    
    # Align HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_threshold_aligned[i])):
            continue
        
        # Long: price breaks above Donchian high + volume spike + strong trend (ADX > 25)
        if (close[i] > donchian_high_aligned[i] and 
            volume[i] > vol_threshold_aligned[i] and 
            adx_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume spike + strong trend (ADX > 25)
        elif (close[i] < donchian_low_aligned[i] and 
              volume[i] > vol_threshold_aligned[i] and 
              adx_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price crosses back through 10-bar Donchian midpoint
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donchian_mid_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donchian_mid_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian20_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0