#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data
    weekly_pivot = (high_1d + low_1d + close_1d) / 3.0
    weekly_range = high_1d - low_1d
    r1 = 2 * weekly_pivot - low_1d
    s1 = 2 * weekly_pivot - high_1d
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get 6h data for Donchian channel (price channel)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # 6h Donchian(20) channel
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    donch_high_20 = high_6h_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_6h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_6h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_6h, donch_low_20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # ATR(20) for volatility filter and stop
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 6h Donchian high with volume and above weekly pivot
            if close[i] > donch_high_20_aligned[i] and volume_filter[i] and close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian low with volume and below weekly pivot
            elif close[i] < donch_low_20_aligned[i] and volume_filter[i] and close[i] < weekly_pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 6h Donchian low OR ATR-based stop
            if close[i] < donch_low_20_aligned[i] or close[i] < (high[max(0, i-1)] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 6h Donchian high OR ATR-based stop
            if close[i] > donch_high_20_aligned[i] or close[i] > (low[max(0, i-1)] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Volume"
timeframe = "6h"
leverage = 1.0