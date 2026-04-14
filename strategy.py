#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Confirmation and Chop Filter
# Uses Donchian(20) breakouts for trend capture, volume > 1.5x average for confirmation,
# and Choppiness Index > 61.8 to avoid choppy markets. Works in bull/bear by only
# taking breakouts with volume in trending regimes. Target: 25-35 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index (14) on 1d
    def calculate_choppiness(high_arr, low_arr, close_arr, window=14):
        atr = np.zeros_like(close_arr)
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        chop = np.full_like(close_arr, 50.0)
        valid = (atr > 0) & (hh > ll)
        chop[valid] = 100 * np.log10(np.sum(tr[-window+1:] if i >= window-1 else tr[:i+1]) / (window * atr[i]) / (hh[i] - ll[i])) / np.log10(window)
        # Fix: compute properly
        chop = np.full_like(close_arr, 50.0)
        for i in range(window, len(close_arr)):
            period_tr = np.nansum(tr[i-window+1:i+1])
            if atr[i] > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(period_tr / (window * atr[i]) / (hh[i] - ll[i])) / np.log10(window)
        return chop
    
    chop = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is not too choppy (CHOP <= 61.8 = trending)
        if chop_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume confirmation
            if close[i] > donch_high[i] and volume[i] > 1.5 * vol_ma[i]:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below Donchian low with volume confirmation
            elif close[i] < donch_low[i] and volume[i] > 1.5 * vol_ma[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0