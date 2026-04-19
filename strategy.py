#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsVixFix_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Vix Fix (WVF) on 1d data
    # WVF = ((Highest Close in period - Low) / Highest Close in period) * 100
    wvf_period = 22
    highest_close = np.full_like(close_1d, np.nan)
    for i in range(wvf_period - 1, len(close_1d)):
        highest_close[i] = np.max(close_1d[i - wvf_period + 1:i + 1])
    
    wvf = np.full_like(close_1d, np.nan)
    for i in range(wvf_period - 1, len(close_1d)):
        if highest_close[i] != 0:
            wvf[i] = ((highest_close[i] - low_1d[i]) / highest_close[i]) * 100
    
    # WVF mean and standard deviation for z-score
    wvf_mean = np.full_like(close_1d, np.nan)
    wvf_std = np.full_like(close_1d, np.nan)
    for i in range(wvf_period - 1, len(close_1d)):
        if i >= 2 * wvf_period - 2:  # Need enough data for std
            window = wvf[i - wvf_period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                wvf_mean[i] = np.mean(valid)
                wvf_std[i] = np.std(valid) if len(valid) > 1 else 0
    
    # WVF z-score
    wvf_z = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(wvf[i]) and not np.isnan(wvf_mean[i]) and wvf_std[i] > 0:
            wvf_z[i] = (wvf[i] - wvf_mean[i]) / wvf_std[i]
    
    # Align WVF z-score to 6h timeframe
    wvf_z_aligned = align_htf_to_ltf(prices, df_1d, wvf_z)
    
    # 6h Donchian channel breakout
    donchian_period = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(donchian_period - 1, len(high)):
        highest_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        lowest_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        if np.isnan(wvf_z_aligned[i]) or np.isnan(highest_high[i]) or \
           np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wvf_z_val = wvf_z_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: WVF spike (fear) + price breaks above Donchian high
            if wvf_z_val > 2.0 and price > highest_high[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: WVF spike (fear) + price breaks below Donchian low
            elif wvf_z_val > 2.0 and price < lowest_low[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: WVF returns to normal or price retrace to midpoint
            if wvf_z_val < 0.5 or price < (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: WVF returns to normal or price retrace to midpoint
            if wvf_z_val < 0.5 or price > (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals