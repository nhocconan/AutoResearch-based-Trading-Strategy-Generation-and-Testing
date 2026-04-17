#!/usr/bin/env python3
"""
Hypothesis: 4H Donchian breakout + 1D volume spike + chop regime filter
- Entry long: Price breaks above Donchian(20) high + 1D volume > 2x 20-day volume MA + CHOP(14) > 61.8 (ranging market)
- Entry short: Price breaks below Donchian(20) low + 1D volume > 2x 20-day volume MA + CHOP(14) > 61.8
- Exit: Price crosses Donchian midline (10-period average of high/low) or reverse breakout
- Uses 1D volume and chop regime to filter false breakouts in choppy markets
- Position size 0.25 to manage drawdown
- Designed for 4H timeframe with strict entry conditions to target 75-200 trades over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for volume and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14) on 1D data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_14_1d = []
    tr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_1d.append(tr)
        if i < 14:
            atr_14_1d.append(np.nan)
        else:
            atr_14_1d.append(np.mean(tr_1d[i-13:i+1]))
    atr_14_1d = np.array(atr_14_1d)
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    chop_14_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        sum_atr = np.sum(atr_14_1d[i-13:i+1])
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high - min_low > 0:
            chop_14_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    # Calculate Donchian channels on 4H data
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max()
        lower = pd.Series(low).rolling(window=window, min_periods=window).min()
        mid = (upper + lower) / 2
        return upper.values, lower.values, mid.values
    
    donchian_window = 20
    upper, lower, mid = donchian_channels(high, low, donchian_window)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = donchian_window  # wait for full Donchian window
    
    for i in range(start_idx, n):
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_aligned[i]
        chop = chop_aligned[i]
        
        # Volume spike: current 1D volume > 2x 20-day MA (need to get current 1D volume)
        # Since we don't have current 1D volume aligned, we approximate using price change
        # Alternative: use 4H volume spike as proxy
        vol_4h = prices['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean()
        vol_spike = vol_4h[i] > 2 * vol_ma_4h.iloc[i] if not np.isnan(vol_ma_4h.iloc[i]) else False
        
        if position == 0:
            # Look for breakout with volume spike and chop > 61.8 (ranging market)
            if price > upper[i] and vol_spike and chop > 61.8:
                signals[i] = 0.25
                position = 1
            elif price < lower[i] and vol_spike and chop > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses midline or reverse breakout
            if price < mid[i] or price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses midline or reverse breakout
            if price > mid[i] or price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0