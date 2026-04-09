#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1d Donchian breakout with 1w volume confirmation
# Donchian(20) breakout on 1d provides clear trend signals
# 1w volume spike confirms institutional participation
# In trending markets: follow 1d Donchian breakout direction
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakouts capture strong moves in both directions

name = "12h_1d_1w_donchian_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_20 = rolling_max(high_1d, 20)
    lower_20 = rolling_min(low_1d, 20)
    
    # Calculate 1d breakout signals
    breakout_up = close_1d > upper_20  # Close above upper Donchian
    breakout_down = close_1d < lower_20  # Close below lower Donchian
    
    # Load 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w volume ratio (current vs 20-period average)
    def rolling_mean(arr, window):
        res = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.mean(arr[i-window+1:i+1])
        return res
    
    vol_ma_20 = rolling_mean(volume_1w, 20)
    volume_ratio = np.where(vol_ma_20 > 0, volume_1w / vol_ma_20, 1.0)
    
    # Volume confirmation: significant spike (>1.5x average)
    volume_confirm = volume_ratio > 1.5
    
    # Align 1d indicators to 12h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down.astype(float))
    
    # Align 1w volume confirmation to 12h timeframe
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long on downside breakout or loss of volume confirmation
            if breakout_down_aligned[i] > 0.5 or volume_confirm_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short on upside breakout or loss of volume confirmation
            if breakout_up_aligned[i] > 0.5 or volume_confirm_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on upside breakout with volume confirmation
            if breakout_up_aligned[i] > 0.5 and volume_confirm_aligned[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Enter short on downside breakout with volume confirmation
            elif breakout_down_aligned[i] > 0.5 and volume_confirm_aligned[i] > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals