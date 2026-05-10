#!/usr/bin/env python3
# 1D_WeeklyDonchian_Breakout_Volume
# Hypothesis: Weekly Donchian channel breakouts with volume confirmation capture strong trends across bull and bear markets.
# The weekly timeframe provides robust trend context, reducing false signals in noisy daily data.
# Volume confirmation ensures breakouts are supported by participation, increasing reliability.
# Designed for low trade frequency (7-25/year) to minimize fee drag and improve generalization.

name = "1D_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channel (20-period)
    donchian_len = 20
    high_roll = pd.Series(high_1w).rolling(window=donchian_len, min_periods=donchian_len).max()
    low_roll = pd.Series(low_1w).rolling(window=donchian_len, min_periods=donchian_len).min()
    upper_1w = high_roll.values
    lower_1w = low_roll.values
    
    # Align weekly Donchian levels to daily timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Weekly average volume for confirmation
    vol_avg_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or \
           np.isnan(vol_avg_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_avg_1w_aligned[i] if vol_avg_1w_aligned[i] > 0 else False
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian upper with volume
            if close[i] > upper_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian lower with volume
            elif close[i] < lower_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly Donchian lower
            if close[i] < lower_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly Donchian upper
            if close[i] > upper_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals