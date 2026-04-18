#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA(34) trend filter.
# Long when price breaks above 4h Donchian upper band with volume > 1.5x 20-period average and price > 1d EMA(34).
# Short when price breaks below 4h Donchian lower band with volume > 1.5x 20-period average and price < 1d EMA(34).
# Exit when price returns to the 4h Donchian middle band (20-period SMA).
# Uses Donchian channel for structure, volume surge for conviction, daily EMA for trend filter.
# Designed for ~20-50 trades/year per symbol.
name = "4h_Donchian_20_Volume_EMA34_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 4h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = ((high_max + low_min) / 2.0)
    
    # EMA(34) on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(middle[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = high_max[i]
        lower = low_min[i]
        mid = middle[i]
        ema_val = ema_34_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume surge and above EMA
            if close_val > upper and vol_filter and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume surge and below EMA
            elif close_val < lower and vol_filter and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band
            if close_val <= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band
            if close_val >= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals