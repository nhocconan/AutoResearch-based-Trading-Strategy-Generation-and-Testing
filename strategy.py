#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ATR Volatility Filter + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Filtering by 1d ATR (low volatility regime) reduces false breakouts in choppy markets. 
Volume spike confirms institutional participation. Works in bull/bear via volatility filtering.
Target: 20-50 trades/year (75-200 over 4 years).
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
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian(20) channels
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian calculation
    start_idx = donchian_window
    
    for i in range(start_idx, n):
        # Skip if ATR not ready
        if np.isnan(atr_14_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_value = atr_14_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Volatility filter: only trade when 1d ATR is below its 50-period median (low vol regime)
        if i >= 50:
            atr_ma_50 = np.mean(atr_14_aligned[i-49:i+1])
        else:
            atr_ma_50 = np.mean(atr_14_aligned[:i+1])
        low_volatility = atr_value < atr_ma_50  # trade in low volatility regimes
        
        # Breakout signals
        if position == 0:
            # Long: price breaks above upper Donchian channel with volume spike and low volatility
            long_condition = (curr_close > upper_channel) and volume_spike and low_volatility
            # Short: price breaks below lower Donchian channel with volume spike and low volatility
            short_condition = (curr_close < lower_channel) and volume_spike and low_volatility
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of channel or volatility increases significantly
            midpoint = (upper_channel + lower_channel) / 2.0
            high_volatility = atr_value > (atr_ma_50 * 1.5)  # exit if volatility spikes
            if curr_close <= midpoint or high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of channel or volatility increases significantly
            midpoint = (upper_channel + lower_channel) / 2.0
            high_volatility = atr_value > (atr_ma_50 * 1.5)  # exit if volatility spikes
            if curr_close >= midpoint or high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0