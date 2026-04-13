#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter
    # Long: price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND chop > 61.8 (range)
    # Short: price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND chop > 61.8 (range)
    # Exit: opposite Donchian breakout or chop < 38.2 (trend regime)
    # Using 1d for chop regime (structure) and 4h only for entry timing
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 20-50 trades/year (~80-200 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d chop regime (choppiness index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # align with index
    
    # ATR(14) for 1d
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Chop = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    chop_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(atr_1d[i-13:i+1])
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high > min_low and atr_sum > 0:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Align 1d chop to 4h (wait for completed 1d bar)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian(20) on 4h
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        vol_confirm = volume_spike[i]
        chop_high = chop_1d_aligned[i] > 61.8  # range regime
        chop_low = chop_1d_aligned[i] < 38.2   # trend regime
        
        # Entry logic: Donchian breakout + volume + chop regime (range)
        long_entry = (close[i] > donch_high[i]) and vol_confirm and chop_high
        short_entry = (close[i] < donch_low[i]) and vol_confirm and chop_high
        
        # Exit logic: opposite breakout or chop < 38.2 (trend regime)
        long_exit = (close[i] < donch_low[i]) or chop_low
        short_exit = (close[i] > donch_high[i]) or chop_low
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0