#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Choppiness Index > 61.8 = ranging market (mean revert at Donchian bounds)
# Choppiness Index < 38.2 = trending market (breakout in direction of trend)
# In ranging: long at lower band, short at upper band
# In trending: long on upper breakout, short on lower breakout
# Volume spike (>1.5x 20-period average) confirms the move
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_Chop_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr = np.full_like(close_1d, np.nan, dtype=float)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:15])  # first ATR
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.full_like(close_1d, np.nan, dtype=float)
    if len(atr) >= 28:  # need 14 + 14
        for i in range(27, len(atr_sum)):
            atr_sum[i] = np.nansum(atr[i-13:i+1])
    
    # Choppiness Index = 100 * log10(atr_sum / (highest_high - lowest_low)) / log10(14)
    highest_high = np.full_like(close_1d, np.nan, dtype=float)
    lowest_low = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(13, len(close_1d)):
        highest_high[i] = np.nanmax(high_1d[i-13:i+1])
        lowest_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    chop = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(27, len(close_1d)):
        if atr_sum[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h
    highest_high_20 = np.full_like(close, np.nan, dtype=float)
    lowest_low_20 = np.full_like(close, np.nan, dtype=float)
    for i in range(19, len(close)):
        highest_high_20[i] = np.max(high[i-19:i+1])
        lowest_low_20[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        donchian_upper = highest_high_20[i]
        donchian_lower = lowest_low_20[i]
        
        if position == 0:
            # Determine regime
            if chop_val > 61.8:  # ranging market
                # Mean reversion at Donchian bounds
                if close[i] <= donchian_lower and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_upper and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_val < 38.2:  # trending market
                # Breakout in direction of trend
                if close[i] > donchian_upper and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_lower and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # else: neutral zone (38.2-61.8), no action
                
        elif position == 1:
            # Long: exit on opposite Donchian touch or chop becomes extreme ranging
            if close[i] >= donchian_upper or (chop_val > 61.8 and close[i] >= (donchian_upper + donchian_lower) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on opposite Donchian touch or chop becomes extreme ranging
            if close[i] <= donchian_lower or (chop_val > 61.8 and close[i] <= (donchian_upper + donchian_lower) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals