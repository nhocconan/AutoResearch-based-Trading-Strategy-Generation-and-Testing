#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
# - Long: Price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + chop > 61.8 (range)
# - Short: Price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + chop > 61.8 (range)
# - Exit: Opposite Donchian breakout or chop < 38.2 (trend)
# - Uses 4h for entry timing, 1d for volume and chop confirmation
# - Designed for low trade frequency (<50/year) with high win rate in ranging markets
# - Works in both bull/bear by fading extremes in choppy regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d (using 14-period)
    atr_14 = []
    for i in range(len(high_1d)):
        if i < 14:
            atr_14.append(np.nan)
        else:
            tr = max(high_1d[i] - low_1d[i],
                      abs(high_1d[i] - close_1d[i-1]),
                      abs(low_1d[i] - close_1d[i-1]))
            atr_14.append(tr)
    atr_14 = np.array(atr_14)
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum_14 / (highest_high_14 - lowest_low_14)) / log10(14)
    chop = 100 * np.log10(atr_sum_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Align 1d indicators to 4h
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or \
           np.isnan(vol_ma_20_4h[i]) or np.isnan(chop_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        # Need to get current 1d volume - use the most recent complete 1d bar
        # Since we're on 4h, we approximate using the aligned volume MA
        vol_confirmed = volume_1d[min(i//16, len(volume_1d)-1)] > 1.5 * vol_ma_20[min(i//16, len(vol_ma_20)-1)] if i >= 16*20 else False
        
        # Chop regime: range-bound market (chop > 61.8)
        chop_range = chop_4h[i] > 61.8
        
        if position == 0:
            # Long entry: break above Donchian high + volume confirmed + chop range
            if close_4h[i] > highest_high_20[i] and vol_confirmed and chop_range:
                signals[i] = 0.25
                position = 1
            # Short entry: break below Donchian low + volume confirmed + chop range
            elif close_4h[i] < lowest_low_20[i] and vol_confirmed and chop_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low OR chop trend (chop < 38.2)
            if close_4h[i] < lowest_low_20[i] or chop_4h[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high OR chop trend (chop < 38.2)
            if close_4h[i] > highest_high_20[i] or chop_4h[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeChop"
timeframe = "4h"
leverage = 1.0