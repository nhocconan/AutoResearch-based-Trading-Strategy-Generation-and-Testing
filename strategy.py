#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long: price breaks above Donchian upper band AND volume > 1.5x average AND chop < 61.8 (trending)
# Short: price breaks below Donchian lower band AND volume > 1.5x average AND chop < 61.8 (trending)
# Exit: price crosses Donchian middle band or chop > 61.8 (range) or volume < average
# Uses 1d HTF for volume average and chop calculation to reduce noise.
# Designed to capture strong trends in both bull and bear markets with minimal whipsaws.
# Discrete position sizing: 0.0 (flat), ±0.25 (position) to limit fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_middle = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
        donchian_middle[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # Get 1d HTF data for volume average and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_avg_1d[i] = np.mean(vol_1d[i-20:i])
    
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        atr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    chop = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(atr_1d[i-14:i])
        highest_high = np.max(high_1d[i-14:i])
        lowest_low = np.min(low_1d[i-14:i])
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(donchian_middle[i]) or np.isnan(vol_avg_1d_aligned[i]) or \
           np.isnan(chop_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_now = prices['volume'].values[i]  # Current 4h volume
        
        if position == 1:  # Long position
            # Exit conditions: price < middle OR chop > 61.8 (range) OR volume < average
            if price < donchian_middle[i] or chop_aligned[i] > 61.8 or vol_1d_now < vol_avg_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price > middle OR chop > 61.8 (range) OR volume < average
            if price > donchian_middle[i] or chop_aligned[i] > 61.8 or vol_1d_now < vol_avg_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout + volume confirmation + trending regime (chop < 61.8)
            if price > donchian_upper[i] and vol_1d_now > 1.5 * vol_avg_1d_aligned[i] and chop_aligned[i] < 61.8:
                position = 1
                signals[i] = 0.25
            elif price < donchian_lower[i] and vol_1d_now > 1.5 * vol_avg_1d_aligned[i] and chop_aligned[i] < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals