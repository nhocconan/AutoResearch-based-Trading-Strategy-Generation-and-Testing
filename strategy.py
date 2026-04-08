#!/usr/bin/env python3
# 4h_12h_donchian_volume_chop_v1
# Hypothesis: 4h Donchian breakout with 12h volume confirmation and choppiness regime filter.
# Long: price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND CHOP(14) > 61.8 (ranging market)
# Short: price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND CHOP(14) > 61.8
# Exit: opposite Donchian breakout or CHOP < 38.2 (trending market)
# Designed to capture mean-reversion breaks in ranging markets while avoiding whipsaws in strong trends.
# Works in both bull and bear markets by fading breakouts in ranging conditions (CHOP > 61.8).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Choppiness Index (14-period) - requires HTF 12h for calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # True Range for 12h
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = df_12h['high'].values[0] - df_12h['low'].values[0]
    for i in range(1, len(df_12h)):
        tr_12h[i] = max(
            df_12h['high'].values[i] - df_12h['low'].values[i],
            abs(df_12h['high'].values[i] - df_12h['close'].values[i-1]),
            abs(df_12h['low'].values[i] - df_12h['close'].values[i-1])
        )
    
    # ATR (14-period) for 12h
    atr_12h = np.zeros(len(df_12h))
    atr_12h[:14] = np.nan
    for i in range(14, len(df_12h)):
        atr_12h[i] = np.mean(tr_12h[i-14:i])
    
    # Sum of ATR over 14 periods
    atr_sum_12h = np.zeros(len(df_12h))
    atr_sum_12h[:14] = np.nan
    for i in range(14, len(df_12h)):
        atr_sum_12h[i] = np.sum(atr_12h[i-14:i])
    
    # Max high and min low over 14 periods for 12h
    max_high_12h = np.zeros(len(df_12h))
    max_high_12h[:14] = np.nan
    min_low_12h = np.zeros(len(df_12h))
    min_low_12h[:14] = np.nan
    for i in range(14, len(df_12h)):
        max_high_12h[i] = np.max(df_12h['high'].values[i-14:i])
        min_low_12h[i] = np.min(df_12h['low'].values[i-14:i])
    
    # Choppiness Index: 100 * log10( sum(ATR) / (max_high - min_low) ) / log10(14)
    chop_12h = np.zeros(len(df_12h))
    for i in range(14, len(df_12h)):
        if max_high_12h[i] > min_low_12h[i] and atr_sum_12h[i] > 0:
            chop_12h[i] = 100 * np.log10(atr_sum_12h[i] / (max_high_12h[i] - min_low_12h[i])) / np.log10(14)
        else:
            chop_12h[i] = 50  # neutral when undefined
    
    # Align chop to 4h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_12h_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        chop = chop_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR market starts trending (CHOP < 38.2)
            if close[i] < donch_low[i] or chop < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR market starts trending (CHOP < 38.2)
            if close[i] > donch_high[i] or chop < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high AND volume confirmation AND ranging market (CHOP > 61.8)
            if close[i] > donch_high[i] and vol_ratio > 1.5 and chop > 61.8:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low AND volume confirmation AND ranging market (CHOP > 61.8)
            elif close[i] < donch_low[i] and vol_ratio > 1.5 and chop > 61.8:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals