#!/usr/bin/env python3
# 4h_donchian_1d_volume_chop_v1
# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Long: price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + CHOP(14) > 61.8 (range)
# Short: price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + CHOP(14) > 61.8 (range)
# Exit: Donchian(10) opposite breakout or volume < 1.2x average
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_chop_v1"
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
    
    # 1d HTF data for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Volume MA (20-period)
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high_n) - min(low_n))))
    tr1 = np.maximum(high_1d[1:] - low_1d[:-1], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr1 = np.concatenate([[np.nan], tr1])
    tr2 = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(np.maximum(tr1, tr2)).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop_denominator = np.log10(14) * (max_high_14 - min_low_14)
    chop = 100 * np.log10(atr_sum_14 / chop_denominator)
    
    # Align 1d indicators to 4h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(volume_ma_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d[i // 16] > 1.5 * volume_ma_1d_aligned[i] if i // 16 < len(volume_1d) else False
        
        # Choppiness filter: CHOP > 61.8 (range-bound market)
        chop_confirmed = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian(10) low OR volume drops
            if close[i] < donchian_low_10[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian(10) high OR volume drops
            if close[i] > donchian_high_10[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need both volume and chop confirmation
            if volume_confirmed and chop_confirmed:
                # Long: price breaks above Donchian(20) high
                if close[i] > donchian_high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian(20) low
                elif close[i] < donchian_low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals