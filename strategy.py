#!/usr/bin/env python3
# 4h_donchian_volume_chop_v5
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + CHOP > 61.8 (ranging market = mean reversion setup).
# Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + CHOP > 61.8.
# Uses 1d HTF for choppiness regime to avoid trading against strong daily trends.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-40 trades/year.
# Works in both bull and bear markets by fading breaks in ranging regimes (CHOP > 61.8).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_v5"
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
    
    # 1d HTF data for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high_14 = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    
    # Align 1d choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian(20)
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(max_high_20[i]) or
            np.isnan(min_low_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        if chop_aligned[i] <= 61.8:
            # In trending markets, stay flat to avoid whipsaws
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR volume drops
            if close[i] < min_low_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR volume drops
            if close[i] > max_high_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian high
                if close[i] > max_high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low
                elif close[i] < min_low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals