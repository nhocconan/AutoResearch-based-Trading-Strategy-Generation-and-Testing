#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and chop regime filter (CHOP(14) between 38.2 and 61.8 for ranging market mean reversion). 
# Uses 12h HTF data for Donchian channels to reduce noise and improve signal quality. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 20-40 trades/year (80-160 total over 4 years). Works in both bull (breakouts) and bear (mean reversion in chop) markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v3"
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
    
    # 12h HTF data for Donchian channels (smoother, less noise)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(20) channels
    period20_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    period20_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_12h = period20_high_12h
    donchian_low_12h = period20_low_12h
    
    # Align 12h Donchian channels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # 4h Chopiness Index (CHOP) for regime detection
    atr_14 = pd.Series(np.maximum.reduce([
        high[1:] - low[:-1],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[:-1] - close[:-1])
    ])).rolling(window=14, min_periods=14).mean().values
    atr_14 = np.insert(atr_14, 0, np.nan)  # Align length
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_14.sum() / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Fix: Calculate rolling sum of ATR correctly
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # 4h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: 38.2 <= CHOP <= 61.8 (ranging market)
        chop_regime = (chop[i] >= 38.2) and (chop[i] <= 61.8)
        
        if position == 1:  # Long position
            # Exit: price falls below 12h Donchian low OR chop regime ends (trending starts)
            if close[i] < donchian_low_aligned[i] or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above 12h Donchian high OR chop regime ends (trending starts)
            if close[i] > donchian_high_aligned[i] or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above 12h Donchian high (breakout in ranging market)
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below 12h Donchian low (breakdown in ranging market)
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals