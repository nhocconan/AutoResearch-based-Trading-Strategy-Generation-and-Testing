#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for chop regime filter (chop > 61.8 = range).
- Entry: Price breaks above Donchian(20) high (long) or below Donchian(20) low (short) on 4h close,
         with volume > 1.5x 20-period volume MA AND chop regime > 61.8 (range market).
- Direction: In chop regime (>61.8), we mean-revert: long on lower band break, short on upper band break.
- Chop regime filter avoids trending markets where breakouts fail, focuses on ranging markets.
- Exit: Price returns to Donchian(20) midpoint or chop regime shifts to trending (< 38.2).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying dips in range, in bear via selling rallies in range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d chop regime (EHLERS CHOPPINESS INDEX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])  # first bar
    
    # ATR(14) and highest high/lowest low over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    denominator = hh_14 - ll_14
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(sum_atr_14 / denominator) / np.log10(14)
    
    # Align chop regime to 4h timeframe (completed 1d bar only)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) channels on 4h
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2  # midpoint
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20) + 1  # Donchian(20), chop(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        in_range = chop_val > 61.8  # chop > 61.8 = ranging market
        
        if position == 0:
            if in_range:
                # In range: mean reversion - long at lower band, short at upper band
                if (close[i] <= donchian_l[i] and volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] >= donchian_h[i] and volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
            # Optional: in trending markets (chop < 38.2), could add breakout logic
            # but we focus only on ranging for now to keep trades low and win rate high
        elif position == 1:
            # Long exit: price returns to midpoint or chop shifts to trending
            if (close[i] >= donchian_m[i]) or (chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint or chop shifts to trending
            if (close[i] <= donchian_m[i]) or (chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ChopRange_MeanReversion_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0