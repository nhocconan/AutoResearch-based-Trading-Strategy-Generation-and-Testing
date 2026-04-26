#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_RegimeFilter_v2
Hypothesis: Donchian(20) breakouts on 4h timeframe with volume confirmation and choppiness regime filter capture strong momentum moves in both bull and bear markets. Volume spike confirms breakout validity. Choppiness index regime filter ensures we trade in trending markets (CHOP < 38.2) and avoid whipsaws in ranging markets. Position size: 0.25. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF regime filter (Choppiness Index)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_period = 14
    tr_series = pd.Series(tr)
    atr = tr_series.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Sum of ATR over period
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR) / (HH - LL)) / log10(N)
    # Avoid division by zero and log of zero/negative
    hl_range = highest_high - lowest_low
    chop_raw = np.where(
        (hl_range > 0) & (sum_atr > 0) & ~np.isnan(hl_range) & ~np.isnan(sum_atr),
        100 * np.log10(sum_atr / hl_range) / np.log10(14),
        50  # neutral value when invalid
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Volume spike detection on 4h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ema[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: trade only in trending markets (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        # Donchian(20) breakout levels
        lookback = 20
        if i >= lookback:
            highest_high_20 = np.max(high[i-lookback:i])
            lowest_low_20 = np.min(low[i-lookback:i])
        else:
            highest_high_20 = np.max(high[:i]) if i > 0 else high[i]
            lowest_low_20 = np.min(low[:i]) if i > 0 else low[i]
        
        # Long logic: price breaks above Donchian high with volume spike + in trending regime
        if close[i] > highest_high_20 and volume_spike[i] and is_trending:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below Donchian low with volume spike + in trending regime
        elif close[i] < lowest_low_20 and volume_spike[i] and is_trending:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Donchian level or regime changes to ranging
        elif position == 1 and (close[i] < lowest_low_20 or not is_trending):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > highest_high_20 or not is_trending):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_RegimeFilter_v2"
timeframe = "4h"
leverage = 1.0