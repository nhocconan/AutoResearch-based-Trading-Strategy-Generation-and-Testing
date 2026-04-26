#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter_v1
Hypothesis: Donchian(20) breakouts on 4h with 1d EMA50 trend filter, volume spike confirmation, and chop regime filter.
Donchian channels provide objective trend-following structure. Breakouts above/below 20-period high/low
with volume expansion and 1d trend alignment capture sustained moves. Chop filter avoids whipsaws in ranging markets.
Targets 20-40 trades/year (80-160 over 4 years) for low fee drag and robust test performance.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian(20) channels on 4h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    # Choppiness Index filter to avoid ranging markets
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    chop = np.zeros_like(close)
    mask = (range_14 > 0) & (sum_atr_14 > 0)
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    chop_filter = chop < 45  # Only trade when trending (CHOP < 45)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(chop[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long: price breaks above Donchian(20) high with volume spike + uptrend + trending market
        if (close[i] > highest_high_20[i] and 
            volume_spike[i] and 
            uptrend and 
            chop_filter[i]):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short: price breaks below Donchian(20) low with volume spike + downtrend + trending market
        elif (close[i] < lowest_low_20[i] and 
              volume_spike[i] and 
              downtrend and 
              chop_filter[i]):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit: price returns to opposite Donchian level or trend weakens or market becomes choppy
        elif position == 1 and (close[i] < lowest_low_20[i] or not uptrend or not chop_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > highest_high_20[i] or not downtrend or not chop_filter[i]):
            signals[i] = 0.0
            position = 0
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0