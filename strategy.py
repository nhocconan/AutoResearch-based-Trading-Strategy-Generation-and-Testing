#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1
Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
- Uses 4h timeframe for optimal trade frequency (target: 75-200 total trades over 4 years)
- Donchian(20) breakout captures trend continuation after consolidation
- Volume confirmation (1.5x average volume) ensures institutional participation
- Choppiness index filter: only trade when CHOP > 50 (avoid strong trends where breakouts fail)
- Works in bull/bear markets by trading breakouts in both directions with volume/regime confirmation
- Designed for 19-50 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for choppiness filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with 1d indices
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(TR(14)) / (HH(14) - LL(14))) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_denominator = hh_1d - ll_1d
    chop_denominator = np.where(chop_denominator == 0, np.nan, chop_denominator)
    chop_ratio = sum_tr_14 / chop_denominator
    chop_ratio = np.where((chop_ratio <= 0) | np.isnan(chop_ratio), np.nan, chop_ratio)
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align choppiness to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian(20) breakout levels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Only trade when market is choppy (avoid strong trends)
        choppy_market = chop_1d_aligned[i] > 50
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND choppy market
            if close[i] > period20_high[i] and volume_spike[i] and choppy_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND choppy market
            elif close[i] < period20_low[i] and volume_spike[i] and choppy_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR volume drops significantly
            if close[i] < period20_low[i] or volume[i] < (volume_ma[i] * 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR volume drops significantly
            if close[i] > period20_high[i] or volume[i] < (volume_ma[i] * 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0