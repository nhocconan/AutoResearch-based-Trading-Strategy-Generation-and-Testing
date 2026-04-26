#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_v1
Hypothesis: Donchian(20) breakout with volume confirmation and choppiness regime filter on 4h. 
Long when price breaks above upper band + volume spike + chop>61.8 (range) for mean reversion long at support.
Short when price breaks below lower band + volume spike + chop<38.2 (trend) for trend-following short.
Uses discrete sizing (0.25) to minimize fee drag. Designed for low trade frequency (<50/year) to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index regime filter (14-period)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * (np.log10(sum_atr_14) - np.log10(range_14)) / np.log10(14)
    chop_regime_range = chop > 61.8  # ranging market
    chop_regime_trend = chop < 38.2   # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), volume MA, ATR, chop
    start_idx = max(20, 20, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        in_range = chop_regime_range[i]
        in_trend = chop_regime_trend[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band + volume spike + ranging regime (mean reversion long at resistance breakout)
            long_signal = (close_val > highest_high[i]) and vol_spike and in_range
            
            # Short: price breaks below lower Donchian band + volume spike + trending regime (trend-following short at support breakdown)
            short_signal = (close_val < lowest_low[i]) and vol_spike and in_trend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price hits ATR stoploss OR reverses to opposite Donchian band
            if (close_val < entry_price - 2.5 * atr[i]) or (close_val < lowest_low[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price hits ATR stoploss OR reverses to opposite Donchian band
            if (close_val > entry_price + 2.5 * atr[i]) or (close_val > highest_high[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0