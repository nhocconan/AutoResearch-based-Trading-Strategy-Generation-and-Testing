#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1
Hypothesis: Donchian(20) breakout on 4h with volume confirmation (>2x average) and choppiness regime filter (CHOP > 61.8 = range) to avoid false breakouts in sideways markets. Only long when price > upper band, only short when price < lower band. Uses discrete position sizing (0.25) and ATR-based stoploss (2.0x). Designed for low trade frequency (<30/year) to minimize fee drag. Works in both bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    chop_ratio = np.where(range_14 > 0, sum_atr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_filter = chop > 61.8  # Only trade in ranging markets (CHOP > 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), volume MA, ATR, CHOP
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
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band AND volume spike AND chop filter
            long_signal = (close_val > highest_high[i]) and vol_spike and chop_ok
            
            # Short: price breaks below lower Donchian band AND volume spike AND chop filter
            short_signal = (close_val < lowest_low[i]) and vol_spike and chop_ok
            
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
            # Exit: price hits ATR stoploss OR re-enters Donchian channel (mean reversion in range)
            if (close_val < entry_price - 2.0 * atr[i]) or (close_val < highest_high[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price hits ATR stoploss OR re-enters Donchian channel
            if (close_val > entry_price + 2.0 * atr[i]) or (close_val > lowest_low[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0