#!/usr/bin/env python3
"""
4h Donchian(20) breakout + volume spike + choppiness regime filter
Hypothesis: Donchian breakouts capture momentum, volume confirms institutional participation,
and choppiness filter avoids whipsaws in ranging markets. Works in both bull (long breakouts)
and bear (short breakouts) by using symmetric entry conditions.
Designed for 4h timeframe with tight entry conditions to achieve 20-50 trades/year.
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
    
    # Get 1d data for choppiness regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for choppiness
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True range for current period
    tr_current = np.maximum(high_1d - low_1d,
                           np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                    np.abs(low_1d - np.roll(close_1d, 1))))
    tr_current[0] = np.nan
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(tr_current).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(atr)/log10(hh-ll)) / log10(14)
    chop_raw = 100 * np.log10(sum_atr_14) / np.log10(14) / np.log10(hh_14 - ll_14)
    chop_raw = np.where((hh_14 - ll_14) > 0, chop_raw, 50)  # avoid division by zero
    
    # Align choppiness to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate Donchian channels on 4h (20-period)
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = max(donchian_window, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        chop_value = chop_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND chop < 61.8 (trending)
            long_entry = (curr_high > dc_high[i]) and vol_spike and (chop_value < 61.8)
            # Short: price breaks below Donchian low AND volume spike AND chop < 61.8 (trending)
            short_entry = (curr_low < dc_low[i]) and vol_spike and (chop_value < 61.8)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Donchian low OR chop > 61.8 (ranging)
            if (curr_low < dc_low[i]) or (chop_value > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian high OR chop > 61.8 (ranging)
            if (curr_high > dc_high[i]) or (chop_value > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0