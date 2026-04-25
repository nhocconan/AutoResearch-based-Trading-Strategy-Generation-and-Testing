#!/usr/bin/env python3
"""
6h Williams Alligator + Chop Regime + Volume Spike
Hypothesis: The Williams Alligator (jaw/teeth/lips) identifies trends; when aligned (jaws down/teeth up/lips up for long, inverse for short) with low choppiness and volume confirmation, it captures sustained moves in both bull and bear markets. Chop regime filter avoids false signals in ranging markets. Discrete sizing (0.25) controls drawdown.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator from 1d: SMA of median price (hlc3)
    hlc3 = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    jaw = pd.Series(hlc3).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shift 8
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shift 5
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shift 3
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Chop regime filter: CHOP(14) > 61.8 = range (avoid), < 38.2 = trend (favor)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        hhll = highest_high - lowest_low
        
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(atr_sum / np.log(10) / hhll)
        return chop
    
    chop_values = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator warmup and volume MA
    start_idx = max(50, 21)  # Alligator needs ~21+8=29, vol MA 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        # Alligator alignment: jaw < teeth < lips for long, jaw > teeth > lips for short
        bullish_align = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_align = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + volume spike + not choppy
            long_entry = bullish_align and vol_spike and not_choppy
            short_entry = bearish_align and vol_spike and not_choppy
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator alignment breaks (jaw > teeth) OR choppy regime
            if (jaw_aligned[i] >= teeth_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator alignment breaks (jaw < teeth) OR choppy regime
            if (jaw_aligned[i] <= teeth_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ChopRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0