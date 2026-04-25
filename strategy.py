#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + Chop Regime Filter
Hypothesis: Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trends; 
long when price > LIPS and alligator aligned bullish, short when price < JAW and bearish.
Volume spike confirms momentum, chop regime filter avoids ranging markets.
Designed for 12h timeframe targeting 12-37 trades/year. Works in bull/bear via trend following.
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
    
    # Get 1d data for Williams Alligator and chop calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: smoothed medians
    # JAW (Blue): 13-period SMMA smoothed 8 bars ahead
    # TEETH (Red): 8-period SMMA smoothed 5 bars ahead  
    # LIPS (Green): 5-period SMMA smoothed 3 bars ahead
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.convolve(arr, np.ones(period)/period, mode='valid')
        result[period-1:len(arr)-1] = sma[:-1]  # align properly
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(median_1d, 13)
    teeth_raw = smma(median_1d, 8)
    lips_raw = smma(median_1d, 5)
    
    # Apply smoothing offsets
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Chop regime filter: avoid ranging markets
    def true_range(h, l, pc):
        tr1 = h - l
        tr2 = np.abs(h - pc)
        tr3 = np.abs(l - pc)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_close[0] = df_1d['close'].values[0]
    tr = true_range(df_1d['high'].values, df_1d['low'].values, prev_close)
    atr1 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = log10(sum(TR14)/(HHV14-LLV14)) * 100
    hhvl = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    llvl = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_raw = np.where((hhvl - llvl) > 0, 
                        np.log10(pd.Series(tr).rolling(window=14, min_periods=14).sum().values / (hhvl - llvl)) * 100, 
                        50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Jaw > Teeth > Lips
        bullish_align = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_align = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        if position == 0:
            # Look for entry signals
            # Long: price > Lips AND bullish alignment AND volume spike AND chop < 61.8 (trending)
            long_entry = (curr_close > lips_val) and bullish_align and vol_spike and (chop_val < 61.8)
            # Short: price < Jaw AND bearish alignment AND volume spike AND chop < 61.8 (trending)
            short_entry = (curr_close < jaw_val) and bearish_align and vol_spike and (chop_val < 61.8)
            
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
            # Exit: price crosses below Teeth OR chop > 61.8 (ranging) OR Alligator loses bullish alignment
            if (curr_close < teeth_val) or (chop_val > 61.8) or not bullish_align:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Teeth OR chop > 61.8 (ranging) OR Alligator loses bearish alignment
            if (curr_close > teeth_val) or (chop_val > 61.8) or not bearish_align:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0