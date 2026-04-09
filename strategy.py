#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# Volume spike confirms breakout authenticity; chop filter avoids whipsaws in ranging markets
# Works in bull/bear: Alligator adapts to trend, volume confirms momentum, chop filter prevents false signals
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_1d_williams_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams Alligator and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift the lines (Jaw: 8, Teeth: 5, Lips: 3)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (wait for 1d bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate chopiness index on 1d timeframe (14-period)
    def true_range(high_arr, low_arr, close_arr):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_arr[0] - low_arr[0]  # First TR is just high-low
        return tr
    
    atr_1d = true_range(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    atr_ma_14 = np.full(len(atr_1d), np.nan)
    for i in range(len(atr_1d)):
        if i < 13:
            atr_ma_14[i] = np.nan
        else:
            atr_ma_14[i] = np.mean(atr_1d[i-13:i+1])
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = np.full(len(df_1d), np.nan)
    ll_14 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 13:
            hh_14[i] = np.nan
            ll_14[i] = np.nan
        else:
            hh_14[i] = np.max(df_1d['high'].values[i-13:i+1])
            ll_14[i] = np.min(df_1d['low'].values[i-13:i+1])
    
    # Chopiness Index = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 13 or np.isnan(hh_14[i]) or np.isnan(ll_14[i]) or hh_14[i] == ll_14[i]:
            chop_1d[i] = np.nan
        else:
            sum_atr = np.sum(atr_1d[i-13:i+1])
            chop_1d[i] = 100 * np.log10(sum_atr / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period average volume on 1d for volume confirmation
    avg_volume_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 19:
            avg_volume_1d[i] = np.nan
        else:
            avg_volume_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    
    # Align average volume to 12h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average (1d)
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Chop filter: avoid ranging markets (chop > 61.8) and extreme trending (chop < 38.2)
        # We want moderate trending: 38.2 <= chop <= 61.8
        chop_filter = (chop_aligned[i] >= 38.2) and (chop_aligned[i] <= 61.8)
        
        if position == 1:  # Long position
            # Exit: Alligator lines cross (lips < teeth < jaw - bearish alignment) OR chop too high
            if (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]) or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (lips > teeth > jaw - bullish alignment) OR chop too high
            if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]) or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Alligator alignment + chop filter
            if volume_confirmed and chop_filter:
                # Long entry: Lips > Teeth > Jaw (bullish alignment)
                if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Lips < Teeth < Jaw (bearish alignment)
                elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals