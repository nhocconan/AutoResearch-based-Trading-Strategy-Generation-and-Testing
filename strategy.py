#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + Chop Regime Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength on 12h chart.
Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and choppy regime (CHOP > 61.8).
Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and choppy regime.
Uses 12h timeframe with 1d HTF for Alligator and chop filter. Targets 50-150 total trades over 4 years.
Works in both bull and bear markets: chop filter avoids whipsaws in ranging markets, volume confirmation
avoids false signals, Alligator provides clear trend alignment. Discrete position sizing (0.25) minimizes fee churn.
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
    
    # Get 1d data for Williams Alligator and chop filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Shift forward: Jaw +8, Teeth +5, Lips +3
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate Choppiness Index on 1d timeframe
    def true_range(high_arr, low_arr, close_arr):
        """Calculate True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        return tr
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr = true_range(high_1d, low_1d, close_1d_arr)
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])  # First ATR is average of first 14 TR
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Choppiness Index: CHOP = 100 * log10(SUM(ATR14) / (MAX(HIGH) - MIN(LOW))) / log10(n)
    chop_raw = np.full_like(close_1d_arr, np.nan)
    lookback = 14
    for i in range(lookback, len(close_1d_arr)):
        sum_atr = np.sum(atr_14[i-lookback+1:i+1])
        max_high = np.max(high_1d[i-lookback+1:i+1])
        min_low = np.min(low_1d[i-lookback+1:i+1])
        if max_high > min_low and sum_atr > 0:
            chop_raw[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(lookback)
    
    # Align 1d indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator, chop, and volume MA
    start_idx = max(20, 14)  # 20 for volume MA, 14 for chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20_12h[i])):
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
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma_12h
        
        # Chop regime filter: CHOP > 61.8 indicates ranging/choppy market (good for mean reversion)
        # In choppy markets, we fade Alligator extremes; in trending markets, we follow
        chop_regime = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals
            # Long: Lips > Teeth > Jaw (bullish alignment) AND volume confirmation AND chop regime
            long_entry = (lips_val > teeth_val and teeth_val > jaw_val and 
                         volume_confirm and chop_regime)
            # Short: Lips < Teeth < Jaw (bearish alignment) AND volume confirmation AND chop regime
            short_entry = (lips_val < teeth_val and teeth_val < jaw_val and 
                          volume_confirm and chop_regime)
            
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
            # Exit: Alligator alignment breaks down (Lips <= Teeth) OR chop regime ends
            if (lips_val <= teeth_val or chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator alignment breaks down (Lips >= Teeth) OR chop regime ends
            if (lips_val >= teeth_val or chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0