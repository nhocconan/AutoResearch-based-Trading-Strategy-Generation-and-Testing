#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Choppiness Regime Filter
# Williams Alligator (JAWS=13, TEETH=8, LIPS=5 SMMA) identifies trend direction and strength
# 1d volume spike (>2.0x 20-period average) confirms institutional participation
# Choppiness Index (CHOP) > 61.8 = ranging market (fade extremes), CHOP < 38.2 = trending (ride trend)
# This combination avoids whipsaws in sideways markets while capturing strong trends
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d volume spike confirmation (>2.0x 20-period average)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > 2.0 * vol_ma_20
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # First value
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(TRSUM / (HH - LL)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop = np.full_like(close_1d, 50.0)  # Default to neutral
    valid = (hl_range > 0) & (~np.isnan(tr_sum))
    chop[valid] = 100 * np.log10(tr_sum[valid] / hl_range[valid]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Williams Alligator on 12h timeframe (SMMA with specific periods)
    # JAWS: 13-period SMMA, shifted 8 bars forward
    # TEETH: 8-period SMMA, shifted 5 bars forward  
    # LIPS: 5-period SMMA, shifted 3 bars forward
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition (JAWS 8, TEETH 5, LIPS 3)
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First values become NaN due to roll
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 13)  # Volume MA, Chop, Jaws warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol_spike = vol_spike_aligned[i]
        curr_chop = chop_aligned[i]
        curr_jaws = jaws_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        
        # Alligator trend detection:
        # Uptrend: Lips > Teeth > Jaws (green, aligned upward)
        # Downtrend: Lips < Teeth < Jaws (red, aligned downward)
        # Otherwise: sideways/transition
        is_uptrend = (curr_lips > curr_teeth) and (curr_teeth > curr_jaws)
        is_downtrend = (curr_lips < curr_teeth) and (curr_teeth < curr_jaws)
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: trend turns down OR chop too high (ranging) OR no volume spike
            if not is_uptrend or curr_chop > 61.8 or not curr_vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns up OR chop too high (ranging) OR no volume spike
            if not is_downtrend or curr_chop > 61.8 or not curr_vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: clear uptrend + low chop (trending) + volume spike
            if is_uptrend and curr_chop < 38.2 and curr_vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: clear downtrend + low chop (trending) + volume spike
            elif is_downtrend and curr_chop < 38.2 and curr_vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals