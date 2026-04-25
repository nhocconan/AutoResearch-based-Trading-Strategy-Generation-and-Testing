#!/usr/bin/env python3
"""
6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
Hypothesis: Camarilla levels derived from prior 1d range provide high-probability reversal (R3/S3) and breakout (R4/S4) zones on 6h timeframe. Volume confirmation filters false signals. Works in bull/bear via discrete sizing (0.25) and strict entry conditions targeting 12-37 trades/year.
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
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior 1d Camarilla levels (using completed 1d candle)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S4 = close - 1.5*(high-low), S3 = close - 1.1*(high-low)
    rng = high_1d - low_1d
    r4 = close_1d + 1.5 * rng
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    s4 = close_1d - 1.5 * rng
    
    # Align HTF levels to LTF (6h)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R4 with volume (continuation) OR price rejects at S3 with volume (fade)
            long_breakout = (curr_high > r4_aligned[i]) and vol_spike
            long_fade = (curr_low <= s3_aligned[i]) and (curr_close > s3_aligned[i]) and vol_spike
            
            # Short: price breaks below S4 with volume (continuation) OR price rejects at R3 with volume (fade)
            short_breakout = (curr_low < s4_aligned[i]) and vol_spike
            short_fade = (curr_high >= r3_aligned[i]) and (curr_close < r3_aligned[i]) and vol_spike
            
            if long_breakout or long_fade:
                signals[i] = 0.25
                position = 1
            elif short_breakout or short_fade:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price reaches R3 (take profit) OR breaks below S3 (stop)
            if (curr_high >= r3_aligned[i]) or (curr_low < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price reaches S3 (take profit) OR breaks above R3 (stop)
            if (curr_low <= s3_aligned[i]) or (curr_high > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_VolumeSpike"
timeframe = "6h"
leverage = 1.0