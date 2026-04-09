#!/usr/bin/env python3
# 6h_camarilla_pivot_volume_v1
# Hypothesis: 6h strategy using Camarilla pivot levels from 1d timeframe with volume confirmation. Enters long when price breaks above R4 with volume > 1.5x 20-period average (breakout continuation); enters short when price breaks below S4 with volume confirmation. Uses discrete position sizing (0.25) to limit fee drag. Targets 12-37 trades/year by requiring strict breakout conditions. Works in both bull and bear markets by capturing strong momentum moves aligned with volume spikes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r4 = pivot + range_ * 1.1 / 2
    r3 = pivot + range_ * 1.1 / 4
    r2 = pivot + range_ * 1.1 / 6
    r1 = pivot + range_ * 1.1 / 12
    s1 = pivot - range_ * 1.1 / 12
    s2 = pivot - range_ * 1.1 / 6
    s3 = pivot - range_ * 1.1 / 4
    s4 = pivot - range_ * 1.1 / 2
    return r4, r3, r2, r1, pivot, s1, s2, s3, s4

name = "6h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for Camarilla levels
    r4_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    # Calculate Camarilla levels for each 1d bar
    for i in range(len(df_1d)):
        r4, _, _, _, _, _, _, _, s4 = calculate_camarilla_pivot(high_1d[i], low_1d[i], close_1d[i])
        r4_1d[i] = r4
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe (with 1-bar delay for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price drops below R3 (take profit) or reverses below R4 (stop)
            if close[i] < r4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above S3 (take profit) or reverses above S4 (stop)
            if close[i] > s4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and Camarilla breakout
            if volume_confirmed:
                # Long: price breaks above R4 with volume
                if close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4 with volume
                elif close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals