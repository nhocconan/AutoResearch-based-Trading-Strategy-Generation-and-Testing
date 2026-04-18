#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Squeeze_Play
Hypothesis: Combines 1d Camarilla pivot levels (R1/S1) with Bollinger Band squeeze
for breakout trading. Uses Bollinger Band width percentile to identify low volatility
periods (squeeze) followed by expansion breakouts. Works in both bull and bear markets
by capturing volatility expansion after consolidation.
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Formula: P = (H + L + C)/3, Range = H - L
    # R1 = C + (H - L) * 1.1/12, S1 = C - (H - L) * 1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # Calculate Bollinger Bands on 1d close for squeeze detection
    # BB(20, 2): middle = SMA(20), std = stddev(20)
    bb_middle = np.full(len(close_1d), np.nan)
    bb_std = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        bb_middle[i] = np.mean(close_1d[i-20:i])
        bb_std[i] = np.std(close_1d[i-20:i])
    
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width percentile (252-period lookback ~ 1 year)
    bb_width_percentile = np.full(len(bb_width), np.nan)
    lookback = 252  # ~1 year of daily data
    
    for i in range(lookback, len(bb_width)):
        if not np.isnan(bb_width[i-lookback:i]).any():
            bb_width_percentile[i] = (bb_width[i] - np.min(bb_width[i-lookback:i])) / \
                                    (np.max(bb_width[i-lookback:i]) - np.min(bb_width[i-lookback:i])) * 100
    
    # Squeeze condition: BB width below 20th percentile (low volatility)
    squeeze = bb_width_percentile < 20
    
    # Expansion condition: BB width above 50th percentile (volatility expanding)
    expansion = bb_width_percentile > 50
    
    # Align 1d data to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze, additional_delay_bars=0)
    expansion_aligned = align_htf_to_ltf(prices, df_1d, expansion, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, lookback)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for squeeze followed by expansion breakout
            if expansion_aligned[i] and i > 0 and squeeze_aligned[i-1]:
                # Long: break above R1 with volume confirmation
                vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else volume[i]
                if close[i] > r1_aligned[i] and volume[i] > vol_ma * 1.5:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S1 with volume confirmation
                elif close[i] < s1_aligned[i] and volume[i] > vol_ma * 1.5:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or squeeze breaks down
            if close[i] < (pivot[0] if len(pivot) > 0 else r1_aligned[i]) * 0.995 or not expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or squeeze breaks down
            if close[i] > (pivot[0] if len(pivot) > 0 else s1_aligned[i]) * 1.005 or not expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_Pivot_Squeeze_Play"
timeframe = "4h"
leverage = 1.0