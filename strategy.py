#/usr/bin/env python3
"""
6h_Pivot_Volume_Squeeze
Hypothesis: Combines 15-minute price range contraction (squeeze) with 1d pivot points and volume expansion for breakouts.
Works in both bull and bear markets by trading breakouts of consolidation with volume confirmation.
Uses Bollinger Band width for squeeze detection and 1d pivot levels for directional bias.
Target: 20-30 trades/year with strict entry conditions to avoid overtrading.
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
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d pivot points (standard floor trader pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Bollinger Bands for squeeze detection (20-period, 2 std)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std_dev)
    lower_bb = sma - (bb_std * std_dev)
    bb_width = (upper_bb - lower_bb) / sma  # Normalized width
    
    # Squeeze condition: BB width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_expansion = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(std_dev[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume expansion
        long_breakout = (close[i] > r1_aligned[i]) and vol_expansion[i]
        short_breakout = (close[i] < s1_aligned[i]) and vol_expansion[i]
        
        # Exit conditions: opposite breakout or loss of momentum
        long_exit = (close[i] < s1_aligned[i]) or (not vol_expansion[i] and position == 1)
        short_exit = (close[i] > r1_aligned[i]) or (not vol_expansion[i] and position == -1)
        
        # Only trade during squeeze breakouts (avoid choppy markets)
        if squeeze[i]:
            if long_breakout and position <= 0:
                signals[i] = 0.25
                position = 1
            elif short_breakout and position >= 0:
                signals[i] = -0.25
                position = -1
            elif long_exit and position == 1:
                signals[i] = 0.0
                position = 0
            elif short_exit and position == -1:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position during squeeze
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Outside squeeze: flatten position (avoid chop)
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Pivot_Volume_Squeeze"
timeframe = "6h"
leverage = 1.0