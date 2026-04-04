#!/usr/bin/env python3
"""
exp_6451_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout in direction of 1d daily pivot bias (above/below PP) with volume confirmation.
Works in bull/bear: pivot provides structural bias, Donchian captures breakouts, volume filters false signals.
Target trades: 12-37/year (50-150 over 4 years). Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6451_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d daily pivot points (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 = 2*P - L
    r1 = 2 * pp - low_1d
    # Support 1 = 2*P - H
    s1 = 2 * pp - high_1d
    # Bias: 1 if price > PP (bullish), -1 if price < PP (bearish), 0 otherwise
    bias = np.where(close_1d > pp, 1, np.where(close_1d < pp, -1, 0))
    
    # Align HTF arrays to LTF with shift(1) for completed bars only
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    bias_aligned = align_htf_to_ltf(prices, df_1d, bias)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    high_roll = prices['high'].rolling(window=lookback, min_periods=lookback).max().values
    low_roll = prices['low'].rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 6h volume > 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_confirm = prices['volume'].values > vol_ma
    
    signals = np.zeros(n)
    
    # Start from lookback to ensure valid Donchian levels
    start_idx = max(lookback, 20)  # also need HTF data
    
    for i in range(start_idx, n):
        # Get aligned HTF values for current bar
        bias_val = bias_aligned[i]
        
        # Skip if no clear bias
        if bias_val == 0:
            continue
            
        # Long condition: price breaks above Donchian upper AND bullish bias AND volume confirmation
        if prices['close'].values[i] > high_roll[i] and bias_val == 1 and vol_confirm[i]:
            signals[i] = 0.25  # 25% position
            
        # Short condition: price breaks below Donchian lower AND bearish bias AND volume confirmation
        elif prices['close'].values[i] < low_roll[i] and bias_val == -1 and vol_confirm[i]:
            signals[i] = -0.25  # 25% position
            
        # Exit conditions: reverse signal or loss of bias/volume
        elif signals[i-1] != 0:
            current_pos = 1 if signals[i-1] > 0 else -1
            # Exit if price reverses through opposite Donchian level
            if current_pos == 1 and prices['close'].values[i] < low_roll[i]:
                signals[i] = 0.0
            elif current_pos == -1 and prices['close'].values[i] > high_roll[i]:
                signals[i] = 0.0
            # Exit if bias changes
            elif bias_val == 0 or bias_val != current_pos:
                signals[i] = 0.0
            # Exit if volume confirmation fails
            elif not vol_confirm[i]:
                signals[i] = 0.0
            # Otherwise hold position
            else:
                signals[i] = signals[i-1]
    
    return signals