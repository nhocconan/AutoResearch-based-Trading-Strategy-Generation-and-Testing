#!/usr/bin/env python3
"""
4h_WilliamsAlligator_Range_Bound_Reversal
Hypothesis: In range-bound markets, price tends to reverse when touching the Williams Alligator's jaw (13-period smoothed median) after showing momentum divergence. Uses Williams Alligator (13,8,5 SMAs with 8,5,3 shifts) from 1d timeframe to identify range, and RSI(2) for short-term reversal signals. Works in both bull and bear markets by focusing on mean reversion within ranges rather than trend following. Low trade frequency due to strict range identification and reversal confirmation requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate median prices for 1d
    median_1d = (df_1d['high'] + df_1d['low']) / 2
    
    # Williams Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = median_1d.rolling(window=13, min_periods=13).mean().shift(8)
    teeth_raw = median_1d.rolling(window=8, min_periods=8).mean().shift(5)
    lips_raw = median_1d.rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw = jaw_raw.values
    teeth = teeth_raw.values
    lips = lips_raw.values
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Range identification: market is ranging when Alligator lines are intertwined
    # Calculate the average distance between lines
    jaw_teeth_dist = np.abs(jaw_aligned - teeth_aligned)
    teeth_lips_dist = np.abs(teeth_aligned - lips_aligned)
    lips_jaw_dist = np.abs(lips_aligned - jaw_aligned)
    avg_dist = (jaw_teeth_dist + teeth_lips_dist + lips_jaw_dist) / 3
    
    # Normalize by price to get relative distance
    price_avg = (jaw_aligned + teeth_aligned + lips_aligned) / 3
    relative_dist = avg_dist / price_avg
    
    # Range threshold: when relative distance is small, market is ranging
    # Use 20-period rolling average of relative distance to establish normal range
    dist_ma = pd.Series(relative_dist).rolling(window=20, min_periods=20).mean().values
    is_ranging = relative_dist < (dist_ma * 0.5)  # Much tighter than normal
    
    # RSI(2) for short-term reversal signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=2, min_periods=2).mean()
    avg_loss = loss.rolling(window=2, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(40, 20)  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(is_ranging[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        range_flag = is_ranging[i]
        rsi_val = rsi[i]
        
        if position == 0:
            if range_flag:
                # Long reversal: price near lips/jaw and RSI oversold
                if price <= lips_val * 1.005 and rsi_val < 20:
                    signals[i] = 0.25
                    position = 1
                # Short reversal: price near teeth/jaw and RSI overbought
                elif price >= teeth_val * 0.995 and rsi_val > 80:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI overbought or price moves significantly above lips
            if rsi_val > 70 or price > lips_val * 1.02:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI oversold or price moves significantly below teeth
            if rsi_val < 30 or price < teeth_val * 0.98:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_WilliamsAlligator_Range_Bound_Reversal"
timeframe = "4h"
leverage = 1.0