#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_VolumeSpike
Hypothesis: Trade 6h timeframe using weekly pivot points for directional bias (from 1w), 
daily EMA50 for trend filter, and daily volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price > weekly pivot R1 AND above daily EMA50 AND volume spike. 
Enter short when price < weekly pivot S1 AND below daily EMA50 AND volume spike. 
Exit on opposite pivot touch or trend reversal. Uses discrete sizing 0.25 to balance 
return and drawdown. Target 12-37 trades/year on 6h timeframe. Works in bull/bear via 
weekly pivot structure and trend filter.
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
    
    # Get 1w data for weekly pivot points (PP, R1, S1)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly pivot points to 6h timeframe (completed weekly bar only)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get 1d data for daily EMA50 trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly R1 AND above daily EMA50 AND volume spike
            long_setup = (close[i] > r1_1w_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price below weekly S1 AND below daily EMA50 AND volume spike
            short_setup = (close[i] < s1_1w_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches weekly S1 OR closes below daily EMA50
            if (close[i] <= s1_1w_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches weekly R1 OR closes above daily EMA50
            if (close[i] >= r1_1w_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0