#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Combines Camarilla pivot levels from daily timeframe with 1-day EMA trend filter and volume confirmation.
Goes long when price breaks above R1 in an uptrend with above-average volume.
Goes short when price breaks below S1 in a downtrend with above-average volume.
Uses daily timeframe for pivot calculation and trend filter to reduce noise and improve win rate.
Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # === DAILY DATA FOR PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Handle first day where shift(1) is NaN
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    
    # Calculate R1 and S1
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: 20-period average volume
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    # Handle edges
    vol_ma[:10] = np.mean(volume[:20]) if len(volume) >= 20 else volume[0]
    vol_ma[-10:] = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA34 or breaks below S1
            if close[i] < ema34_1d_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above EMA34 or breaks above R1
            if close[i] > ema34_1d_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals