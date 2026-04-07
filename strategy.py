#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with KAMA Trend Filter and Volume Spike
Hypothesis: Weekly trend filter reduces whipsaw in daily breakouts. 
Weekly KAMA adapts to volatility, capturing strong trends while avoiding chop.
Volume spikes confirm institutional participation. Works in bull/bear by following weekly trend.
Target: 15-25 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_kama_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian Channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume Spike Detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Higher threshold for lower frequency
    
    # Weekly KAMA Trend Filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    
    # Calculate Efficiency Ratio and Smoothing Constants
    change = np.abs(np.diff(weekly_close, prepend=weekly_close[0]))
    volatility = np.sum(np.abs(np.diff(weekly_close, prepend=weekly_close[0])), axis=0)
    # Handle the array operation correctly
    volatility_series = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility_series > 0, change / volatility_series, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    
    # Calculate KAMA
    kama = np.full_like(weekly_close, np.nan, dtype=float)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    
    # Align to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(kama_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly KAMA
            if close[i] < kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly KAMA
            if close[i] > kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high + price above weekly KAMA + volume spike
            if (close[i] > high_roll[i-1] and 
                close[i] > kama_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low + price below weekly KAMA + volume spike
            elif (close[i] < low_roll[i-1] and 
                  close[i] < kama_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals