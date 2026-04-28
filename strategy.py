#!/usr/bin/env python3
"""
4h_KAMA_Trend_Breakout_1dTrend_Filter
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) identifies the 4-hour trend direction. 
Enter long when price breaks above KAMA with 1-day uptrend and volume confirmation; 
enter short when price breaks below KAMA with 1-day downtrend and volume confirmation. 
Exit when price reverses back across KAMA with volume confirmation. 
Uses adaptive smoothing to reduce whipsaw in ranging markets while maintaining trend sensitivity. 
Designed for 15-30 trades per year with strict entry conditions to minimize fee drag.
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
    
    # Calculate KAMA on 4h data (ER = 10, Fast = 2, Slow = 30)
    change = np.abs(np.diff(close, n=10))  # 10-period absolute change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Initialize at period 10
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price crosses above KAMA + 1-day uptrend + volume surge
        long_entry = (close[i] > kama[i] and close[i-1] <= kama[i-1] and 
                     ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                     volume_surge[i])
        
        # Short: price crosses below KAMA + 1-day downtrend + volume surge
        short_entry = (close[i] < kama[i] and close[i-1] >= kama[i-1] and 
                      ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                      volume_surge[i])
        
        # Exit when price crosses back across KAMA with volume confirmation
        long_exit = (close[i] < kama[i] and close[i-1] >= kama[i-1] and 
                    volume_surge[i])
        short_exit = (close[i] > kama[i] and close[i-1] <= kama[i-1] and 
                     volume_surge[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_Breakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0