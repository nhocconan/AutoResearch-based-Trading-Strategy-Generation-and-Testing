#!/usr/bin/env python3
"""
4h_Camarilla_R4_S4_Breakout_1dEMA50_VolumeSpike
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R4 level AND 1d trend is up (close > EMA50) AND volume > 2.0x 20-period average volume. Enter short when price breaks below Camarilla S4 level AND 1d trend is down (close < EMA50) AND volume > 2.0x 20-period average volume. Uses tighter R4/S4 levels for fewer, higher-quality trades and a slightly lower volume threshold to balance frequency. Target: 20-50 trades/year.
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
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r4 = prev_close_1d + 1.1 * camarilla_range / 2
    s4 = prev_close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels and EMA50 to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: dynamic threshold based on volume percentile
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.maximum(volume_ma, 1e-10)
    volume_spike = volume_ratio > 2.0  # Slightly lower threshold for more signals
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r4_aligned[i]
        breakout_down = close[i] < s4_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R4 + volume spike + 1d uptrend
            long_signal = breakout_up and volume_spike[i] and trend_uptrend
            
            # Short: breakout below S4 + volume spike + 1d downtrend
            short_signal = breakout_down and volume_spike[i] and trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend change to downtrend
            if not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend change to uptrend
            if not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0