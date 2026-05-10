#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Uses 1-day Camarilla resistance R1 and support S1 levels as breakout levels,
filtered by 1-day EMA trend and volume spikes. Long when price breaks above R1 in uptrend,
short when price breaks below S1 in downtrend. Designed for low trade frequency and high win rate
in both bull and bear markets by trading with the higher timeframe trend.
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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Daily data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    range_ = prev_high - prev_low
    camarilla_r1 = prev_close + range_ * 1.1 / 12
    camarilla_s1 = prev_close - range_ * 1.1 / 12
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = prev_close > ema34_1d
    trend_1d_down = prev_close < ema34_1d
    
    # Align daily levels and trend to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above R1, uptrend, volume confirmation
            if (close[i] > r1_aligned[i] and trend_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, downtrend, volume confirmation
            elif (close[i] < s1_aligned[i] and trend_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 or trend changes
            if (close[i] < s1_aligned[i] or trend_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R1 or trend changes
            if (close[i] > r1_aligned[i] or trend_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals