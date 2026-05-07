#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: 12h Camarilla R3/S3 breakouts with 1-week trend filter and volume confirmation.
This strategy targets major trend reversals and continuations on higher timeframes.
R3/S3 levels represent stronger breakout points than R1/S1, reducing false signals.
The 1-week trend filter ensures alignment with the major trend, improving performance in both bull and bear markets.
Volume confirmation ensures institutional participation.
Target: 15-30 trades per year (~60-120 over 4 years) with position size 0.25.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Roll to get previous day values
    prev_high = np.roll(prev_high_1d, 1)
    prev_low = np.roll(prev_low_1d, 1)
    prev_close = np.roll(prev_close_1d, 1)
    # Fill first values
    prev_high[0] = prev_high_1d[0]
    prev_low[0] = prev_low_1d[0]
    prev_close[0] = prev_close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    R3 = close + (range_val * 1.1 / 4)
    S3 = close - (range_val * 1.1 / 4)
    
    # Volume ratio: current volume / 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 periods for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(R3[i]) or np.isnan(S3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above R3 or below S3
        breakout_up = close[i] > R3[i-1]  # Use previous bar's level to avoid look-ahead
        breakout_down = close[i] < S3[i-1]  # Use previous bar's level
        
        # Volume confirmation: volume > 1.8x average
        volume_confirm = vol_ratio[i] > 1.8
        
        # Trend filter from 1-week EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below S3 or trend reversal
            if close[i] < S3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above R3 or trend reversal
            if close[i] > R3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals