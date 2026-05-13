#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels (R1, S1) from the daily chart act as strong support/resistance.
Breakouts above R1 or below S1 with volume confirmation and daily EMA trend filter capture
institutional order flow. Designed for low trade frequency (20-40/year) with trend-following
logic that works in both bull and bear markets by following the daily trend.
"""

name = "4h_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Camarilla levels from previous day
    # Using daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels: R1, S1 from previous day's range
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each day
    camarilla_range = (prev_high - prev_low) * 1.1 / 12
    r1_level = prev_close + camarilla_range
    s1_level = prev_close - camarilla_range
    
    # Align daily levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Daily EMA 50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and uptrend
            if close[i] > r1_4h[i] and volume_confirm[i]:
                if close[i] > ema_50_1d_aligned[i]:  # Only long in uptrend
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and downtrend
            elif close[i] < s1_4h[i] and volume_confirm[i]:
                if close[i] < ema_50_1d_aligned[i]:  # Only short in downtrend
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below S1 or trend changes
            if close[i] < s1_4h[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above R1 or trend changes
            if close[i] > r1_4h[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals