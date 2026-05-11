#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla pivot levels (S1, R1)
    prev_close = close_1d[:-1]  # Shift by 1 to get previous day
    prev_high = high_1d[:-1]
    prev_low = low_1d[:-1]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Camarilla levels: S1 = close - 1.1/12 * range, R1 = close + 1.1/12 * range
    s1 = prev_close - (1.1 / 12) * range_
    r1 = prev_close + (1.1 / 12) * range_
    
    # Align to 12h timeframe (wait for previous day to close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # 1d trend: EMA34 > EMA89 (using close)
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    trend_up = ema_34 > ema_89
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for EMA and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + uptrend + volume confirmation
            if close[i] > r1_aligned[i] and trend_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + downtrend + volume confirmation
            elif close[i] < s1_aligned[i] and not trend_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below S1 or trend reverses
            if close[i] < s1_aligned[i] or not trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above R1 or trend reverses
            if close[i] > r1_aligned[i] or trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals