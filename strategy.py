#!/usr/bin/env python3
name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's Camarilla pivot levels (R1, S1)
    # Using previous day's OHLC to avoid look-ahead
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (1.1/12.0) * (prev_high - prev_low)
    s1 = pivot - (1.1/12.0) * (prev_high - prev_low)
    
    # Calculate 20-day EMA for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1-day indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema20_12h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema20_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average volume
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > avg_volume * 1.3
        else:
            volume_confirm = False
        
        if position == 0:
            # Enter long: price above R1 + above EMA20 + volume confirmation
            if close[i] > r1_12h[i] and close[i] > ema20_12h[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price below S1 + below EMA20 + volume confirmation
            elif close[i] < s1_12h[i] and close[i] < ema20_12h[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below S1 or below EMA20
            if close[i] < s1_12h[i] or close[i] < ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above R1 or above EMA20
            if close[i] > r1_12h[i] or close[i] > ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals