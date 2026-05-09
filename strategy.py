#!/usr/bin/env python3
name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's data for today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day's data
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2x 24-period average
    volume_ma = np.zeros(n)
    for i in range(24, n):
        volume_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not ready
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Enter long: price crosses above Camarilla R1 + uptrend + volume spike
            if close[i] > camarilla_r1_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below Camarilla S1 + downtrend + volume spike
            elif close[i] < camarilla_s1_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Camarilla S1
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Camarilla R1
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals