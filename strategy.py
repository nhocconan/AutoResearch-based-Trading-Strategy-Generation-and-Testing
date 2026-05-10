#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Buy breaks above Camarilla R3 and sell breaks below S3 on 12h charts, with daily trend filter (EMA34) and volume confirmation. Works in bull/bear by following daily trend, using Camarilla levels as key support/resistance from institutional order flow, and volume to confirm breakout strength. Target: 15-30 trades/year per symbol.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # 12h Camarilla levels (based on previous day's range)
    # Calculate daily high/low/close for Camarilla formula
    # We'll use 1d data to compute Camarilla levels for the 12h chart
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].shift(1).values  # previous day high
    pl = df_1d['low'].shift(1).values   # previous day low
    pc = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla formulas
    R3 = pc + (ph - pl) * 1.1 / 2
    S3 = pc - (ph - pl) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily trend filter: EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R3 + volume confirmation
            if daily_up and close[i] > R3_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + price breaks below S3 + volume confirmation
            elif daily_down and close[i] < S3_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price moves back below R3 or trend changes
            if close[i] < R3_aligned[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price moves back above S3 or trend changes
            if close[i] > S3_aligned[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals