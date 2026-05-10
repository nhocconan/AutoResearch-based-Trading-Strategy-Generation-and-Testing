#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout at Camarilla R1 (long) or S1 (short) with daily trend and volume confirmation.
# Uses 1d Camarilla levels from previous day, 4h price breakout, and volume > 1.5x average.
# Works in bull/bear by following daily trend and using volume to confirm institutional interest.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily Camarilla levels from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i == 0:
            R1[i] = np.nan
            S1[i] = np.nan
        else:
            # Previous day's range
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            if range_val <= 0:
                R1[i] = np.nan
                S1[i] = np.nan
            else:
                R1[i] = prev_close + range_val * 1.1 / 12
                S1[i] = prev_close - range_val * 1.1 / 12
    
    # Align daily Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily trend filter (EMA50)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R1 + volume
            if daily_up and volume_confirm:
                if close[i] > R1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: daily downtrend + price breaks below S1 + volume
            elif daily_down and volume_confirm:
                if close[i] < S1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price drops below S1 or trend changes
            if close[i] < S1_aligned[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above R1 or trend changes
            if close[i] > R1_aligned[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals