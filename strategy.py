#!/usr/bin/env python3
# 12H_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakouts from Camarilla R1/S1 levels with daily trend filter and volume confirmation.
# Long when: price breaks above R1 in daily uptrend with volume spike.
# Short when: price breaks below S1 in daily downtrend with volume spike.
# Uses 1w trend filter for stronger regime filtering.
# Works in bull/bear by following higher timeframe trends and using volume to confirm institutional interest.
# Target: 12-37 trades/year per symbol.

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
    
    # 12h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Weekly trend filter (stronger regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for 12h
        # Using previous bar's high, low, close
        if i == 0:
            continue
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_val = ph - pl
        
        # Avoid division by zero
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla levels
        r1 = pc + (range_val * 1.1 / 12)
        s1 = pc - (range_val * 1.1 / 12)
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0  # Strong volume spike
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily AND weekly uptrend + price breaks above R1 + volume spike
            if daily_up and weekly_up and close[i] > r1 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily AND weekly downtrend + price breaks below S1 + volume spike
            elif daily_down and weekly_down and close[i] < s1 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend breaks or price returns below R1
            if not (daily_up and weekly_up) or close[i] < r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend breaks or price returns above S1
            if not (daily_down and weekly_down) or close[i] > s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals