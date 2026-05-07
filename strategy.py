#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    trend_up = close > sma50_1w_aligned
    trend_down = close < sma50_1w_aligned
    
    # Daily pivot points (from previous day)
    high_prev = prices['high'].shift(1).values
    low_prev = prices['low'].shift(1).values
    close_prev = prices['close'].shift(1).values
    
    PP = (high_prev + low_prev + close_prev) / 3.0
    R1 = PP * 2 - low_prev
    S1 = PP * 2 - high_prev
    # Align pivot levels (they are based on previous day, so no additional alignment needed)
    # Since pivot is based on previous day, it's already known at current day open
    # But we align to be safe with any data misalignment
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for SMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(sma50_1w_aligned[i]) or np.isnan(PP[i]) or np.isnan(R1[i]) or np.isnan(S1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume surge and 1w uptrend
            if close[i] > R1[i] and vol_surge[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume surge and 1w downtrend
            elif close[i] < S1[i] and vol_surge[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend turns down
            if close[i] < S1[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 or trend turns up
            if close[i] > R1[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly trend filter with daily pivot breakouts captures institutional moves in both bull and bear markets.
# Long when price breaks above daily R1 with volume confirmation in weekly uptrend.
# Short when price breaks below daily S1 with volume confirmation in weekly downtrend.
# Uses weekly SMA50 for trend, daily pivot points for institutional levels, and volume surge for conviction.
# Designed for 1d timeframe to target 15-25 trades per year with low frequency to minimize fee drag.
# Works in bull markets (breaks above R1 in uptrend) and bear markets (breaks below S1 in downtrend).