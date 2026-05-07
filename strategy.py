#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Use daily Camarilla pivot levels R1/S1 breakout with weekly trend filter and volume confirmation.
Long when price breaks above R1 in weekly uptrend with volume > 1.5x average.
Short when price breaks below S1 in weekly downtrend with volume > 1.5x average.
Exit when price returns to daily pivot point (PP).
Designed for 1d to capture multi-day moves with low frequency (target 10-25 trades/year).
Works in both bull and bear markets via weekly trend filter.
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily pivot points (using previous day's HLC)
    # Shift by 1 to use previous day's data
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First day has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pp + (range_hl * 1.08333)  # R1 = PP + 0.0833 * range
    s1 = pp - (range_hl * 1.08333)  # S1 = PP - 0.0833 * range
    
    # Volume confirmation: 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(pp[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend determination
        trend_1w_up = close[i] > ema_50_1w_aligned[i]
        trend_1w_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 in weekly uptrend with volume confirmation
            if (close[i] > r1[i] and close[i-1] <= r1[i-1] and  # breakout above R1
                trend_1w_up and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in weekly downtrend with volume confirmation
            elif (close[i] < s1[i] and close[i-1] >= s1[i-1] and  # breakdown below S1
                  trend_1w_down and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to or below pivot point
            if close[i] <= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to or above pivot point
            if close[i] >= pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals