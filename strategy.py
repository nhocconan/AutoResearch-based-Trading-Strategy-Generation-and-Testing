#!/usr/bin/env python3
"""
6h_weekly_pivot_1d_trend_volume_v1
Hypothesis: Weekly pivot levels (classic) act as institutional support/resistance. Price respects these levels in both bull and bear markets. Uses 1d EMA for trend filter and volume spike for confirmation. Designed for 6h timeframe to capture multi-day moves with low frequency (target: 20-50 trades/year) to minimize fee drift. Works in bull markets by buying at weekly S1/S2 with trend up, and in bear markets by selling at weekly R1/R2 with trend down.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for classic pivot points (based on previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly classic pivot points (based on previous week)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Support 2 = Pivot - (High - Low)
    # Resistance 1 = (2 * Pivot) - Low
    # Resistance 2 = Pivot + (High - Low)
    weekly_close = df_1w['close'].shift(1).values
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    r1 = (2 * weekly_pivot) - weekly_high
    r2 = weekly_pivot + weekly_range
    s1 = (2 * weekly_pivot) - weekly_low
    s2 = weekly_pivot - weekly_range
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_20 = df_1d['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all weekly data to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation (24-period average = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S1 or trend turns bearish
            if close[i] <= s1_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above R1 or trend turns bullish
            if close[i] >= r1_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or breaks above S1 with volume and bullish trend
            if (close[i] >= s1_aligned[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or breaks below R1 with volume and bearish trend
            elif (close[i] <= r1_aligned[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals