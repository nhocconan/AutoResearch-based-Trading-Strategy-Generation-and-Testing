#!/usr/bin/env python3
name = "6h_WeeklyPivot_Pullback_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Previous week's OHLC for Weekly Pivot calculation
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly Pivot Point and Support/Resistance levels
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r1 = 2 * weekly_pivot - prev_week_low
    weekly_s1 = 2 * weekly_pivot - prev_week_high
    weekly_r2 = weekly_pivot + (prev_week_high - prev_week_low)
    weekly_s2 = weekly_pivot - (prev_week_high - prev_week_low)
    
    # Align Weekly Pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # 1d EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average (on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Wait for EMA and volume MA warmup
        # Skip if any critical value is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to S1 in uptrend with volume confirmation
            if (low[i] <= weekly_s1_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: pullback to R1 in downtrend with volume confirmation
            elif (high[i] >= weekly_r1_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below pivot or drops below 1d EMA
            if close[i] < weekly_pivot_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above pivot or rises above 1d EMA
            if close[i] > weekly_pivot_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals