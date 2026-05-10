#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_1dTrend
# Hypothesis: Breakout from weekly pivot levels with daily trend alignment and volume confirmation.
# Weekly pivot levels act as key support/resistance, and breakouts above/below these levels
# with volume confirmation and daily trend alignment provide high-probability trades.
# This strategy targets low-frequency, high-conviction trades suitable for 1d timeframe.
# Works in both bull and bear markets by trading breakouts in the direction of weekly trend.

name = "1d_WeeklyPivot_Breakout_1dTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly timeframe data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (High + Low + Close) / 3
    # R1 = 2*Pivot - Low
    # S1 = 2*Pivot - High
    # R2 = Pivot + (High - Low)
    # S2 = Pivot - (High - Low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Daily trend filter (EMA 50)
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close > ema50
    daily_downtrend = close < ema50
    
    # Volume confirmation (volume > 1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume and daily uptrend
            if (close[i] > weekly_r1_aligned[i] and 
                volume_confirm[i] and 
                daily_uptrend[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume and daily downtrend
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_confirm[i] and 
                  daily_downtrend[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below weekly pivot or daily trend turns down
            if (close[i] < weekly_pivot_aligned[i] or 
                not daily_uptrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above weekly pivot or daily trend turns up
            if (close[i] > weekly_pivot_aligned[i] or 
                not daily_downtrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals