#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout + Daily Trend + Volume Spike
# Weekly pivots provide strong institutional support/resistance levels.
# Breakout above R1 or below S1 with daily trend alignment and volume confirmation
# captures momentum moves with institutional validation.
# Target: 20-40 trades/year (80-160 over 4 years) to avoid fee drag.
name = "6h_WeeklyPivot_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    # We need to shift by 1 to avoid look-ahead (use previous week's OHLC)
    weekly_high = df_weekly['high'].shift(1).values
    weekly_low = df_weekly['low'].shift(1).values
    weekly_close = df_weekly['close'].shift(1).values
    
    # Calculate pivot points
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_6h = align_htf_to_ltf(prices, df_weekly, r1)
    s1_6h = align_htf_to_ltf(prices, df_weekly, s1)
    r2_6h = align_htf_to_ltf(prices, df_weekly, r2)
    s2_6h = align_htf_to_ltf(prices, df_weekly, s2)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Daily EMA50 for trend filter
    ema50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_6h = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(ema50_daily_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above R1 with daily uptrend and volume spike
            if close[i] > r1_6h[i] and close[i] > ema50_daily_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with daily downtrend and volume spike
            elif close[i] < s1_6h[i] and close[i] < ema50_daily_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below pivot OR daily trend turns down
            if close[i] < pivot_6h[i] or close[i] < ema50_daily_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above pivot OR daily trend turns up
            if close[i] > pivot_6h[i] or close[i] > ema50_daily_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals