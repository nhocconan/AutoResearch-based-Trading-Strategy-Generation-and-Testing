#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Daily Pivot Breakout + Weekly Trend + Volume Spike
# Daily pivots provide strong intraday support/resistance levels.
# Breakout above R1 or below S1 with weekly trend alignment and volume confirmation
# captures momentum moves with institutional validation.
# Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag.
name = "12h_DailyPivot_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's data)
    # We need to shift by 1 to avoid look-ahead (use previous day's OHLC)
    daily_high = df_daily['high'].shift(1).values
    daily_low = df_daily['low'].shift(1).values
    daily_close = df_daily['close'].shift(1).values
    
    # Calculate pivot points
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Align daily pivot levels to 12h
    pivot_12h = align_htf_to_ltf(prices, df_daily, pivot)
    r1_12h = align_htf_to_ltf(prices, df_daily, r1)
    s1_12h = align_htf_to_ltf(prices, df_daily, s1)
    
    # Weekly EMA50 for trend filter
    ema50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_12h = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema50_weekly_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above R1 with weekly uptrend and volume spike
            if close[i] > r1_12h[i] and close[i] > ema50_weekly_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with weekly downtrend and volume spike
            elif close[i] < s1_12h[i] and close[i] < ema50_weekly_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below pivot OR weekly trend turns down
            if close[i] < pivot_12h[i] or close[i] < ema50_weekly_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above pivot OR weekly trend turns up
            if close[i] > pivot_12h[i] or close[i] > ema50_weekly_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals