#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout (R1/S1) with Weekly Trend Filter and Volume Spike
# Uses daily Camarilla pivot levels (R1/S1) for breakout signals, weekly EMA34 for trend alignment,
# and volume spike for confirmation. Designed for 12h timeframe to target 50-150 trades over 4 years.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with volume).
name = "12h_Camarilla_R1S1_WeeklyTrend_Volume"
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
    
    # Get daily data for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    range_ = daily_high - daily_low
    r1 = pivot + range_ * 1.1 / 12
    s1 = pivot - range_ * 1.1 / 12
    
    # Align daily Camarilla levels to 12h
    r1_12h = align_htf_to_ltf(prices, df_daily, r1)
    s1_12h = align_htf_to_ltf(prices, df_daily, s1)
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_weekly = pd.Series(df_weekly['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema34_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above daily Camarilla R1 with weekly uptrend and volume spike
            if close[i] > r1_12h[i] and close[i] > ema34_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below daily Camarilla S1 with weekly downtrend and volume spike
            elif close[i] < s1_12h[i] and close[i] < ema34_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below daily Camarilla S1 OR weekly trend turns down
            if close[i] < s1_12h[i] or close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above daily Camarilla R1 OR weekly trend turns up
            if close[i] > r1_12h[i] or close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals