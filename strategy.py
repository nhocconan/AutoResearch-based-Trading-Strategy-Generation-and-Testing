#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1D Weekly Pivot (R1/S1) Breakout with Weekly Trend Filter and Volume Spike
# Uses weekly pivot levels for key support/resistance, weekly EMA34 for trend alignment,
# and volume spike for confirmation. Designed for 7-25 trades/year to avoid fee drag.
# Works in bull markets (breakouts with trend) and bear markets (fades from pivot levels with trend).
name = "1d_WeeklyPivot_R1S1_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and EMA trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_weekly = pd.Series(df_weekly['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Weekly pivot levels (using previous week's OHLC)
    weekly_high = df_weekly['high'].shift(1).values
    weekly_low = df_weekly['low'].shift(1).values
    weekly_close = df_weekly['close'].shift(1).values
    
    # Pivot calculations
    range_hl = weekly_high - weekly_low
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    r2 = pivot + range_hl * 1.1 / 6
    s2 = pivot - range_hl * 1.1 / 6
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    r4 = pivot + range_hl * 1.1 / 2
    s4 = pivot - range_hl * 1.1 / 2
    
    # Align pivot levels to 1d
    pivot_1d = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_1d = align_htf_to_ltf(prices, df_weekly, r1)
    s1_1d = align_htf_to_ltf(prices, df_weekly, s1)
    r2_1d = align_htf_to_ltf(prices, df_weekly, r2)
    s2_1d = align_htf_to_ltf(prices, df_weekly, s2)
    r3_1d = align_htf_to_ltf(prices, df_weekly, r3)
    s3_1d = align_htf_to_ltf(prices, df_weekly, s3)
    r4_1d = align_htf_to_ltf(prices, df_weekly, r4)
    s4_1d = align_htf_to_ltf(prices, df_weekly, s4)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d[i]) or np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or
            np.isnan(r2_1d[i]) or np.isnan(s2_1d[i]) or np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or
            np.isnan(r4_1d[i]) or np.isnan(s4_1d[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above R1 with weekly uptrend and volume spike
            if close[i] > r1_1d[i] and close[i] > ema34_1d[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with weekly downtrend and volume spike
            elif close[i] < s1_1d[i] and close[i] < ema34_1d[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below pivot OR weekly trend turns down
            if close[i] < pivot_1d[i] or close[i] < ema34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above pivot OR weekly trend turns up
            if close[i] > pivot_1d[i] or close[i] > ema34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals