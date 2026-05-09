#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WeeklyPivot_R2S2_Rejection_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly OHLC for pivot levels (based on previous week's OHLC)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate pivot and levels from previous week's OHLC
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    prev_weekly_range = prev_high_1w - prev_low_1w
    pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r2 = pivot + 1.1 * prev_weekly_range / 2
    r3 = pivot + 1.1 * prev_weekly_range
    s2 = pivot - 1.1 * prev_weekly_range / 2
    s3 = pivot - 1.1 * prev_weekly_range
    
    # Align weekly pivot levels to daily
    r2_1d = align_htf_to_ltf(prices, df_1w, r2)
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s2_1d = align_htf_to_ltf(prices, df_1w, s2)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    
    # 20-day volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_1d[i]) or np.isnan(s2_1d[i]) or np.isnan(ema20_1d[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-day average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above weekly R2 with uptrend and volume spike
            if close[i] > r2_1d[i] and close[i] > ema20_1d[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S2 with downtrend and volume spike
            elif close[i] < s2_1d[i] and close[i] < ema20_1d[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly S2 OR trend turns down
            if close[i] < s2_1d[i] or close[i] < ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly R2 OR trend turns up
            if close[i] > r2_1d[i] or close[i] > ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals