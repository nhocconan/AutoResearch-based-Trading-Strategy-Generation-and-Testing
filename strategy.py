#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_DailyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels (use previous week's OHLC)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter and volume context
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Weekly pivot from previous week's OHLC
    prev_week_high = np.roll(df_w['high'].values, 1)
    prev_week_low = np.roll(df_w['low'].values, 1)
    prev_week_close = np.roll(df_w['close'].values, 1)
    # Handle first bar
    prev_week_high[0] = df_w['high'].values[0]
    prev_week_low[0] = df_w['low'].values[0]
    prev_week_close[0] = df_w['close'].values[0]
    
    # Weekly pivot point and key levels
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    # Using weekly R1/S1 for breakout (more sensitive than R3/S3 for 6h)
    weekly_r1 = weekly_pivot + (weekly_range * 1.1 / 4)  # R1 level
    weekly_s1 = weekly_pivot - (weekly_range * 1.1 / 4)  # S1 level
    
    # Align weekly levels to 6h timeframe
    weekly_r1_6h = align_htf_to_ltf(prices, df_w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_w, weekly_s1)
    
    # Daily EMA34 trend filter
    close_d = df_d['close'].values
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_d = (close_d > ema34_d).astype(float)
    trend_d_6h = align_htf_to_ltf(prices, df_d, trend_d)
    
    # Volume spike: current volume > 2.0 * 24-period average (6h bars = 4 per day, so 24 = ~6 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or 
            np.isnan(trend_d_6h[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1 with volume spike and daily uptrend
            long_cond = (close[i] > weekly_r1_6h[i] and vol_spike[i] and trend_d_6h[i] > 0.5)
            
            # Short entry: price breaks below weekly S1 with volume spike and daily downtrend
            short_cond = (close[i] < weekly_s1_6h[i] and vol_spike[i] and trend_d_6h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 (reversal signal)
            if close[i] < weekly_s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above weekly R1 (reversal signal)
            if close[i] > weekly_r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot breakout strategy on 6h timeframe with daily trend filter and volume confirmation.
# Uses weekly R1/S1 levels for timely breakouts (more sensitive than R3/S3 for 6h). 
# Enters long when price breaks above weekly R1 with volume spike and daily uptrend (close > EMA34).
# Enters short when price breaks below weekly S1 with volume spike and daily downtrend (close < EMA34).
# Exits on reversal through S1/R1 respectively. 
# Weekly pivot provides institutional reference points, volume spike confirms institutional interest,
# daily EMA34 filters for higher probability trades. Targets 15-25 trades/year on 6h to avoid overtrading.
# Works in bull markets (breakouts with trend) and bear markets (reversals from extreme weekly levels).